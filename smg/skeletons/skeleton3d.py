import numpy as np
import vg

from typing import Dict, List, Optional, Tuple

from smg.utility import Cylinder, Shape, Sphere

from .keypoint import Keypoint


class Skeleton3D:
    """A 3D skeleton."""

    # NESTED TYPES

    class KeypointOrienter:
        """
        A keypoint orienter.

        As its name suggests, a keypoint orienter is used to specify the orientation of a keypoint (i.e. find a set
        of xyz coordinate axes for the keypoint). The y axis is defined so as to point from the keypoint of interest
        to a separate keypoint. The z axis is defined to be perpendicular to the y axis and as much as possible in
        the direction of the normal of a triangle whose vertices are three of the keypoints. In practice, if the
        unit normal of the triangle is n, we compute x = cross(y, n) and z = cross(x, y). This means that n can't
        point in the same direction as y: we ensure this by construction.
        """

        # CONSTRUCTOR

        def __init__(self, skeleton: "Skeleton3D", keypoint_name: str, other_keypoint_name: str,
                     parent_keypoint_name: Optional[str], triangle: Tuple[str, str, str], midhip_from_rest: np.ndarray):
            """
            Construct a keypoint orienter.

            :param skeleton:                The skeleton containing the keypoint of interest.
            :param keypoint_name:           The name of the keypoint of interest.
            :param other_keypoint_name:     The name of the other keypoint defining the direction of the y axis.
            :param parent_keypoint_name:    The name of the parent keypoint in the skeleton (if any) of the keypoint
                                            of interest.
            :param triangle:                A triple of keypoint names specifying a triangle that is used to determine
                                            the direction of the z axis.
            :param midhip_from_rest:        A 3*3 rotation matrix specifying the transformation from the orientation
                                            of the keypoint of interest to the orientation of the mid-hip keypoint
                                            when the skeleton is in its rest pose (a T shape, with arms outstretched).
            """
            self.__midhip_from_rest = midhip_from_rest                 # type: np.ndarray
            self.__rest_from_midhip = np.linalg.inv(midhip_from_rest)  # type: np.ndarray
            self.__skeleton = skeleton                                 # type: Skeleton3D
            self.__triangle = triangle                                 # type: Tuple[str, str, str]

            # Look up the various keypoints.
            self.__keypoint = self.__skeleton.keypoints[keypoint_name]              # type: Keypoint
            self.__other_keypoint = self.__skeleton.keypoints[other_keypoint_name]  # type: Keypoint

            self.__parent_keypoint = self.__skeleton.keypoints[parent_keypoint_name] \
                if parent_keypoint_name is not None else None  # type: Optional[Keypoint]

            self.__triangle_keypoints = tuple(
                [self.__skeleton.keypoints[name] for name in self.__triangle]
            )  # type: Tuple[Keypoint, Keypoint, Keypoint]

        # PROPERTIES

        @property
        def keypoint(self) -> Keypoint:
            """
            Get the keypoint of interest.

            :return:    The keypoint of interest.
            """
            return self.__keypoint

        @property
        def midhip_from_rest(self) -> np.ndarray:
            """
            Get a 3*3 rotation matrix specifying the transformation from the orientation of the keypoint of interest
            to the orientation of the mid-hip keypoint when the skeleton is in its rest pose.

            :return:    The 3*3 rotation matrix as specified.
            """
            return self.__midhip_from_rest

        @property
        def other_keypoint(self) -> Keypoint:
            """
            Get the other keypoint defining the direction of the y axis.

            :return:    The other keypoint defining the direction of the y axis.
            """
            return self.__other_keypoint

        @property
        def parent_keypoint(self) -> Optional[Keypoint]:
            """
            Get the parent keypoint in the skeleton (if any) of the keypoint of interest.

            :return:    The parent keypoint in the skeleton (if any) of the keypoint of interest.
            """
            return self.__parent_keypoint

        @property
        def rest_from_midhip(self) -> np.ndarray:
            """
            Get a 3*3 rotation matrix specifying the transformation from the orientation of the mid-hip keypoint
            to the orientation of the keypoint of interest when the skeleton is in its rest pose.

            :return:    The 3*3 rotation matrix as specified.
            """
            return self.__rest_from_midhip

        @property
        def triangle_vertices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Get the vertices of the triangle that is used to determine the direction of the z axis.

            .. note::
                The winding order of the triangle is counter-clockwise.

            :return:    The vertices of the triangle that is used to determine the direction of the z axis.
            """
            # noinspection PyTypeChecker
            return tuple([keypoint.position for keypoint in self.__triangle_keypoints])

    # CONSTRUCTOR

    def __init__(self, keypoints: Dict[str, Keypoint], keypoint_pairs: List[Tuple[str, str]]):
        """
        Construct a skeleton.

        :param keypoints:       The keypoints that have been detected for the skeleton.
        :param keypoint_pairs:  Pairs of names denoting keypoints that should be joined by bones.
        """
        self.__keypoints = keypoints  # type: Dict[str, Keypoint]

        # Filter the pairs of names, keeping only those for which both keypoints have been detected.
        self.__keypoint_pairs = [
            (i, j) for i, j in keypoint_pairs if i in self.__keypoints and j in self.__keypoints
        ]  # type: List[Tuple[str, str]]

        # Construct a set of bounding shapes for the skeleton.
        self.__bounding_shapes = []  # type: List[Shape]
        self.__add_bounding_shapes()

        # Try to add orienters for relevant keypoints in the skeleton.
        self.__keypoint_orienters = {}  # type: Dict[str, Skeleton3D.KeypointOrienter]
        self.__try_add_keypoint_orienters()

        # Compute the global poses for the keypoints that have orienters.
        self.__global_keypoint_poses = {}  # type: Dict[str, np.ndarray]
        self.__compute_global_keypoint_poses()

        # Compute the local rotations for the keypoints that have orienters.
        self.__local_keypoint_rotations = {}  # type: Dict[str, np.ndarray]
        self.__compute_local_keypoint_rotations()

    # SPECIAL METHODS

    def __repr__(self) -> str:
        """
        Get a string representation of the skeleton.

        :return:    A string representation of the skeleton.
        """
        return "Skeleton3D({}, {})".format(repr(self.__keypoints), repr(self.__keypoint_pairs))

    # PROPERTIES

    @property
    def bones(self) -> List[Tuple[Keypoint, Keypoint]]:
        """
        Get the bones of the skeleton.

        :return:    The bones of the skeleton, as a list of detected keypoint pairs.
        """
        return [(self.__keypoints[i], self.__keypoints[j]) for i, j in self.__keypoint_pairs]

    @property
    def bounding_shapes(self) -> List[Shape]:
        """
        Get the bounding shapes for the skeleton.

        :return:    The bounding shapes for the skeleton.
        """
        return self.__bounding_shapes

    @property
    def global_keypoint_poses(self) -> Dict[str, np.ndarray]:
        """
        Get the global poses for any keypoints for which they have been calculated.

        :return:    The global poses for any keypoints for which they have been calculated.
        """
        return self.__global_keypoint_poses

    @property
    def keypoint_orienters(self) -> Dict[str, KeypointOrienter]:
        """
        Get the skeleton's keypoint orienters.

        :return:    The skeleton's keypoint orienters.
        """
        return self.__keypoint_orienters

    @property
    def keypoints(self) -> Dict[str, Keypoint]:
        """
        Get the detected keypoints of the skeleton.

        :return:    The detected keypoints of the skeleton, as a keypoint name -> keypoint map.
        """
        return self.__keypoints

    @property
    def local_keypoint_rotations(self) -> Dict[str, np.ndarray]:
        """
        Get the local rotations for any keypoints for which they have been calculated.

        :return:    The local rotations for any keypoints for which they have been calculated.
        """
        return self.__local_keypoint_rotations

    # PUBLIC STATIC METHODS

    @staticmethod
    def make_bone_key(keypoint1: Keypoint, keypoint2: Keypoint) -> Tuple[str, str]:
        """
        Make a key that can be used to look up a bone in a dictionary.

        :param keypoint1:   The keypoint at one end of the bone.
        :param keypoint2:   The keypoint at the other end of the bone.
        :return:            The key for the bone.
        """
        # noinspection PyTypeChecker
        return tuple(sorted([keypoint1.name, keypoint2.name]))

    # PRIVATE METHODS

    def __add_bounding_shapes(self) -> None:
        """Add a set of bounding shapes to the skeleton."""
        self.__add_cylinder_for_bone("LElbow", "LWrist", 0.2, 0.2, top_stretch=3.0)
        self.__add_cylinder_for_bone("LHip", "LKnee", 0.2, 0.15, top_stretch=1.25)
        self.__add_cylinder_for_bone("LKnee", "LAnkle", 0.15, top_stretch=1.5)
        self.__add_cylinder_for_bone("LShoulder", "LElbow", 0.2, top_stretch=1.5)
        self.__add_cylinder_for_bone("Nose", "MidHip", 0.4)
        self.__add_cylinder_for_bone("RElbow", "RWrist", 0.2, 0.2, top_stretch=3.0)
        self.__add_cylinder_for_bone("RHip", "RKnee", 0.2, 0.15, top_stretch=1.25)
        self.__add_cylinder_for_bone("RKnee", "RAnkle", 0.15, top_stretch=1.5)
        self.__add_cylinder_for_bone("RShoulder", "RElbow", 0.2, top_stretch=1.5)

        neck_keypoint = self.__keypoints.get("Neck")  # type: Optional[Keypoint]
        nose_keypoint = self.__keypoints.get("Nose")  # type: Optional[Keypoint]
        if neck_keypoint is not None and nose_keypoint is not None:
            neck_pos, nose_pos = neck_keypoint.position, nose_keypoint.position
            self.__bounding_shapes.append(Sphere(centre=nose_pos, radius=1.25 * np.linalg.norm(nose_pos - neck_pos)))

    def __add_cylinder_for_bone(self, base_keypoint_name: str, top_keypoint_name: str, base_radius: float,
                                top_radius: Optional[float] = None, *, base_stretch: float = 1.0,
                                top_stretch: float = 1.0) -> None:
        """
        Add a bounding cylinder for the specified bone to the skeleton.

        :param base_keypoint_name:  The name of the keypoint associated with the cylinder's base.
        :param top_keypoint_name:   The name of the keypoint associated with the cylinder's top.
        :param base_radius:         The radius of the cylinder's base.
        :param top_radius:          The radius of the cylinder's top.
        :param base_stretch:        The factor by which to stretch the base of the cylinder.
        :param top_stretch:         The factor by which to stretch the top of the cylinder.
        """
        if top_radius is None:
            top_radius = base_radius

        base_keypoint = self.__keypoints.get(base_keypoint_name)  # type: Optional[Keypoint]
        top_keypoint = self.__keypoints.get(top_keypoint_name)    # type: Optional[Keypoint]

        if base_keypoint is not None and top_keypoint is not None:
            self.__bounding_shapes.append(Cylinder(
                base_centre=top_keypoint.position + base_stretch * (base_keypoint.position - top_keypoint.position),
                base_radius=base_radius,
                top_centre=base_keypoint.position + top_stretch * (top_keypoint.position - base_keypoint.position),
                top_radius=top_radius
            ))

    def __compute_global_keypoint_poses(self) -> None:
        """Compute the global poses for the keypoints (as keypoint space to world space transformations)."""
        for keypoint_name, orienter in self.keypoint_orienters.items():
            v0, v1, v2 = orienter.triangle_vertices

            # Compute the coordinate axes for the keypoint.
            y = vg.normalize(orienter.other_keypoint.position - orienter.keypoint.position)  # type: np.ndarray
            n = vg.normalize(np.cross(v1 - v0, v2 - v0))                                     # type: np.ndarray
            x = vg.normalize(np.cross(y, n))                                                 # type: np.ndarray
            z = vg.normalize(np.cross(x, y))                                                 # type: np.ndarray

            # Thence construct the global pose for the keypoint.
            w_t_c = np.eye(4)                                                                # type: np.ndarray
            w_t_c[0:3, 0:3] = np.column_stack([x, y, z])
            w_t_c[0:3, 3] = orienter.keypoint.position
            self.__global_keypoint_poses[keypoint_name] = w_t_c

    def __compute_local_keypoint_rotations(self) -> None:
        """Compute the local rotations for the keypoints (needed for avatar driving)."""
        # For each keypoint with an orienter:
        for keypoint_name, orienter in self.keypoint_orienters.items():
            parent_orienter = None     # type: Optional[Skeleton3D.KeypointOrienter]
            world_from_parent = None   # type: Optional[np.ndarray]
            world_from_current = None  # type: Optional[np.ndarray]

            # If it has a parent in the skeleton:
            if orienter.parent_keypoint is not None:
                # Try to get the parent's orienter, as well as the global poses of the keypoint and its parent.
                parent_keypoint_name = orienter.parent_keypoint.name  # type: str
                parent_orienter = self.keypoint_orienters.get(parent_keypoint_name)
                world_from_current = self.__global_keypoint_poses.get(keypoint_name)
                world_from_parent = self.__global_keypoint_poses.get(parent_keypoint_name)

            # If all of those are available, use them to compute the local rotation for the keypoint.
            if parent_orienter is not None and world_from_parent is not None and world_from_current is not None:
                # Derivation:
                #
                # m0Tp0 * wTp^-1 * wTc * m0Tc0^-1
                # = m0Tp0 * pTw * wTc * c0Tm0
                # = m0Tp0 * (pTp0 * p0Tm0 * m0Tw) * (wTm0 * m0Tc0 * c0Tc) * c0Tm0
                # = (m0Tp0 * pTp0 * p0Tm0) * (m0Tc0 * c0Tc * c0Tm0)
                self.__local_keypoint_rotations[keypoint_name] = \
                    parent_orienter.midhip_from_rest @ \
                    np.linalg.inv(world_from_parent[0:3, 0:3]) @ \
                    world_from_current[0:3, 0:3] @ \
                    orienter.rest_from_midhip

            # Otherwise, set the local rotation for the keypoint to the identity matrix (as a default).
            else:
                self.__local_keypoint_rotations[keypoint_name] = np.eye(3)

    def __try_add_keypoint_orienter(self, keypoint_name: str, other_keypoint_name: str,
                                    parent_keypoint_name: Optional[str], triangle: Tuple[str, str, str],
                                    midhip_from_rest: np.ndarray) -> None:
        """
        Try to add an orienter for the specified keypoint to the skeleton.

        .. note::
            If the skeleton does not have one or more of the keypoints that the orienter needs,
            no orienter will be added.

        :param keypoint_name:           The name of the keypoint of interest.
        :param other_keypoint_name:     The name of the other keypoint defining the direction of the y axis.
        :param parent_keypoint_name:    The name of the parent keypoint in the skeleton (if any) of the keypoint
                                        of interest.
        :param triangle:                A triple of keypoint names specifying a triangle that is used to determine
                                        the direction of the z axis.
        :param midhip_from_rest:        A 3*3 rotation matrix specifying the transformation from the orientation
                                        of the keypoint of interest to the orientation of the mid-hip keypoint
                                        when the skeleton is in its rest pose (a T shape, with arms outstretched).
        """
        # If any of the keypoints needed by the orienter are missing, early out.
        for name in [keypoint_name, other_keypoint_name, *triangle]:
            if self.__keypoints.get(name) is None:
                return

        if parent_keypoint_name is not None and self.__keypoints.get(parent_keypoint_name) is None:
            return

        # Otherwise, construct the orienter and add it to the skeleton.
        self.__keypoint_orienters[keypoint_name] = Skeleton3D.KeypointOrienter(
            self, keypoint_name, other_keypoint_name, parent_keypoint_name, triangle, midhip_from_rest
        )

    def __try_add_keypoint_orienters(self) -> None:
        """Try to add the orienters for the keypoints to the skeleton."""
        self.__try_add_keypoint_orienter(
            "LElbow", "LWrist", "LShoulder", ("LElbow", "LHip", "LWrist"),
            np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        )
        self.__try_add_keypoint_orienter(
            "LHip", "LKnee", "MidHip", ("RHip", "LHip", "Neck"),
            np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        )
        self.__try_add_keypoint_orienter(
            "LKnee", "LAnkle", "LHip", ("RHip", "LHip", "Neck"),
            np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        )
        self.__try_add_keypoint_orienter(
            "LShoulder", "LElbow", "Neck", ("LShoulder", "LHip", "LWrist"),
            np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        )
        self.__try_add_keypoint_orienter(
            "MidHip", "Neck", None, ("RHip", "LHip", "Neck"),
            np.eye(3)
        )
        self.__try_add_keypoint_orienter(
            "Neck", "MidHip", "MidHip", ("LShoulder", "RShoulder", "MidHip"),
            np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        )
        self.__try_add_keypoint_orienter(
            "RElbow", "RWrist", "RShoulder", ("RElbow", "RWrist", "RHip"),
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        )
        self.__try_add_keypoint_orienter(
            "RHip", "RKnee", "MidHip", ("RHip", "LHip", "Neck"),
            np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        )
        self.__try_add_keypoint_orienter(
            "RKnee", "RAnkle", "RHip", ("RHip", "LHip", "Neck"),
            np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        )
        self.__try_add_keypoint_orienter(
            "RShoulder", "RElbow", "Neck", ("RShoulder", "RWrist", "RHip"),
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        )
