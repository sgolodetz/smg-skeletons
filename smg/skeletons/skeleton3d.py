import numpy as np
import vg

from typing import Dict, List, Optional, Tuple

from smg.utility import Cylinder, Shape, Sphere

from .keypoint import Keypoint
from .keypoint_orienter import KeypointOrienter


class Skeleton3D:
    """A 3D skeleton."""

    # CONSTRUCTOR

    def __init__(self, keypoints: Dict[str, Keypoint], keypoint_pairs: List[Tuple[str, str]],
                 global_keypoint_poses: Optional[Dict[str, np.ndarray]] = None,
                 local_keypoint_rotations: Optional[Dict[str, np.ndarray]] = None):
        """
        Construct a skeleton.

        .. note::
            If either the global keypoint poses or the local keypoint rotations are either None or {},
            then the orientations of the keypoints will be computed internally.

        :param keypoints:                   The keypoints that have been detected for the skeleton.
        :param keypoint_pairs:              Pairs of names denoting keypoints that should be joined by bones.
        :param global_keypoint_poses:       The global poses of relevant keypoints in the skeleton (optional).
        :param local_keypoint_rotations:    The local rotations of relevant keypoints in the skeleton (optional).
        """
        self.__keypoints = keypoints  # type: Dict[str, Keypoint]

        # Filter the pairs of names, keeping only those for which both keypoints have been detected.
        self.__keypoint_pairs = [
            (i, j) for i, j in keypoint_pairs if i in self.__keypoints and j in self.__keypoints
        ]  # type: List[Tuple[str, str]]

        # Construct a set of bounding shapes for the skeleton.
        self.__bounding_shapes = []  # type: List[Shape]
        self.__add_bounding_shapes()

        # If both global keypoint poses and local keypoint rotations have been provided from outside, use them.
        if global_keypoint_poses and local_keypoint_rotations:
            self.__keypoint_orienters = {}                              # type: Dict[str, KeypointOrienter]
            self.__global_keypoint_poses = global_keypoint_poses        # type: Dict[str, np.ndarray]
            self.__local_keypoint_rotations = local_keypoint_rotations  # type: Dict[str, np.ndarray]

        # Otherwise, try to compute global poses and local rotations for relevant keypoints in the skeleton.
        else:
            self.__keypoint_orienters = {}        # type: Dict[str, KeypointOrienter]
            self.__global_keypoint_poses = {}     # type: Dict[str, np.ndarray]
            self.__local_keypoint_rotations = {}  # type: Dict[str, np.ndarray]

            self.__try_add_keypoint_orienters()
            self.__compute_global_keypoint_poses()
            self.__compute_local_keypoint_rotations()

    # SPECIAL METHODS

    def __repr__(self) -> str:
        """
        Get a string representation of the skeleton.

        :return:    A string representation of the skeleton.
        """
        return "Skeleton3D({}, {}, {}, {})".format(
            repr(self.__keypoints),
            repr(self.__keypoint_pairs),
            repr(self.__global_keypoint_poses),
            repr(self.__local_keypoint_rotations)
        )

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
    def compute_local_keypoint_rotations(*, global_keypoint_poses: Dict[str, np.ndarray],
                                         keypoint_parents: Dict[str, str],
                                         midhip_from_rests: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute the local rotations for relevant keypoints in a skeleton.

        .. note::
            This is needed for avatar driving.

        :param global_keypoint_poses:   TODO
        :param keypoint_parents:        TODO
        :param midhip_from_rests:       TODO
        :return:                        TODO
        """
        local_keypoint_rotations = {}  # type: Dict[str, np.ndarray]

        # For each keypoint for which a global pose is available:
        for current_name, world_from_current in global_keypoint_poses.items():
            world_from_parent = None         # type: Optional[np.ndarray]
            midhip_from_rest_current = None  # type: Optional[np.ndarray]
            midhip_from_rest_parent = None   # type: Optional[np.ndarray]

            # If it has a parent in the skeleton:
            parent_name = keypoint_parents.get(current_name)  # type: Optional[str]
            if parent_name is not None:
                # Try to get the relevant transformations for both the keypoint and its parent.
                world_from_parent = global_keypoint_poses.get(parent_name)
                midhip_from_rest_current = midhip_from_rests.get(current_name)
                midhip_from_rest_parent = midhip_from_rests.get(parent_name)

            # If they're all available, use them to compute the local rotation for the keypoint.
            if all([x is not None for x in [world_from_parent, midhip_from_rest_current, midhip_from_rest_parent]]):
                # Derivation:
                #
                # m0Tp0 * wTp^-1 * wTc * m0Tc0^-1
                # = m0Tp0 * pTw * wTc * c0Tm0
                # = m0Tp0 * (pTp0 * p0Tm0 * m0Tw) * (wTm0 * m0Tc0 * c0Tc) * c0Tm0
                # = (m0Tp0 * pTp0 * p0Tm0) * (m0Tc0 * c0Tc * c0Tm0)
                local_keypoint_rotations[current_name] = \
                    midhip_from_rest_parent @ \
                    np.linalg.inv(world_from_parent[0:3, 0:3]) @ \
                    world_from_current[0:3, 0:3] @ \
                    np.linalg.inv(midhip_from_rest_current)

            # Otherwise, set the local rotation for the keypoint to the identity matrix (as a default).
            else:
                local_keypoint_rotations[current_name] = np.eye(3)

        return local_keypoint_rotations

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

    # PUBLIC METHODS

    def make_bare(self) -> "Skeleton3D":
        """
        Remove the keypoint orientation information from the skeleton.

        .. note::
            This is useful when we want to transmit a skeleton across a network but have its keypoint orientation
            information be recomputed at the remote end. (In particular, if no keypoint orientation information is
            present when the skeleton reaches the remote end, it will be recomputed.)

        :return:    The skeleton, after removing its keypoint orientation information.
        """
        self.__keypoint_orienters = {}
        self.__global_keypoint_poses = {}
        self.__local_keypoint_rotations = {}
        return self

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
        """Compute the global poses for relevant keypoints (as keypoint space to world space transformations)."""
        # TODO: Tidy this up.
        for keypoint_name, orienter in self.keypoint_orienters.items():
            self.__global_keypoint_poses[keypoint_name] = orienter.global_pose

    def __compute_local_keypoint_rotations(self) -> None:
        """Compute the local rotations for relevant keypoints in the skeleton (needed for avatar driving)."""
        keypoint_parents = {}  # type: Dict[str, str]
        midhip_from_rests = {}  # type: Dict[str, np.ndarray]
        for keypoint_name, orienter in self.keypoint_orienters.items():
            midhip_from_rests[keypoint_name] = orienter.midhip_from_rest
            if orienter.parent_keypoint is not None:
                keypoint_parents[keypoint_name] = orienter.parent_keypoint.name

        self.__local_keypoint_rotations = Skeleton3D.compute_local_keypoint_rotations(
            global_keypoint_poses=self.__global_keypoint_poses,
            keypoint_parents=keypoint_parents,
            midhip_from_rests=midhip_from_rests
        )

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
        self.__keypoint_orienters[keypoint_name] = KeypointOrienter(
            self.__keypoints, keypoint_name, other_keypoint_name, parent_keypoint_name, triangle, midhip_from_rest
        )

    def __try_add_keypoint_orienters(self) -> None:
        """Try to add orienters for relevant keypoints to the skeleton."""
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
