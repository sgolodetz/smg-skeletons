import numpy as np
import vg

from typing import Dict, List, Optional, Tuple

from smg.utility import Cylinder, Shape, Sphere

from .keypoint import Keypoint
from .keypoint_orienter import KeypointOrienter
from .keypoint_util import KeypointUtil


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
            self.__midhip_from_rests = {}                               # type: Dict[str, np.ndarray]
            self.__local_keypoint_rotations = local_keypoint_rotations  # type: Dict[str, np.ndarray]

        # Otherwise, try to compute global poses and local rotations for relevant keypoints in the skeleton.
        else:
            self.__keypoint_orienters = {}  # type: Dict[str, KeypointOrienter]
            self.__try_add_keypoint_orienters()

            self.__global_keypoint_poses = {}  # type: Dict[str, np.ndarray]
            for keypoint_name, orienter in self.keypoint_orienters.items():
                self.__global_keypoint_poses[keypoint_name] = orienter.global_pose

            self.__midhip_from_rests = {
                "LElbow": np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
                "LHip": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
                "LKnee": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
                "LShoulder": np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
                "MidHip": np.eye(3),
                "Neck": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
                "RElbow": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
                "RHip": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
                "RKnee": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
                "RShoulder": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            }  # type: Dict[str, np.ndarray]

            parent_keypoints = {
                'LElbow': 'LShoulder',
                'LHip': 'MidHip',
                'LKnee': 'LHip',
                'LShoulder': 'Neck',
                'Neck': 'MidHip',
                'RElbow': 'RShoulder',
                'RHip': 'MidHip',
                'RKnee': 'RHip',
                'RShoulder': 'Neck'
            }  # type: Dict[str, str]

            self.__local_keypoint_rotations = KeypointUtil.compute_local_keypoint_rotations(
                global_keypoint_poses=self.__global_keypoint_poses,
                midhip_from_rests=self.__midhip_from_rests,
                parent_keypoints=parent_keypoints
            )

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

    @property
    def midhip_from_rests(self) -> Dict[str, np.ndarray]:
        """
        Get a 3*3 rotation matrix for each relevant keypoint specifying the transformation from the orientation
        of the keypoint to the orientation of the mid-hip keypoint when the skeleton is in its rest pose.

        :return:    A 3*3 rotation matrix for each relevant keypoint specifying the transformation from the
                    orientation of the keypoint to the orientation of the mid-hip keypoint when the skeleton
                    is in its rest pose.
        """
        return self.__midhip_from_rests

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

    def transform(self, m: np.ndarray) -> "Skeleton3D":
        """
        Make a copy of the skeleton whose keypoints have been transformed by the specified transformation.

        .. note::
            We currently calculate the global keypoint poses and local keypoint rotations for the copy
            using our internal approach, rather than transforming the ones in the existing skeleton.
            This is correct for skeletons where these were calculated internally to start with, but
            isn't for those where these were originally calculated externally and passed in. At some
            point we should fix this, but it's low priority as our internal approach is the default.

        :param m:   The transformation to apply.
        :return:    A copy of the skeleton whose keypoints have been transformed by the specified transformation.
        """
        # FIXME: We should ultimately also transform the global keypoint poses and local keypoint rotations.
        new_keypoints = {
            keypoint_name: keypoint.transform(m) for keypoint_name, keypoint in self.__keypoints.items()
        }  # type: Dict[str, Keypoint]
        return Skeleton3D(new_keypoints, self.__keypoint_pairs)

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

    def __try_add_keypoint_orienter(self, keypoint_name: str, other_keypoint_name: str,
                                    triangle: Tuple[str, str, str]) -> None:
        """
        Try to add an orienter for the specified keypoint to the skeleton.

        .. note::
            If the skeleton does not have one or more of the keypoints that the orienter needs,
            no orienter will be added.

        :param keypoint_name:       The name of the keypoint of interest.
        :param other_keypoint_name: The name of the other keypoint defining the direction of the y axis.
        :param triangle:            A triple of keypoint names specifying a triangle that is used to determine
                                    the direction of the z axis.
        """
        # Try to construct the orienter (note that this may fail if any of the keypoints it needs are missing).
        orienter = KeypointOrienter.try_make(
            self.__keypoints, keypoint_name, other_keypoint_name, triangle
        )  # type: Optional[KeypointOrienter]

        # If the orienter has been successfully constructed, add it to the skeleton.
        if orienter is not None:
            self.__keypoint_orienters[keypoint_name] = orienter

    def __try_add_keypoint_orienters(self) -> None:
        """Try to add orienters for relevant keypoints to the skeleton."""
        self.__try_add_keypoint_orienter("LElbow", "LWrist", ("LElbow", "LHip", "LWrist"))
        self.__try_add_keypoint_orienter("LHip", "LKnee", ("RHip", "LHip", "Neck"))
        self.__try_add_keypoint_orienter("LKnee", "LAnkle", ("RHip", "LHip", "Neck"))
        self.__try_add_keypoint_orienter("LShoulder", "LElbow", ("LShoulder", "LHip", "LWrist"))
        self.__try_add_keypoint_orienter("MidHip", "Neck", ("RHip", "LHip", "Neck"))
        self.__try_add_keypoint_orienter("Neck", "MidHip", ("LShoulder", "RShoulder", "MidHip"))
        self.__try_add_keypoint_orienter("RElbow", "RWrist", ("RElbow", "RWrist", "RHip"))
        self.__try_add_keypoint_orienter("RHip", "RKnee", ("RHip", "LHip", "Neck"))
        self.__try_add_keypoint_orienter("RKnee", "RAnkle", ("RHip", "LHip", "Neck"))
        self.__try_add_keypoint_orienter("RShoulder", "RElbow", ("RShoulder", "RWrist", "RHip"))
