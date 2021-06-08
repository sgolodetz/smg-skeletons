import numpy as np
import vg

from typing import Dict, List, Optional, Tuple

from smg.utility import Cylinder, Shape, Sphere


class Skeleton:
    """A skeleton."""

    # NESTED TYPES

    class Keypoint:
        """A keypoint (either 2D or 3D)."""

        # CONSTRUCTOR

        def __init__(self, name: str, position: np.ndarray, score: float = 1.0):
            """
            Construct a keypoint.

            :param name:        The name of the keypoint.
            :param position:    The position of the keypoint (either 2D or 3D).
            :param score:       The score assigned to the keypoint (a float in [0,1]).
            """
            self.__name = name          # type: str
            self.__position = position  # type: np.ndarray
            self.__score = score        # type: float

        # SPECIAL METHODS

        def __repr__(self) -> str:
            """
            Get a string representation of the keypoint.

            :return:    A string representation of the keypoint.
            """
            return "Keypoint({}, {}, {})".format(repr(self.__name), repr(self.__position), repr(self.__score))

        # PROPERTIES

        @property
        def name(self) -> str:
            """
            Get the name of the keypoint.

            :return:    The name of the keypoint.
            """
            return self.__name

        @property
        def position(self) -> np.ndarray:
            """
            Get the position of the keypoint.

            :return:    The position of the keypoint.
            """
            return self.__position

        @property
        def score(self) -> float:
            """
            Get the score assigned to the keypoint.

            :return:    The score assigned to the keypoint (a float in [0,1]).
            """
            return self.__score

    class KeypointOrienter:
        """TODO"""

        # CONSTRUCTOR

        def __init__(self, skeleton: "Skeleton", primary_keypoint_name: str, secondary_keypoint_name: str,
                     parent_keypoint_name: Optional[str], triangle: Tuple[str, str, str], midhip_from_rest: np.ndarray):
            self.__skeleton = skeleton                                                      # type: Skeleton
            self.__primary_keypoint = self.__skeleton.keypoints[primary_keypoint_name]      # type: Skeleton.Keypoint
            self.__secondary_keypoint = self.__skeleton.keypoints[secondary_keypoint_name]  # type: Skeleton.Keypoint
            self.__parent_keypoint = self.__skeleton.keypoints[parent_keypoint_name] \
                if parent_keypoint_name is not None else None  # type: Optional[Skeleton.Keypoint]
            self.__triangle = triangle                                                      # type: Tuple[str, str, str]
            self.__triangle_keypoints = tuple(
                [self.__skeleton.keypoints[name] for name in self.__triangle]
            )  # type: Tuple[Skeleton.Keypoint, Skeleton.Keypoint, Skeleton.Keypoint]
            self.__midhip_from_rest = midhip_from_rest                                      # type: np.ndarray

            self.__w_t_c = self.__calculate_w_t_c()                                         # type: np.ndarray

        # PROPERTIES

        @property
        def midhip_from_rest(self) -> np.ndarray:
            return self.__midhip_from_rest

        @property
        def parent_keypoint(self) -> Optional[np.ndarray]:
            return self.__parent_keypoint

        @property
        def primary_keypoint(self) -> "Skeleton.Keypoint":
            return self.__primary_keypoint

        @property
        def secondary_keypoint(self) -> "Skeleton.Keypoint":
            return self.__secondary_keypoint

        @property
        def triangle_vertices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            # noinspection PyTypeChecker
            return tuple([keypoint.position for keypoint in self.__triangle_keypoints])

        @property
        def w_t_c(self) -> np.ndarray:
            return self.__w_t_c

        # PRIVATE METHODS

        def __calculate_w_t_c(self) -> np.ndarray:
            w_t_c = np.eye(4)                                                                    # type: np.ndarray
            v0, v1, v2 = self.triangle_vertices
            z = vg.normalize(np.cross(v1 - v0, v2 - v0))                                         # type: np.ndarray
            y = vg.normalize(self.secondary_keypoint.position - self.primary_keypoint.position)  # type: np.ndarray
            x = vg.normalize(np.cross(z, y))                                                     # type: np.ndarray
            w_t_c[0:3, 0:3] = np.column_stack([x, y, z])
            w_t_c[0:3, 3] = self.__primary_keypoint.position
            return w_t_c

    # CONSTRUCTOR

    def __init__(self, keypoints: Dict[str, Keypoint], keypoint_pairs: List[Tuple[str, str]]):
        """
        Construct a skeleton.

        :param keypoints:       The keypoints that have been detected for the skeleton.
        :param keypoint_pairs:  Pairs of names denoting keypoints that should be joined by bones.
        """
        self.__keypoints = keypoints  # type: Dict[str, Skeleton.Keypoint]

        # Filter the pairs of names, keeping only those for which both keypoints have been detected.
        self.__keypoint_pairs = [
            (i, j) for i, j in keypoint_pairs if i in self.__keypoints and j in self.__keypoints
        ]  # type: List[Tuple[str, str]]

        # Construct a set of bounding shapes for the skeleton.
        self.__bounding_shapes = []  # type: List[Shape]
        self.__add_bounding_shapes()

        # Construct a set of keypoint orienters for the skeleton.
        self.__keypoint_orienters = {}  # type: Dict[str, Skeleton.KeypointOrienter]
        self.__add_keypoint_orienters()

        # TODO
        self.__joint_rotations = {}      # type: Dict[str, np.ndarray]
        self.__joint_rel_rotations = {}  # type: Dict[str, np.ndarray]

    # SPECIAL METHODS

    def __repr__(self) -> str:
        """
        Get a string representation of the skeleton.

        :return:    A string representation of the skeleton.
        """
        return "Skeleton({}, {})".format(repr(self.__keypoints), repr(self.__keypoint_pairs))

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
    def keypoint_orienters(self) -> Dict[str, KeypointOrienter]:
        return self.__keypoint_orienters

    @property
    def keypoints(self) -> Dict[str, Keypoint]:
        """
        Get the detected keypoints of the skeleton.

        :return:    The detected keypoints of the skeleton, as a keypoint name -> keypoint map.
        """
        return self.__keypoints

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

    def compute_joint_rotations(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        self.__joint_rotations.clear()
        self.__joint_rel_rotations.clear()
        for keypoint_name in self.keypoint_orienters:
            self.__compute_joint_rotations(keypoint_name)
        return self.__joint_rotations, self.__joint_rel_rotations

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

        neck_keypoint = self.__keypoints.get("Neck")  # type: Optional[Skeleton.Keypoint]
        nose_keypoint = self.__keypoints.get("Nose")  # type: Optional[Skeleton.Keypoint]
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

        base_keypoint = self.__keypoints.get(base_keypoint_name)  # type: Optional[Skeleton.Keypoint]
        top_keypoint = self.__keypoints.get(top_keypoint_name)    # type: Optional[Skeleton.Keypoint]

        if base_keypoint is not None and top_keypoint is not None:
            self.__bounding_shapes.append(Cylinder(
                base_centre=top_keypoint.position + base_stretch * (base_keypoint.position - top_keypoint.position),
                base_radius=base_radius,
                top_centre=base_keypoint.position + top_stretch * (top_keypoint.position - base_keypoint.position),
                top_radius=top_radius
            ))

    def __add_keypoint_orienters(self) -> None:
        self.__try_add_keypoint_orienter(
            "LElbow", "LWrist", "LShoulder", ("LElbow", "LHip", "LWrist"),
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        )
        self.__try_add_keypoint_orienter(
            "LHip", "LKnee", "MidHip", ("LHip", "MidHip", "LKnee"),
            np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        )
        self.__try_add_keypoint_orienter(
            "LKnee", "LAnkle", "LHip", ("LKnee", "MidHip", "LAnkle"),
            np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        )
        self.__try_add_keypoint_orienter(
            "LShoulder", "LElbow", "Neck", ("LShoulder", "Neck", "LElbow"),
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
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
            np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        )
        self.__try_add_keypoint_orienter(
            "RHip", "RKnee", "MidHip", ("RHip", "RKnee", "MidHip"),
            np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        )
        self.__try_add_keypoint_orienter(
            "RKnee", "RAnkle", "RHip", ("RKnee", "RAnkle", "MidHip"),
            np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        )
        self.__try_add_keypoint_orienter(
            "RShoulder", "RElbow", "Neck", ("RShoulder", "RElbow", "Neck"),
            np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        )

    def __compute_joint_rotations(self, keypoint_name: str) -> None:
        if self.__joint_rotations.get(keypoint_name) is not None:
            return

        orienter: Skeleton.KeypointOrienter = self.keypoint_orienters[keypoint_name]
        world_from_rest: np.ndarray = self.keypoint_orienters["MidHip"].w_t_c[0:3, 0:3] @ orienter.midhip_from_rest
        current_from_rest: np.ndarray = np.linalg.inv(orienter.w_t_c[0:3, 0:3]) @ world_from_rest

        if orienter.parent_keypoint is not None:
            self.__compute_joint_rotations(orienter.parent_keypoint.name)
            parent_from_rest: np.ndarray = self.__joint_rotations.get(orienter.parent_keypoint.name)
        else:
            parent_from_rest: np.ndarray = np.eye(3)

        current_from_parent: np.ndarray = current_from_rest @ np.linalg.inv(parent_from_rest)
        self.__joint_rotations[keypoint_name] = current_from_rest
        self.__joint_rel_rotations[keypoint_name] = current_from_parent

    def __try_add_keypoint_orienter(self, primary_keypoint_name: str, secondary_keypoint_name: str,
                                    parent_keypoint_name: Optional[str], triangle: Tuple[str, str, str],
                                    rest_orientation: Optional[np.ndarray] = None) \
            -> None:
        for name in [primary_keypoint_name, secondary_keypoint_name, *triangle]:
            if self.__keypoints.get(name) is None:
                return

        if rest_orientation is None:
            rest_orientation = np.eye(3)

        self.__keypoint_orienters[primary_keypoint_name] = Skeleton.KeypointOrienter(
            self, primary_keypoint_name, secondary_keypoint_name, parent_keypoint_name, triangle, rest_orientation
        )
