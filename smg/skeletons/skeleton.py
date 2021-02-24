import numpy as np

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
            self.__name: str = name
            self.__position: np.ndarray = position
            self.__score: float = score

        # SPECIAL METHODS

        def __repr__(self) -> str:
            """
            Get a string representation of the keypoint.

            :return:    A string representation of the keypoint.
            """
            return f"Keypoint({repr(self.__name)}, {repr(self.__position)}, {repr(self.__score)})"

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

    # CONSTRUCTOR

    def __init__(self, keypoints: Dict[str, Keypoint], keypoint_pairs: List[Tuple[str, str]]):
        """
        Construct a skeleton.

        :param keypoints:       The keypoints that have been detected for the skeleton.
        :param keypoint_pairs:  Pairs of names denoting keypoints that should be joined by bones.
        """
        self.__keypoints: Dict[str, Skeleton.Keypoint] = keypoints

        # Filter the pairs of names, keeping only those for which both keypoints have been detected.
        self.__keypoint_pairs: List[Tuple[str, str]] = [
            (i, j) for i, j in keypoint_pairs if i in self.__keypoints and j in self.__keypoints
        ]

        # Construct a set of bounding shapes for the skeleton.
        self.__bounding_shapes: List[Shape] = []
        self.__add_bounding_shapes()

    # SPECIAL METHODS

    def __repr__(self) -> str:
        """
        Get a string representation of the skeleton.

        :return:    A string representation of the skeleton.
        """
        return f"Skeleton({repr(self.__keypoints)}, {repr(self.__keypoint_pairs)})"

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

    # PRIVATE METHODS

    def __add_bounding_shapes(self) -> None:
        self.__add_cylinder_for_bone("LElbow", "LWrist", 0.2, 0.2, top_scaling_factor=3.0)
        self.__add_cylinder_for_bone("LHip", "LKnee", 0.2, 0.15, top_scaling_factor=1.25)
        self.__add_cylinder_for_bone("LKnee", "LAnkle", 0.15, top_scaling_factor=1.5)
        self.__add_cylinder_for_bone("LShoulder", "LElbow", 0.2, top_scaling_factor=1.5)
        self.__add_cylinder_for_bone("Nose", "MidHip", 0.4)
        self.__add_cylinder_for_bone("RElbow", "RWrist", 0.2, 0.2, top_scaling_factor=3.0)
        self.__add_cylinder_for_bone("RHip", "RKnee", 0.2, 0.15, top_scaling_factor=1.25)
        self.__add_cylinder_for_bone("RKnee", "RAnkle", 0.15, top_scaling_factor=1.5)
        self.__add_cylinder_for_bone("RShoulder", "RElbow", 0.2, top_scaling_factor=1.5)

        self.__bounding_shapes.append(Sphere(
            centre=self.__pos("Nose"), radius=1.25 * np.linalg.norm(self.__pos("Nose") - self.__pos("Neck"))
        ))

    def __add_cylinder_for_bone(self, keypoint_name1: str, keypoint_name2: str, radius1: float,
                                radius2: Optional[float] = None, *, base_scaling_factor: float = 1.0,
                                top_scaling_factor: float = 1.0) -> None:
        if radius2 is None:
            radius2 = radius1

        keypoint1: Optional[Skeleton.Keypoint] = self.__keypoints.get(keypoint_name1)
        keypoint2: Optional[Skeleton.Keypoint] = self.__keypoints.get(keypoint_name2)

        if keypoint1 is not None and keypoint2 is not None:
            self.__bounding_shapes.append(Cylinder(
                base_centre=keypoint2.position + base_scaling_factor * (keypoint1.position - keypoint2.position),
                base_radius=radius1,
                top_centre=keypoint1.position + top_scaling_factor * (keypoint2.position - keypoint1.position),
                top_radius=radius2
            ))

    # def __add_sphere_for_bone(self, keypoint_name1: str, keypoint_name2: str, scaling_factor: float = 1.0) -> None:
    #     keypoint1: Optional[Skeleton.Keypoint] = self.__keypoints.get(keypoint_name1)
    #     keypoint2: Optional[Skeleton.Keypoint] = self.__keypoints.get(keypoint_name2)
    #
    #     if keypoint1 is not None and keypoint2 is not None:
    #         self.__bounding_shapes.append(Sphere(
    #             centre=(keypoint1.position + keypoint2.position) / 2,
    #             radius=scaling_factor * np.linalg.norm(keypoint2.position - keypoint1.position) / 2
    #         ))

    def __pos(self, keypoint_name: str) -> np.ndarray:
        return self.__keypoints[keypoint_name].position
