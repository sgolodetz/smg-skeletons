import numpy as np

from typing import Dict, Optional, Tuple

from .keypoint import Keypoint


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

    def __init__(self, keypoints: Dict[str, Keypoint], keypoint_name: str, other_keypoint_name: str,
                 parent_keypoint_name: Optional[str], triangle: Tuple[str, str, str], midhip_from_rest: np.ndarray):
        """
        Construct a keypoint orienter.

        :param keypoints:               A set of keypoints for a skeleton.
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
        self.__midhip_from_rest = midhip_from_rest  # type: np.ndarray
        self.__rest_from_midhip = np.linalg.inv(midhip_from_rest)  # type: np.ndarray
        self.__triangle = triangle  # type: Tuple[str, str, str]

        # Look up the various keypoints.
        self.__keypoint = keypoints[keypoint_name]  # type: Keypoint
        self.__other_keypoint = keypoints[other_keypoint_name]  # type: Keypoint

        self.__parent_keypoint = keypoints[parent_keypoint_name] \
            if parent_keypoint_name is not None else None  # type: Optional[Keypoint]

        self.__triangle_keypoints = tuple(
            [keypoints[name] for name in self.__triangle]
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
