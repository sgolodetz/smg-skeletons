import numpy as np
import vg

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
        self.__midhip_from_rest = midhip_from_rest                 # type: np.ndarray
        self.__rest_from_midhip = np.linalg.inv(midhip_from_rest)  # type: np.ndarray
        self.__triangle = triangle                                 # type: Tuple[str, str, str]

        # Look up the various keypoints.
        self.__keypoint = keypoints[keypoint_name]              # type: Keypoint
        self.__other_keypoint = keypoints[other_keypoint_name]  # type: Keypoint

        self.__parent_keypoint = keypoints[parent_keypoint_name] \
            if parent_keypoint_name is not None else None  # type: Optional[Keypoint]

        self.__triangle_keypoints = tuple(
            [keypoints[name] for name in self.__triangle]
        )  # type: Tuple[Keypoint, Keypoint, Keypoint]

        # Compute the global pose of the keypoint.
        self.__global_pose = self.__compute_global_pose()  # type: np.ndarray

    # PUBLIC STATIC METHODS

    @staticmethod
    def try_make(keypoints: Dict[str, Keypoint], keypoint_name: str, other_keypoint_name: str,
                 parent_keypoint_name: Optional[str], triangle: Tuple[str, str, str],
                 midhip_from_rest: np.ndarray) -> Optional["KeypointOrienter"]:
        """
        Try to make an orienter for the specified keypoint in a skeleton.

        .. note::
            If the skeleton does not have one or more of the keypoints that the orienter needs,
            no orienter will be constructed.

        :param keypoints:               TODO
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
            if keypoints.get(name) is None:
                return None

        if parent_keypoint_name is not None and keypoints.get(parent_keypoint_name) is None:
            return None

        # Otherwise, construct and return the orienter.
        return KeypointOrienter(
            keypoints, keypoint_name, other_keypoint_name, parent_keypoint_name, triangle, midhip_from_rest
        )

    # PROPERTIES

    @property
    def global_pose(self) -> np.ndarray:
        """
        TODO

        :return:    TODO
        """
        return self.__global_pose

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

    # PRIVATE METHODS

    def __compute_global_pose(self) -> np.ndarray:
        """
        TODO

        :return:    TODO
        """
        v0, v1, v2 = self.triangle_vertices

        # Compute the coordinate axes for the keypoint.
        y = vg.normalize(self.other_keypoint.position - self.keypoint.position)  # type: np.ndarray
        n = vg.normalize(np.cross(v1 - v0, v2 - v0))                             # type: np.ndarray
        x = vg.normalize(np.cross(y, n))                                         # type: np.ndarray
        z = vg.normalize(np.cross(x, y))                                         # type: np.ndarray

        # Thence construct the global pose for the keypoint.
        w_t_c = np.eye(4)  # type: np.ndarray
        w_t_c[0:3, 0:3] = np.column_stack([x, y, z])
        w_t_c[0:3, 3] = self.keypoint.position

        return w_t_c
