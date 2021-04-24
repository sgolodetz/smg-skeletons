import cv2
import numpy as np

from typing import List, Tuple

from smg.utility import GeometryUtil

from .skeleton import Skeleton


class SkeletonUtil:
    """Utility functions related to skeletons."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def depopulate_depth_image(depth_image: np.ndarray, people_mask: np.ndarray, *, debug: bool = False) -> np.ndarray:
        """
        Make a 'depopulated' version of a depth image from which the detected people have been removed
        by removing all the pixels marked as people in a mask.

        :param depth_image:     The depth image.
        :param people_mask:     The people mask.
        :param debug:           Whether to show the people mask for debugging purposes.
        :return:                The depopulated depth image.
        """
        # : np.ndarray
        depopulated_depth_image = depth_image.copy()
        depopulated_depth_image = np.where(people_mask == 0, depopulated_depth_image, 0.0)

        # If we're debugging, show the people mask.
        if debug:
            cv2.imshow("People Mask", people_mask)
            cv2.waitKey(1)

        return depopulated_depth_image

    @staticmethod
    def depopulate_depth_image_using_3d_boxes(
        skeletons: List[Skeleton], depth_image: np.ndarray, world_from_camera: np.ndarray,
        intrinsics: Tuple[float, float, float, float], *, debug: bool = False
    ) -> np.ndarray:
        """
        Make a 'depopulated' version of a depth image from which the detected people have been removed
        by making use of person masks made from 3D bounding boxes around their skeletons.

        .. note::
            The depopulation process is intended to be deliberately conservative, in the sense that we try to do
            our best to make sure that all of the detected people have been fully removed, but we don't mind too
            much if some other parts of the depth image are removed as well.

        :param skeletons:           The 3D skeletons of the detected people.
        :param depth_image:         The depth image.
        :param world_from_camera:   The camera pose from which the depth image was captured.
        :param intrinsics:          The camera intrinsics.
        :param debug:               Whether to show the people mask for debugging purposes.
        :return:                    The depopulated depth image.
        """
        # : np.ndarray
        people_mask = SkeletonUtil.make_people_mask_from_3d_boxes(
            skeletons, depth_image, world_from_camera, intrinsics
        )
        return SkeletonUtil.depopulate_depth_image(depth_image, people_mask, debug=debug)

    @staticmethod
    def make_people_mask_from_3d_boxes(
        skeletons: List[Skeleton], depth_image: np.ndarray, world_from_camera: np.ndarray,
        intrinsics: Tuple[float, float, float, float], *, border_size: float = 0.5
    ) -> np.ndarray:
        """
        Make a binary mask for the people corresponding to 3D skeletons detected in a frame by determining
        which world-space points fall within at least one of the 3D bounding boxes around the skeletons.

        :param skeletons:           The 3D skeletons of the detected people.
        :param depth_image:         The depth image.
        :param world_from_camera:   The camera pose from which the depth image was captured.
        :param intrinsics:          The camera intrinsics.
        :param border_size:         The size of border to leave around each skeleton when making its 3D bounding box.
        :return:                    The binary people mask.
        """
        # : np.ndarray
        people_mask = np.zeros(depth_image.shape, dtype=np.uint8)

        # Compute the world-space points image.
        # : np.ndarray
        ws_points = GeometryUtil.compute_world_points_image_fast(
            depth_image, world_from_camera, intrinsics
        )

        # For each detected 3D skeleton:
        for skeleton in skeletons:
            # Make the corresponding person mask.
            # : np.ndarray
            person_mask = SkeletonUtil.make_person_mask_from_3d_box(
                skeleton, depth_image, ws_points, border_size=border_size
            )

            # Add it to the output mask.
            people_mask = np.where(person_mask == 0, people_mask, 255)

        return people_mask

    @staticmethod
    def make_person_mask_from_3d_box(skeleton: Skeleton, depth_image: np.ndarray, ws_points: np.ndarray, *,
                                     border_size: float = 0.5) -> np.ndarray:
        """
        Make a binary mask for the person corresponding to a 3D skeleton detected in a frame by determining
        which world-space points fall within a 3D bounding box around the skeleton.

        .. note::
            The binary mask we make is intended to cover the whole person, but may in practice also include
            other parts of the image. We make no promises as to the tightness of fit.

        :param skeleton:    The detected 3D skeleton.
        :param depth_image: The depth image for the frame.
        :param ws_points:   The world-space points image for the frame.
        :param border_size: The size of border to leave around the skeleton when making its 3D bounding box.
        :return:            The binary person mask.
        """
        # Compute a conservative 3D bounding box around the detected skeleton.
        min_x, min_y, min_z = np.inf, np.inf, np.inf
        max_x, max_y, max_z = -np.inf, -np.inf, -np.inf

        for _, keypoint in skeleton.keypoints.items():
            # : np.ndarray
            p = keypoint.position
            min_x, min_y, min_z = min(min_x, p[0]), min(min_y, p[1]), min(min_z, p[2])
            max_x, max_y, max_z = max(max_x, p[0]), max(max_y, p[1]), max(max_z, p[2])

        min_x -= border_size
        min_y -= border_size
        min_z -= border_size
        max_x += border_size
        max_y += border_size
        max_z += border_size

        # Make a binary mask consisting of all pixels with valid depths whose world-space points are
        # within this bounding box.
        xs, ys, zs = ws_points[:, :, 0], ws_points[:, :, 1], ws_points[:, :, 2]

        return np.where(
            (depth_image != 0.0) &
            (min_x <= xs) & (xs <= max_x) &
            (min_y <= ys) & (ys <= max_y) &
            (min_z <= zs) & (zs <= max_z),
            255, 0
        ).astype(np.uint8)
