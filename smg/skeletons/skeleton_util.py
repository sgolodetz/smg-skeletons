import cv2
import numpy as np

from typing import List, Tuple

from smg.utility import GeometryUtil

from .skeleton import Skeleton


class SkeletonUtil:
    """Utility functions related to skeletons."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def depopulate_depth_image(skeletons: List[Skeleton], depth_image: np.ndarray, world_from_camera: np.ndarray,
                               intrinsics: Tuple[float, float, float, float], *, debug: bool = False) -> np.ndarray:
        """
        Make a 'depopulated' version of a depth image from which the detected people have been removed.

        .. note::
            At most one person mask will be shown for debugging purposes.
        .. note::
            The depopulation process is intended to be deliberately conservative, in the sense that we try to do
            our best to make sure that all of the detected people have been fully removed, but we don't mind too
            much if some other parts of the depth image are removed as well.

        :param skeletons:           The 3D skeletons of the detected people.
        :param depth_image:         The depth image.
        :param world_from_camera:   The camera pose from which the depth image was captured.
        :param intrinsics:          The camera intrinsics.
        :param debug:               Whether to show a person mask for debugging purposes (see note).
        :return:                    The depopulated depth image.
        """
        depopulated_depth_image: np.ndarray = depth_image.copy()

        # Compute the world-space points image.
        ws_points: np.ndarray = GeometryUtil.compute_world_points_image_fast(
            depth_image, world_from_camera, intrinsics
        )

        # For each detected 3D skeleton:
        for skeleton in skeletons:
            # Make the corresponding person mask.
            person_mask: np.ndarray = SkeletonUtil.make_person_mask(skeleton, depth_image, ws_points)

            # Remove it from the output image.
            depopulated_depth_image = np.where(person_mask == 0, depopulated_depth_image, 0.0)

            # If we're debugging, show the person mask.
            if debug:
                cv2.imshow("Person Mask", person_mask)

        return depopulated_depth_image

    @staticmethod
    def make_person_mask(skeleton: Skeleton, depth_image: np.ndarray, ws_points: np.ndarray) -> np.ndarray:
        """
        Make a binary mask for the person corresponding to a 3D skeleton detected in a frame.

        .. note::
            The binary mask we make is intended to cover the whole person, but may in practice also include
            other parts of the image. Currently, we make no promises as to the tightness of fit.

        :param skeleton:    The detected 3D skeleton.
        :param depth_image: The depth image for the frame.
        :param ws_points:   The world-space points image for the frame.
        :return:            The binary person mask.
        """
        # Compute a (very) conservative 3D bounding box around the detected skeleton.
        min_x, min_y, min_z = np.inf, np.inf, np.inf
        max_x, max_y, max_z = -np.inf, -np.inf, -np.inf

        for _, keypoint in skeleton.keypoints.items():
            p: np.ndarray = keypoint.position
            min_x, min_y, min_z = min(min_x, p[0]), min(min_y, p[1]), min(min_z, p[2])
            max_x, max_y, max_z = max(max_x, p[0]), max(max_y, p[1]), max(max_z, p[2])

        min_x -= 0.5
        min_y -= 0.5
        min_z -= 0.5
        max_x += 0.5
        max_y += 0.5
        max_z += 0.5

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
