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
        ws_points: np.ndarray = GeometryUtil.compute_world_points_image_fast(
            depth_image, world_from_camera, intrinsics
        )

        depopulated_depth_image: np.ndarray = depth_image.copy()

        for skeleton in skeletons:
            person_mask: np.ndarray = SkeletonUtil.make_person_mask(skeleton, ws_points)
            depopulated_depth_image = np.where(person_mask == 0, depopulated_depth_image, 0.0)

            if debug:
                cv2.imshow("Person Mask", person_mask)

        return depopulated_depth_image

    @staticmethod
    def make_person_mask(skeleton: Skeleton, ws_points: np.ndarray) -> np.ndarray:
        min_x, min_y, min_z = np.inf, np.inf, np.inf
        max_x, max_y, max_z = -np.inf, -np.inf, -np.inf
        for _, keypoint in skeleton.keypoints.items():
            p: np.ndarray = keypoint.position
            min_x, min_y, min_z = min(min_x, p[0]), min(min_y, p[1]), min(min_z, p[2])
            max_x, max_y, max_z = max(max_x, p[0]), max(max_y, p[1]), max(max_z, p[2])

        min_x -= 0.3
        min_y -= 0.3
        min_z -= 0.3
        max_x += 0.3
        max_y += 0.3
        max_z += 0.3

        # cx, cy, cz = (min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2
        # min_x, min_y, min_z = min(min_x - 0.2, cx - 0.3), min(min_y - 0.2, cy - 0.3), min(min_z - 0.2, cz - 0.3)
        # max_x, max_y, max_z = max(max_x + 0.2, cx + 0.3), max(max_y + 0.2, cy + 0.3), max(max_z + 0.2, cz + 0.3)

        xs, ys, zs = ws_points[:, :, 0], ws_points[:, :, 1], ws_points[:, :, 2]

        return np.where(
            (min_x <= xs) & (xs <= max_x) & (min_y <= ys) & (ys <= max_y) & (min_z <= zs) & (zs <= max_z), 255, 0
        ).astype(np.uint8)
