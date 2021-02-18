import numpy as np

from .skeleton import Skeleton


class SkeletonUtil:
    """Utility functions related to skeletons."""

    # PUBLIC STATIC METHODS

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

        xs, ys, zs = ws_points[:, :, 0], ws_points[:, :, 1], ws_points[:, :, 2]

        return np.where(
            (min_x <= xs) & (xs <= max_x) & (min_y <= ys) & (ys <= max_y) & (min_z <= zs) & (zs <= max_z), 255, 0
        ).astype(np.uint8)
