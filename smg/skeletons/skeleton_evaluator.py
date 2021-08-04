import numpy as np

from typing import Dict, List, Optional, Tuple

from .keypoint import Keypoint
from .skeleton3d import Skeleton3D


class SkeletonEvaluator:
    """Functions to evaluate the skeleton estimation quality."""

    # CONSTRUCTOR

    def __init__(self):
        self.__keypoint_to_index_map: Dict[str, int] = {
            "LAnkle": 0,
            "LElbow": 1,
            "LHip": 2,
            "LKnee": 3,
            "LShoulder": 4,
            "LWrist": 5,
            "Neck": 6,
            "Nose": 7,
            "RAnkle": 8,
            "RElbow": 9,
            "RHip": 10,
            "RKnee": 11,
            "RShoulder": 12,
            "RWrist": 13
        }

    # PUBLIC METHODS

    def calculate_3d_pcks(self, correct_keypoint_table: np.ndarray) -> Dict[str, float]:
        pcks: Dict[str, float] = dict()
        row_count: int = correct_keypoint_table.shape[0]
        for keypoint_name, keypoint_index in self.__keypoint_to_index_map.items():
            pcks[keypoint_name] = 100 * np.sum(correct_keypoint_table[:, keypoint_index]).item() / row_count
        return pcks

    def make_correct_keypoint_table(self, matched_skeletons: List[List[Tuple[Skeleton3D, Optional[Skeleton3D]]]], *,
                                    threshold: float = 0.15) -> np.ndarray:
        rows: List[np.ndarray] = []

        for frame in matched_skeletons:
            row: np.ndarray = np.zeros(len(self.__keypoint_to_index_map), dtype=np.uint8)

            for gt_skeleton, detected_skeleton in frame:
                if detected_skeleton is None:
                    continue

                for keypoint_name, gt_keypoint in gt_skeleton.keypoints.items():
                    keypoint_index: Optional[int] = self.__keypoint_to_index_map.get(keypoint_name)
                    detected_keypoint: Optional[Keypoint] = detected_skeleton.keypoints.get(keypoint_name)
                    if keypoint_index is not None and detected_keypoint is not None:
                        distance: float = np.linalg.norm(detected_keypoint.position - gt_keypoint.position)
                        if distance <= threshold:
                            row[keypoint_index] = 1

            rows.append(row)

        return np.vstack(rows)

    def match_detections_with_ground_truth(
        self, *, detected_skeletons: List[List[Skeleton3D]], gt_skeletons: List[List[Skeleton3D]]
    ) -> List[List[Tuple[Skeleton3D, Optional[Skeleton3D]]]]:
        # TODO
        pass
