import numpy as np

from typing import Dict, List, Optional, Tuple

from .keypoint import Keypoint
from .skeleton3d import Skeleton3D


class SkeletonEvaluator:
    """Functions to evaluate the skeleton estimation quality."""

    # CONSTRUCTOR

    def __init__(self, relevant_keypoints: List[str]):
        self.__keypoint_to_index_map: Dict[str, int] = dict(
            zip(relevant_keypoints, np.arange(len(relevant_keypoints)))
        )

    # PUBLIC STATIC METHODS

    @staticmethod
    def make_correct_keypoint_table(per_joint_error_table: np.ndarray, *, threshold: float = 0.15) -> np.ndarray:
        return np.where(per_joint_error_table <= threshold, 1, 0).astype(np.uint8)

    @staticmethod
    def make_default() -> "SkeletonEvaluator":
        return SkeletonEvaluator([
            "LAnkle", "LElbow", "LHip", "LKnee", "LShoulder", "LWrist", "Neck", "Nose",
            "RAnkle", "RElbow", "RHip", "RKnee", "RShoulder", "RWrist"
        ])

    # PUBLIC METHODS

    def calculate_3d_pcks(self, correct_keypoint_table: np.ndarray) -> Dict[str, float]:
        pcks: Dict[str, float] = dict()
        row_count: int = correct_keypoint_table.shape[0]
        for keypoint_name, keypoint_index in self.__keypoint_to_index_map.items():
            pcks[keypoint_name] = 100 * np.sum(correct_keypoint_table[:, keypoint_index]).item() / row_count
        return pcks

    def calculate_mpjpes(self, per_joint_error_table: np.ndarray) -> Dict[str, float]:
        mpjpes: Dict[str, float] = dict()
        for keypoint_name, keypoint_index in self.__keypoint_to_index_map.items():
            mpjpes[keypoint_name] = np.nanmean(per_joint_error_table[:, keypoint_index]).item()
        return mpjpes

    def make_per_joint_error_table(self, matched_skeletons: List[List[Tuple[Skeleton3D, Optional[Skeleton3D]]]]) \
            -> np.ndarray:
        rows: List[np.ndarray] = []

        for frame in matched_skeletons:
            row: np.ndarray = np.full(len(self.__keypoint_to_index_map), np.nan)

            for gt_skeleton, detected_skeleton in frame:
                if detected_skeleton is None:
                    continue

                for keypoint_name, gt_keypoint in gt_skeleton.keypoints.items():
                    keypoint_index: Optional[int] = self.__keypoint_to_index_map.get(keypoint_name)
                    detected_keypoint: Optional[Keypoint] = detected_skeleton.keypoints.get(keypoint_name)
                    if keypoint_index is not None and detected_keypoint is not None:
                        row[keypoint_index] = np.linalg.norm(detected_keypoint.position - gt_keypoint.position)

            rows.append(row)

        return np.vstack(rows)

    def match_detections_with_ground_truth(
        self, *, detected_skeletons: List[List[Skeleton3D]], gt_skeletons: List[List[Skeleton3D]]
    ) -> List[List[Tuple[Skeleton3D, Optional[Skeleton3D]]]]:
        # TODO
        pass
