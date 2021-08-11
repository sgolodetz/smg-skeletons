import numpy as np

from typing import Dict, List, Optional, Tuple

from .keypoint import Keypoint
from .skeleton3d import Skeleton3D


class SkeletonEvaluator:
    """An evaluator used to evaluate the quality of any skeletons that have been detected."""

    # CONSTRUCTOR

    def __init__(self, relevant_keypoints: List[str]):
        """
        Construct a skeleton evaluator.

        :param relevant_keypoints:  The keypoints to include in the evaluation.
        """
        self.__keypoint_to_index_map = dict(
            zip(relevant_keypoints, np.arange(len(relevant_keypoints)))
        )  # type: Dict[str, int]

    # PUBLIC STATIC METHODS

    @staticmethod
    def make_correct_keypoint_table(per_joint_error_table: np.ndarray, *, threshold: float = 0.15) -> np.ndarray:
        """
        TODO

        :param per_joint_error_table:   TODO
        :param threshold:               TODO
        :return:                        TODO
        """
        return np.where(per_joint_error_table <= threshold, 1, 0).astype(np.uint8)

    @staticmethod
    def make_default() -> "SkeletonEvaluator":
        """
        Make a skeleton evaluator that bases its evaluation on a simplified 14-keypoint model.

        .. note::
            The various skeleton detectors predict different sets of keypoints, but pretty much all of them
            predict at least these ones.

        :return:    The skeleton evaluator.
        """
        return SkeletonEvaluator([
            "LAnkle", "LElbow", "LHip", "LKnee", "LShoulder", "LWrist", "Neck", "Nose",
            "RAnkle", "RElbow", "RHip", "RKnee", "RShoulder", "RWrist"
        ])

    # PUBLIC METHODS

    def calculate_3d_pcks(self, correct_keypoint_table: np.ndarray) -> Dict[str, float]:
        """
        TODO

        :param correct_keypoint_table:  TODO
        :return:                        TODO
        """
        pcks = dict()                                # type: Dict[str, float]
        row_count = correct_keypoint_table.shape[0]  # type: int

        for keypoint_name, keypoint_index in self.__keypoint_to_index_map.items():
            pcks[keypoint_name] = 100 * np.sum(correct_keypoint_table[:, keypoint_index]).item() / row_count

        return pcks

    def calculate_mpjpes(self, per_joint_error_table: np.ndarray) -> Dict[str, float]:
        """
        Calculate the mean (over all skeleton matches and all frames) position errors for each relevant joint type.

        :param per_joint_error_table:   The per-joint position error table.
        :return:                        A dictionary mapping joint names to their mean position errors.
        """
        mpjpes = dict()  # type: Dict[str, float]

        for keypoint_name, keypoint_index in self.__keypoint_to_index_map.items():
            mpjpes[keypoint_name] = np.nanmean(per_joint_error_table[:, keypoint_index]).item()

        return mpjpes

    def make_per_joint_error_table(self, matched_skeletons: List[List[Tuple[Skeleton3D, Optional[Skeleton3D]]]]) \
            -> np.ndarray:
        """
        Make a table in which each row contains the per-joint position errors (in m) for a matched ground truth
        and detected skeleton pair.

        :param matched_skeletons:   A table of the matched ground truth and detected skeletons.
        :return:                    The per-joint position error table.
        """
        rows = []  # type: List[np.ndarray]

        for frame in matched_skeletons:
            row = np.full(len(self.__keypoint_to_index_map), np.nan)  # type: np.ndarray

            for gt_skeleton, detected_skeleton in frame:
                if detected_skeleton is None:
                    continue

                for keypoint_name, gt_keypoint in gt_skeleton.keypoints.items():
                    keypoint_index = self.__keypoint_to_index_map.get(keypoint_name)    # type: Optional[int]
                    detected_keypoint = detected_skeleton.keypoints.get(keypoint_name)  # type: Optional[Keypoint]
                    if keypoint_index is not None and detected_keypoint is not None:
                        row[keypoint_index] = np.linalg.norm(detected_keypoint.position - gt_keypoint.position)

            rows.append(row)

        return np.vstack(rows)

    def match_detections_with_ground_truth(
        self, *, detected_skeletons: List[List[Skeleton3D]], gt_skeletons: List[List[Skeleton3D]]
    ) -> List[List[Tuple[Skeleton3D, Optional[Skeleton3D]]]]:
        """
        TODO

        :param detected_skeletons:  TODO
        :param gt_skeletons:        TODO
        :return:                    TODO
        """
        # TODO
        pass
