import cv2
import numpy as np

from typing import List, Optional, Set, Tuple

from smg.utility import GeometryUtil

from .keypoint import Keypoint
from .skeleton3d import Skeleton3D


class SkeletonUtil:
    """Utility functions related to skeletons."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def calculate_distance_between_skeletons(skeleton1: Skeleton3D, skeleton2: Skeleton3D) -> float:
        """
        Calculate a distance metric between two skeletons.

        .. note::
            This is useful for skeleton matching.
        .. note::
            In practice, the distance we use is the average distance between corresponding keypoints.
            However, that's not part of the function's interface, and shouldn't be relied upon.

        :param skeleton1:   The first skeleton.
        :param skeleton2:   The second skeleton.
        :return:            The distance metric between the skeletons.
        """
        distances = []

        # For each keypoint in the first skeleton:
        for keypoint_name, keypoint1 in skeleton1.keypoints.items():
            # If the second skeleton has a keypoint with the same name:
            keypoint2 = skeleton2.keypoints.get(keypoint_name)
            if keypoint2 is not None:
                # Calculate the distance between the corresponding keypoints and append it to the list.
                distances.append(np.linalg.norm(keypoint2.position - keypoint1.position))

        # If there were any corresponding keypoints at all, calculate and return the average distance
        # between the pairs of corresponding keypoints; otherwise, return an infinite distance.
        return np.mean(distances) if len(distances) > 0 else np.inf

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
        depopulated_depth_image = depth_image.copy()  # type: np.ndarray
        depopulated_depth_image = np.where(people_mask == 0, depopulated_depth_image, 0.0)

        # If we're debugging, show the people mask.
        if debug:
            cv2.imshow("People Mask", people_mask)
            cv2.waitKey(1)

        return depopulated_depth_image

    @staticmethod
    def depopulate_depth_image_using_3d_boxes(
        skeletons: List[Skeleton3D], depth_image: np.ndarray, world_from_camera: np.ndarray,
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
        people_mask = SkeletonUtil.make_people_mask_from_3d_boxes(
            skeletons, depth_image, world_from_camera, intrinsics
        )  # type: np.ndarray
        return SkeletonUtil.depopulate_depth_image(depth_image, people_mask, debug=debug)

    @staticmethod
    def make_people_mask_from_3d_boxes(
        skeletons: List[Skeleton3D], depth_image: np.ndarray, world_from_camera: np.ndarray,
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
        people_mask = np.zeros(depth_image.shape, dtype=np.uint8)  # type: np.ndarray

        # Compute the world-space points image.
        ws_points = GeometryUtil.compute_world_points_image_fast(
            depth_image, world_from_camera, intrinsics
        )  # type: np.ndarray

        # For each detected 3D skeleton:
        for skeleton in skeletons:
            # Make the corresponding person mask.
            person_mask = SkeletonUtil.make_person_mask_from_3d_box(
                skeleton, depth_image, ws_points, border_size=border_size
            )  # type: np.ndarray

            # Add it to the output mask.
            people_mask = np.where(person_mask == 0, people_mask, 255)

        return people_mask

    @staticmethod
    def make_person_mask_from_3d_box(skeleton: Skeleton3D, depth_image: np.ndarray, ws_points: np.ndarray, *,
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
            p = keypoint.position  # type: np.ndarray
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

    @staticmethod
    def match_detections_with_ground_truth(*, detected_skeletons: List[Skeleton3D], gt_skeletons: List[Skeleton3D]) \
            -> List[Tuple[Skeleton3D, Optional[Skeleton3D]]]:
        """
        Match a list of detected skeletons with a list of ground truth ones.

        .. note::
            We suppress false positive detections (these can be evaluated separately if desired) and focus only
            on trying to find a match for each ground truth skeleton. A particular ground truth skeleton may not
            have a match, in which case its pair will be of the form (ground truth skeleton, None).

        :param detected_skeletons:  A list of detected skeletons.
        :param gt_skeletons:        A list of ground truth skeletons.
        :return:                    A list of matched ground truth and detected skeleton pairs.
        """
        matches: List[Tuple[Skeleton3D, Optional[Skeleton3D]]] = []
        used_detections: Set[int] = set()

        for i in range(len(gt_skeletons)):
            gt_skeleton: Skeleton3D = gt_skeletons[i]
            smallest_distance: float = np.inf
            smallest_index: int = -1

            for j in range(len(detected_skeletons)):
                detected_skeleton: Skeleton3D = detected_skeletons[j]
                distance: float = SkeletonUtil.calculate_distance_between_skeletons(gt_skeleton, detected_skeleton)
                if distance < smallest_distance:
                    smallest_distance = distance
                    smallest_index = j

            if smallest_index != -1 and smallest_index not in used_detections:
                matches.append((gt_skeleton, detected_skeletons[smallest_index]))
                used_detections.add(smallest_index)
            else:
                matches.append((gt_skeleton, None))

        return matches

    @staticmethod
    def save_skeletons(filename: str, skeletons: List[Skeleton3D]) -> None:
        """
        Save a list of skeletons to a file.

        :param filename:    The name of the file.
        :param skeletons:   The list of skeletons.
        """
        with open(filename, "w") as f:
            f.write(repr(skeletons))

    @staticmethod
    def string_to_skeletons(skeletons_repr: str) -> List[Skeleton3D]:
        """
        Convert a (trusted) string that contains the representation of a list of skeletons into the list itself.

        :param skeletons_repr:  A string representation of a list of skeletons.
        :return:                The corresponding list of skeletons.
        """
        return eval(
            skeletons_repr, {'array': np.array, 'Keypoint': Keypoint, 'Skeleton3D': Skeleton3D}
        )

    @staticmethod
    def try_load_skeletons(filename: str) -> Optional[List[Skeleton3D]]:
        """
        Try to load a list of skeletons from a file.

        .. note::
            There are lots of things that could go wrong here, in principle, and lots of different exceptions
            that could be raised as a result. However, this function is only ever used on files we trust, and
            that were saved with save_skeletons. As a result, the only thing that's really likely to go wrong
            is that the file's missing for some reason, e.g. because we specified the wrong path. For that
            reason, we don't devote lots of effort to trying to handle all of the different types of error
            explicitly, and instead just return None if anything at all goes wrong. We can't do much better
            than that anyway, since string_to_skeletons uses eval internally, which is inherently unsafe and
            shouldn't be called on untrusted inputs.

        :param filename:    The name of the file.
        :return:            The list of skeletons, if successful, or None otherwise.
        """
        # noinspection PyBroadException
        try:
            with open(filename, "r") as f:
                return SkeletonUtil.string_to_skeletons(f.read())
        except Exception:
            return None
