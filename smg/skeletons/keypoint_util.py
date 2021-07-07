import numpy as np

from typing import Dict, Optional


class KeypointUtil:
    """Utility functions related to the keypoints in a skeleton."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def compute_local_keypoint_rotations(*, global_keypoint_poses: Dict[str, np.ndarray],
                                         midhip_from_rests: Dict[str, np.ndarray],
                                         parent_keypoints: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Compute the local rotations for those keypoints in a skeleton that have known global 6D poses.

        .. note::
            This is needed for avatar driving.
        .. note::
            We will attempt to compute a local rotation for any keypoint that has a known global 6D pose.
            A successful computation requires the keypoint to also have:

             (i) A known midhip-from-rest transform.
            (ii) A parent for which we also know the global 6D pose and midhip-from-rest transform.

            Any keypoint that has a global 6D pose but that cannot satisfy these additional conditions
            will be assigned the identity matrix as its local rotation.

        :param global_keypoint_poses:   The global 6D poses of the skeleton's keypoints.
        :param midhip_from_rests:       A 3*3 rotation matrix for each relevant keypoint specifying the transformation
                                        from the orientation of the keypoint to the orientation of the mid-hip keypoint
                                        when the skeleton is in its rest pose.
        :param parent_keypoints:        A map specifying the child to parent relationships between the keypoints.
        :return:                        The local rotations for the relevant keypoints.
        """
        local_keypoint_rotations = {}  # type: Dict[str, np.ndarray]

        # For each keypoint for which a global pose is available:
        for current_name, world_from_current in global_keypoint_poses.items():
            world_from_parent = None         # type: Optional[np.ndarray]
            midhip_from_rest_current = None  # type: Optional[np.ndarray]
            midhip_from_rest_parent = None   # type: Optional[np.ndarray]

            # If it has a parent in the skeleton:
            parent_name = parent_keypoints.get(current_name)  # type: Optional[str]
            if parent_name is not None:
                # Try to get the relevant transformations for both the keypoint and its parent.
                world_from_parent = global_keypoint_poses.get(parent_name)
                midhip_from_rest_current = midhip_from_rests.get(current_name)
                midhip_from_rest_parent = midhip_from_rests.get(parent_name)

            # If they're all available, use them to compute the local rotation for the keypoint.
            if all([x is not None for x in [world_from_parent, midhip_from_rest_current, midhip_from_rest_parent]]):
                # Derivation:
                #
                # m0Tp0 * wTp^-1 * wTc * m0Tc0^-1
                # = m0Tp0 * pTw * wTc * c0Tm0
                # = m0Tp0 * (pTp0 * p0Tm0 * m0Tw) * (wTm0 * m0Tc0 * c0Tc) * c0Tm0
                # = (m0Tp0 * pTp0 * p0Tm0) * (m0Tc0 * c0Tc * c0Tm0)
                local_keypoint_rotations[current_name] = \
                    midhip_from_rest_parent @ \
                    np.linalg.inv(world_from_parent[0:3, 0:3]) @ \
                    world_from_current[0:3, 0:3] @ \
                    np.linalg.inv(midhip_from_rest_current)

            # Otherwise, set the local rotation for the keypoint to the identity matrix (as a default).
            else:
                local_keypoint_rotations[current_name] = np.eye(3)

        return local_keypoint_rotations
