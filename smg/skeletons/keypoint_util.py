import numpy as np

from typing import Dict, Optional


class KeypointUtil:
    """TODO"""

    # PUBLIC STATIC METHODS

    @staticmethod
    def compute_local_keypoint_rotations(*, global_keypoint_poses: Dict[str, np.ndarray],
                                         keypoint_parents: Dict[str, str],
                                         midhip_from_rests: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute the local rotations for relevant keypoints in a skeleton.

        .. note::
            This is needed for avatar driving.

        :param global_keypoint_poses:   TODO
        :param keypoint_parents:        TODO
        :param midhip_from_rests:       TODO
        :return:                        TODO
        """
        local_keypoint_rotations = {}  # type: Dict[str, np.ndarray]

        # For each keypoint for which a global pose is available:
        for current_name, world_from_current in global_keypoint_poses.items():
            world_from_parent = None         # type: Optional[np.ndarray]
            midhip_from_rest_current = None  # type: Optional[np.ndarray]
            midhip_from_rest_parent = None   # type: Optional[np.ndarray]

            # If it has a parent in the skeleton:
            parent_name = keypoint_parents.get(current_name)  # type: Optional[str]
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
