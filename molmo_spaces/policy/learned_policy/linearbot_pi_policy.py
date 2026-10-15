"""Pi policy client for the i2rt LinearBot: bimanual-YAM interface on the LinearBot move groups."""

import numpy as np

from molmo_spaces.policy.learned_policy.bimanual_yam_pi_policy import BimanualYamPiPolicy
from molmo_spaces.robots.robot_views.linearbot_view import LinearBotGripperGroup


class LinearBotPiPolicy(BimanualYamPiPolicy):
    """Model state/action are the 14-dim [left arm, left grip, right arm, right grip] vector.

    The LinearBot gripper has one servo per finger, ctrl from OPEN (-0.0475) to CLOSED (0.002);
    the model sees one value per gripper, 1 = open, 0 = closed. Base and lift are omitted from
    the action dict, which holds them stationary.
    """

    CAMERA_SHORT_KEYS = {
        "left_wrist_camera": "left",
        "right_wrist_camera": "right",
        "head_camera": "top",
    }
    _OPEN = LinearBotGripperGroup.OPEN
    _CLOSED = LinearBotGripperGroup.CLOSED

    def _normalize_gripper(self, gripper_qpos) -> float:
        q = float(np.atleast_1d(gripper_qpos)[0])
        return float(np.clip((self._CLOSED - q) / (self._CLOSED - self._OPEN), 0.0, 1.0))

    def _scale_gripper_action(self, gripper_value: float) -> np.ndarray:
        if self.grasping_type != "continuous":
            gripper_value = float(gripper_value > 0.5)
        ctrl = self._CLOSED + float(np.clip(gripper_value, 0.0, 1.0)) * (self._OPEN - self._CLOSED)
        return np.array([ctrl, ctrl])
