"""Pi policy clients for the i2rt LinearBot (two YAM arms on a mobile base + lift).

Both use the 14-dim [left arm 6, left grip, right arm 6, right grip] state / action, which is also
openpi's ALOHA layout. Grip: 1 = open, 0 = closed; each LinearBot gripper has one servo per finger,
ctrl from OPEN (-0.0475) to CLOSED (0.002). Base and lift are left out of the action dict, which
holds them stationary.

- LinearBotPiPolicy: LeRobot gRPC pi05 server (MolmoSpaces' bimanual-YAM client).
- LinearBotOpenpiPolicy: openpi websocket server (openpi scripts/serve_policy.py), ALOHA keys.
"""

import numpy as np

from molmo_spaces.policy.learned_policy.bimanual_yam_pi_policy import BimanualYamPiPolicy
from molmo_spaces.policy.learned_policy.pi_policy import PI_Policy
from molmo_spaces.policy.learned_policy.utils import resize_with_pad
from molmo_spaces.robots.robot_views.linearbot_view import LinearBotGripperGroup

OPEN, CLOSED = LinearBotGripperGroup.OPEN, LinearBotGripperGroup.CLOSED


def grip_to_unit(finger_qpos) -> float:
    """Finger position -> 1 open .. 0 closed."""
    q = float(np.atleast_1d(finger_qpos)[0])
    return float(np.clip((CLOSED - q) / (CLOSED - OPEN), 0.0, 1.0))


def unit_to_grip_ctrl(value: float) -> np.ndarray:
    """1 open .. 0 closed -> both finger servo targets."""
    ctrl = CLOSED + float(np.clip(value, 0.0, 1.0)) * (OPEN - CLOSED)
    return np.array([ctrl, ctrl])


class LinearBotPiPolicy(BimanualYamPiPolicy):
    CAMERA_SHORT_KEYS = {
        "left_wrist_camera": "left",
        "right_wrist_camera": "right",
        "head_camera": "top",
    }

    def _normalize_gripper(self, gripper_qpos) -> float:
        return grip_to_unit(gripper_qpos)

    def _scale_gripper_action(self, gripper_value: float) -> np.ndarray:
        if self.grasping_type != "continuous":
            gripper_value = float(gripper_value > 0.5)
        return unit_to_grip_ctrl(gripper_value)


class LinearBotOpenpiPolicy(PI_Policy):
    """{"state": 14, "images": {cam_high, cam_left_wrist, cam_right_wrist} as 3x224x224 uint8,
    "prompt"} -> actions [horizon, 14] (absolute joint targets + grips)."""

    CAMERAS = {"cam_high": "head_camera", "cam_left_wrist": "left_wrist_camera", "cam_right_wrist": "right_wrist_camera"}

    def render(self, obs):
        pass

    def obs_to_model_input(self, obs):
        if isinstance(obs, list):
            obs = obs[0]
        q = obs["qpos"]
        state = np.concatenate([np.asarray(q["left_arm"][:6]), [grip_to_unit(q["left_gripper"])],
                                np.asarray(q["right_arm"][:6]), [grip_to_unit(q["right_gripper"])]])
        return {
            "state": state.astype(np.float32),
            "images": {k: np.ascontiguousarray(resize_with_pad(obs[c], 224, 224).transpose(2, 0, 1))
                       for k, c in self.CAMERAS.items()},
            "prompt": self.task.get_task_description().lower(),
        }

    def _grip(self, value: float) -> np.ndarray:
        if self.grasping_type != "continuous":
            value = float(value > self.grasping_threshold)
        return unit_to_grip_ctrl(value)

    def model_output_to_action(self, model_output):
        a = np.asarray(model_output, np.float32)
        return {"left_arm": a[0:6], "left_gripper": self._grip(a[6]),
                "right_arm": a[7:13], "right_gripper": self._grip(a[13])}
