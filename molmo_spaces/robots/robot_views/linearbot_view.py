"""
Robot view for the i2rt LinearBot: a holonomic Flow Base, a vertical linear lift, and two
6-DOF YAM arms with parallel grippers mounted on the lift carriage.

Arm and gripper move groups are the bimanual YAM ones (same joint/actuator names, attached with
the `left_` / `right_` prefixes); the base follows the mobile Franka's virtual holonomic joints.
"""

import numpy as np
from mujoco import MjData

from molmo_spaces.robots.robot_views.abstract import (
    HoloJointsRobotBaseGroup,
    MJCFFrameMixin,
    RobotBaseGroup,
    RobotView,
    SimplyActuatedMoveGroup,
)
from molmo_spaces.robots.robot_views.bimanual_yam_view import (
    BimanualYamArmGroup,
    BimanualYamGripperGroup,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_pose


class LinearBotBaseGroup(HoloJointsRobotBaseGroup):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        model = mj_data.model
        world_site_id = model.site(f"{namespace}world").id
        holo_base_site_id = model.site(f"{namespace}base_site").id
        joints = [model.joint(f"{namespace}base_{axis}").id for axis in ["x", "y", "theta"]]
        act = [model.actuator(f"{namespace}base_{axis}_act").id for axis in ["x", "y", "theta"]]
        root_body_id = model.body(f"{namespace}base").id
        super().__init__(mj_data, world_site_id, holo_base_site_id, joints, act, root_body_id)


class LinearBotLiftGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    """The vertical linear lift: one slide joint carrying the carriage both arms hang from."""

    def __init__(self, mj_data: MjData, base: RobotBaseGroup, namespace: str = "") -> None:
        model = mj_data.model
        self._carriage_id = model.body(f"{namespace}carriage").id
        self._column_id = model.body(f"{namespace}base").id
        super().__init__(
            mj_data,
            [model.joint(f"{namespace}lift").id],
            [model.actuator(f"{namespace}lift_act").id],
            self._column_id,
            base,
        )

    @property
    def leaf_frame_id(self) -> int:
        return self._carriage_id

    @property
    def leaf_frame_type(self):
        return "body"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._column_id)


class LinearBotRobotView(RobotView):
    """Move groups: base (x, y, theta), lift, right_arm, right_gripper, left_arm, left_gripper.

    The right gripper is listed first, so single-arm planners (which take the first gripper
    move group) use the right arm.
    """

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = LinearBotBaseGroup(mj_data, namespace=namespace)
        move_groups = {
            "base": base,
            "lift": LinearBotLiftGroup(mj_data, base, namespace=namespace),
            "right_arm": BimanualYamArmGroup(mj_data, "right", base, namespace=namespace),
            "right_gripper": BimanualYamGripperGroup(mj_data, "right", base, namespace=namespace),
            "left_arm": BimanualYamArmGroup(mj_data, "left", base, namespace=namespace),
            "left_gripper": BimanualYamGripperGroup(mj_data, "left", base, namespace=namespace),
        }
        super().__init__(mj_data, move_groups)

    @property
    def name(self) -> str:
        return f"{self._namespace}linearbot"

    @property
    def base(self) -> LinearBotBaseGroup:
        return self._move_groups["base"]
