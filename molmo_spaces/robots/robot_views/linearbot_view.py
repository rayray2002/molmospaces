"""
Robot view for the i2rt LinearBot: a holonomic Flow Base, a vertical linear lift, and two
6-DOF YAM arms with parallel grippers mounted on the lift carriage.

Arm and gripper move groups are the bimanual YAM ones (same joint/actuator names, attached with
the `left_` / `right_` prefixes); the base follows the mobile Franka's virtual holonomic joints.
"""

from typing import Literal

import numpy as np
from mujoco import MjData

from molmo_spaces.robots.robot_views.abstract import (
    GripperGroup,
    HoloJointsRobotBaseGroup,
    MJCFFrameMixin,
    RobotBaseGroup,
    RobotView,
    SimplyActuatedMoveGroup,
)
from molmo_spaces.robots.robot_views.bimanual_yam_view import BimanualYamArmGroup
from molmo_spaces.utils.linalg_utils import normalize_ang_error
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


class LinearBotGripperGroup(MJCFFrameMixin, GripperGroup):
    """The 95 mm linear gripper of yam_ai2.xml: two fingers, one position servo each.

    Finger joints run from -0.0485 (open, pads ~101 mm apart) through 0 (yam_ai2.xml's closed
    end, pads still 4 mm apart) to CLOSED, where LinearBotRobot lets the pads meet.
    """

    OPEN = -0.0475  # actuator ctrlrange end
    CLOSED = 0.002
    PAD_GAP_AT_ZERO = 0.004

    def __init__(self, mj_data: MjData, side: Literal["left", "right"], base, namespace: str = ""):
        model = mj_data.model
        p = f"{namespace}{side}_"
        joint_ids = [model.joint(f"{p}left_finger").id, model.joint(f"{p}right_finger").id]
        act_ids = [model.actuator(f"{p}gripper_left_tip").id, model.actuator(f"{p}gripper_right_tip").id]
        super().__init__(mj_data, joint_ids, act_ids, model.body(f"{p}link_6").id, base)
        self._ee_site_id = model.site(f"{p}grasp_site").id

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    def set_gripper_ctrl_open(self, open: bool) -> None:
        self.ctrl = [self.OPEN, self.OPEN] if open else [self.CLOSED, self.CLOSED]

    @property
    def inter_finger_dist_range(self) -> tuple[float, float]:
        return 0.0, self.PAD_GAP_AT_ZERO - 2 * self.OPEN

    @property
    def inter_finger_dist(self) -> float:
        """Gap between the two tip pads."""
        return float(max(0.0, self.PAD_GAP_AT_ZERO - self.joint_pos.sum()))

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return self.leaf_frame_to_world


class LinearBotRobotView(RobotView):
    """Move groups: base (x, y, theta), lift, right_arm, right_gripper, left_arm, left_gripper.

    The right gripper is listed first, so single-arm planners (which take the first gripper
    move group) use the right arm.
    """

    def __init__(self, mj_data: MjData, namespace: str = "", active_arm: Literal["right", "left"] = "right",
                 move_base: bool = False) -> None:
        self._namespace = namespace
        self.active_arm = active_arm
        self.move_base = move_base
        self.lock_lift = False
        idle = "left" if active_arm == "right" else "right"
        base = LinearBotBaseGroup(mj_data, namespace=namespace)
        move_groups = {"base": base, "lift": LinearBotLiftGroup(mj_data, base, namespace=namespace)}
        for side in (active_arm, idle):  # single-arm planners use the first gripper move group
            move_groups[f"{side}_arm"] = BimanualYamArmGroup(mj_data, side, base, namespace=namespace)
            move_groups[f"{side}_gripper"] = LinearBotGripperGroup(mj_data, side, base, namespace=namespace)
        super().__init__(mj_data, move_groups)

    @property
    def name(self) -> str:
        return f"{self._namespace}linearbot"

    @property
    def base(self) -> LinearBotBaseGroup:
        return self._move_groups["base"]

    def set_active_arm(self, side: Literal["left", "right"], *, lock_lift: bool = False) -> None:
        """Select the single-arm planner's gripper while preserving all move groups."""
        if side not in ("left", "right"):
            raise ValueError(f"Unknown LinearBot arm: {side}")
        self.active_arm = side
        self.lock_lift = lock_lift
        first = (f"{side}_arm", f"{side}_gripper")
        self._move_groups = {
            key: self._move_groups[key]
            for key in (*first, *(key for key in self._move_groups if key not in first))
        }
        self.__dict__.pop("_gripper_movegroup_ids_cache", None)  # get_gripper_movegroup_ids caches the order

    def get_ik_excluded_movegroup_ids(self) -> list[str]:
        """Manipulation planners keep the base and the idle arm still: the base drives between
        phases, not during them (IK has no collision model and pushed the base into counters), and
        single-arm planners use the active arm (first gripper move group)."""
        return ([] if self.move_base else ["base"]) + [f"{'left' if self.active_arm == 'right' else 'right'}_arm"] + (
            ["lift"] if self.lock_lift else []
        )

    # navigation planners (AStarNavToObjPolicy) track [x, y, theta] base waypoints, as on RBY1
    def distance_to(self, move_group_ids: list[str], target_pose: list) -> float:
        assert len(target_pose) == 3, f"Expected [x, y, theta] pose, got {target_pose}"
        q = np.concatenate([self.get_move_group(m).joint_pos for m in move_group_ids])
        return float(np.linalg.norm([q[0] - target_pose[0], q[1] - target_pose[1],
                                     normalize_ang_error(q[2] - target_pose[2])]))

    def is_close_to(self, move_group_ids: list[str], target_pose: list, threshold: float = 0.05) -> bool:
        return self.distance_to(move_group_ids, target_pose) < threshold
