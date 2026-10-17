"""Compose two unmodified PickPlannerPolicy instances on LinearBot."""

import logging

from molmo_spaces.policy.solvers.object_manipulation.pick_planner_policy import PickPlannerPolicy

log = logging.getLogger(__name__)


def arm_pick_config(config, side):
    """Project the selected pickup into the ordinary pick planner's config."""
    cfg = config.model_copy()
    cfg.task_config = config.task_config.model_copy(deep=True)
    if side == "right":
        for field in ("name", "start_pose", "goal_pose"):
            setattr(cfg.task_config, "pickup_obj_" + field,
                    getattr(config.task_config, "right_pickup_obj_" + field))
    return cfg


class BimanualPickPlannerPolicy(PickPlannerPolicy):
    def reset(self, reset_retries=True):
        self._original_arm = self.robot_view.active_arm
        self._original_lock = self.robot_view.lock_lift
        self._held = {}
        self._start_arm("left")

    def _start_arm(self, side):
        self.side = side
        self.robot_view.set_active_arm(side, lock_lift=side == "right")
        self.child = PickPlannerPolicy(arm_pick_config(self.config, side), self.task)
        self.child.reset()
        self.target_poses = self.child.target_poses
        log.info("Bimanual pick: starting %s hand", side)

    @property
    def retry_count(self):
        return self.child.retry_count if hasattr(self, "child") else 0

    def get_phase(self):
        return self.child.get_phase()

    def get_action(self, info):
        action = self.child.get_action(info)
        action.update(self._held)
        if action.get("done") and self.side == "left" and action.get("success", True):
            left = self.task.arm_info("left")
            log.info("Bimanual pick: left hand done: %s", left)
            if not left["success"]:
                return {"done": True, "success": False}
            self._held = {mg: self.robot_view.get_move_group(mg).ctrl.copy()
                          for mg in ("left_arm", "left_gripper", "lift")}
            try:
                self._start_arm("right")
            except ValueError as exc:
                log.info("Bimanual right-hand planning failed: %s", exc)
                return {"done": True, "success": False}
            action.pop("done")
        return action

    def close(self):
        self.robot_view.set_active_arm(self._original_arm, lock_lift=self._original_lock)
