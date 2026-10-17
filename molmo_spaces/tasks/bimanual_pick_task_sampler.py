"""LinearBot two-object pick: the pick sampler's object + base placement for the left hand, plus a
second graspable object within the right arm's reach."""

import logging

import mujoco
import numpy as np

from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.tasks.bimanual_pick_task import BimanualPickTask
from molmo_spaces.tasks.pick_task import PickTask
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
from molmo_spaces.tasks.task_sampler_errors import ObjectPlacementError
from molmo_spaces.utils.grasp_sample import get_noncolliding_grasp_mask
from molmo_spaces.utils.grasps import get_pickup_grasps
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb
from molmo_spaces.utils.mujoco_scene_utils import place_object_near
from molmo_spaces.utils.pose import pose_mat_to_7d

log = logging.getLogger(__name__)


class BimanualPickTaskSampler(PickTaskSampler):
    reach: float = 0.55  # YAM reach from its mount, as for the single-arm LinearBot gates
    min_separation: float = 0.10  # between the two pickups

    def _mount_xy(self, env: CPUMujocoEnv, side: str) -> np.ndarray:
        cfg = self.config.robot_config
        base = env.current_robot.robot_view.base.pose
        y = cfg.arm_spacing / 2 * (1 if side == "left" else -1)
        return (base @ np.array([cfg.arm_mount_x, y, 0.0, 1.0]))[:2]

    def _graspable(self, env: CPUMujocoEnv, obj) -> bool:
        uid = self.get_asset_uid_from_object(env, obj.name)
        if not uid or not self._has_grasps(obj, uid):
            return False
        try:
            grasps = get_pickup_grasps(env, obj, include_flipped=False,
                                       grasp_libraries=self.config.task_sampler_config.grasp_libraries)
        except ValueError:
            return False
        if len(grasps) == 0:
            return False
        try:
            return bool(np.any(get_noncolliding_grasp_mask(env.current_model, env.current_data, grasps, 64)))
        except KeyError:  # no grasp-collision probes in the scene: skip the check, as the pick sampler does
            return True

    def _post_placement_setup(self, env: CPUMujocoEnv, pickup_obj_name: str, supporting_geom_id: int) -> bool:
        if not super()._post_placement_setup(env, pickup_obj_name, supporting_geom_id):
            return False
        om = env.object_managers[env.current_batch_index]
        left = om.get_object_by_name(pickup_obj_name)
        if np.linalg.norm(left.position[:2] - self._mount_xy(env, "left")) > self.reach:
            log.info(f"[BIMANUAL] {pickup_obj_name} out of the left arm's reach")
            return False
        right_mount = self._mount_xy(env, "right")
        others = [om.get_object_by_name(o.name) for o in self.candidate_objects if o.name != pickup_obj_name]
        dist = {o.name: float(np.linalg.norm(o.position[:2] - right_mount)) for o in others}
        candidates = sorted(
            (o for o in others if dist[o.name] <= self.reach
             and np.linalg.norm(o.position[:2] - left.position[:2]) >= self.min_separation
             and abs(o.position[2] - left.position[2]) < 0.4),
            key=lambda o: dist[o.name])
        right = next((o for o in candidates if self._graspable(env, om.get_object_by_name(o.name))), None)
        if right is None:  # kitchens rarely have one in reach: bring the nearest graspable one over
            right = self._relocate_into_right_reach(env, om, left, supporting_geom_id, sorted(others, key=lambda o: dist[o.name]))
        if right is None:
            log.info(f"[BIMANUAL] no graspable second object in the right arm's reach ({len(candidates)} near)")
            return False
        right = om.get_object_by_name(right.name)
        cfg = self.config.task_config
        cfg.right_pickup_obj_name = right.name
        cfg.right_pickup_obj_start_pose = pose_mat_to_7d(right.pose).tolist()
        goal = pose_mat_to_7d(right.pose)
        goal[2] += 0.05
        cfg.right_pickup_obj_goal_pose = goal.tolist()
        return True

    relocate_tries: int = 4  # nearest graspable candidates tried when none is in reach

    def _relocate_into_right_reach(self, env: CPUMujocoEnv, om, left, supporting_geom_id: int, others: list):
        """Move a graspable scene object onto the left pickup's support, ~0.3 m in front of the
        right arm mount, like the pick-and-place sampler stages receptacles."""
        base = env.current_robot.robot_view.base.pose
        bottom_z = body_aabb(env.current_model, env.current_data, left.body_id)[0][2]
        target = np.r_[self._mount_xy(env, "right") + 0.30 * base[:2, 0], bottom_z]
        tried = 0
        for o in others:
            if tried >= self.relocate_tries:
                break
            obj = om.get_object_by_name(o.name)
            if not self._graspable(env, obj):
                continue
            tried += 1
            try:
                place_object_near(env.current_data, obj.body_id, target, min_dist=0.0, max_dist=0.12,
                                  reference_pos=left.position, max_dist_to_reference=0.8,
                                  supporting_geom_id=supporting_geom_id)
            except ObjectPlacementError:
                continue
            mujoco.mj_forward(env.current_model, env.current_data)
            obj = om.get_object_by_name(o.name)
            ok = (np.linalg.norm(obj.position[:2] - self._mount_xy(env, "right")) <= self.reach
                  and np.linalg.norm(obj.position[:2] - left.position[:2]) >= self.min_separation
                  and self._graspable(env, obj))
            log.info(f"[BIMANUAL] relocated {o.name} next to the right arm: {'ok' if ok else 'rejected'}")
            if ok:
                return obj
        return None

    def _sample_task(self, env: CPUMujocoEnv) -> PickTask:
        task = super()._sample_task(env)
        om = env.object_managers[env.current_batch_index]
        name = self.config.task_config.right_pickup_obj_name
        _, filtered = self._generate_referral_expressions(env, name, [om.get_object(name)])
        self.config.task_config.referral_expressions["right_pickup_obj_name"] = om.sample_expression(filtered)
        return task

    def _task_cls(self) -> type[PickTask]:
        return BimanualPickTask
