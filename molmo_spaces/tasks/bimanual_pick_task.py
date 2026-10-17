"""LinearBot two-object pickup with contact-based per-hand success."""

import numpy as np

from molmo_spaces.env.abstract_sensors import SensorSuite
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.env.sensors import ObjectStartPoseSensor
from molmo_spaces.tasks.pick_task import PickTask


class BimanualPickTask(PickTask):
    def get_task_description(self) -> str:
        refs = self.config.task_config.referral_expressions
        return (f"Pick up the {refs['pickup_obj_name']} with the left hand and the "
                f"{refs['right_pickup_obj_name']} with the right hand.")

    def get_task_objects(self, batch_index: int = 0) -> dict[str, str]:
        objects = super().get_task_objects(batch_index)
        self.deduplicate_task_objects_name(
            self.config.task_config, "right_pickup_obj_name", objects, "right_pickup_obj"
        )
        return objects

    def _create_sensor_suite_from_config(self, config) -> SensorSuite:
        suite = super()._create_sensor_suite_from_config(config)
        suite.extend([ObjectStartPoseSensor(
            object_name=config.task_config.right_pickup_obj_name, uuid="right_obj_start"
        )])
        return suite

    def arm_info(self, side: str, batch_index: int = 0) -> dict:
        cfg = self.config.task_config
        prefix = "right_" if side == "right" else ""
        data = self.env.mj_datas[batch_index]
        obj = MlSpacesObject(data=data, object_name=getattr(cfg, prefix + "pickup_obj_name"))
        view = self.env.robots[batch_index].robot_view
        gripper = view.get_gripper(f"{side}_gripper")
        # this arm's links and fingers (bodies named "<ns><side>_..."), never the other arm or the base
        prefix_body = f"{view._namespace}{side}_"
        hand_geoms = {g for g in range(data.model.ngeom)
                      if data.model.body(data.model.geom_bodyid[g]).name.startswith(prefix_body)}
        hand_contact = False
        other_contact = False
        for contact in data.contact[: data.ncon]:
            roots = [data.model.body_rootid[data.model.geom_bodyid[g]]
                     for g in (contact.geom1, contact.geom2)]
            if (roots[0] == obj.body_id) ^ (roots[1] == obj.body_id):
                other_geom = contact.geom2 if roots[0] == obj.body_id else contact.geom1
                if other_geom in hand_geoms:
                    hand_contact = True
                else:
                    other_contact = True
        lift = float(obj.position[2] - getattr(cfg, prefix + "pickup_obj_start_pose")[2])
        closed = bool(np.all(gripper.ctrl > gripper.OPEN + 0.01))
        return dict(lift_height=lift, hand_contact=hand_contact, other_contact=other_contact,
                    gripper_closed=closed,
                    success=bool(lift >= cfg.succ_pos_threshold and hand_contact
                                 and not other_contact and closed))

    def get_info(self) -> list[dict]:
        metrics = []
        for i in range(self.env.n_batch):
            left, right = self.arm_info("left", i), self.arm_info("right", i)
            metrics.append(dict(left=left, right=right,
                                success=left["success"] and right["success"],
                                episode_step=self.episode_step_count))
        return metrics

    def get_reward(self) -> np.ndarray:
        return np.array([sum(max(0.0, info[side]["lift_height"])
                             * info[side]["success"] for side in ("left", "right"))
                         for info in self.get_info()])

    def get_obs_scene(self):
        scene = super().get_obs_scene()
        scene.update(text=self.get_task_description(),
                     right_object_name=self.config.task_config.right_pickup_obj_name)
        return scene
