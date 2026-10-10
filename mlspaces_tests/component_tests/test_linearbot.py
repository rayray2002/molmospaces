"""Tests for the i2rt LinearBot (holonomic base + linear lift + two YAM arms)."""

import mujoco
import numpy as np
from mujoco import MjData, MjSpec, mjtGeom
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.robot_configs import LinearBotRobotConfig
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.robots.linearbot import LinearBotKinematics, LinearBotRobot


def _build(yaw: float = 0.0):
    cfg = LinearBotRobotConfig()
    spec = MjSpec()
    spec.worldbody.add_geom(name="floor", type=mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
    quat = R.from_euler("z", yaw).as_quat(scalar_first=True)
    LinearBotRobot.add_robot_to_scene(cfg, spec, cfg.robot_namespace, [1.0, -2.0], list(quat))
    LinearBotRobot.apply_control_overrides(spec, cfg)
    model = spec.compile()
    data = MjData(model)
    view = cfg.robot_view_factory(data, cfg.robot_namespace)
    view.set_qpos_dict(cfg.init_qpos)
    mujoco.mj_forward(model, data)
    return cfg, model, data, view


def test_lift_tracks_and_base_holds_its_yaw():
    _, model, data, view = _build(yaw=2.5)
    controllers = {mg: JointPosController(view.get_move_group(mg)) for mg in view.move_group_ids()}
    controllers["lift"].set_target(np.array([0.8]))
    controllers["base"].set_target(view.base.noop_ctrl)  # hold, the way planner actions do
    for _ in range(int(3.0 / model.opt.timestep)):
        for mg, c in controllers.items():
            view.get_move_group(mg).ctrl = c.compute_ctrl_inputs()
        mujoco.mj_step(model, data)
    assert abs(view.get_move_group("lift").joint_pos[0] - 0.8) < 0.02
    pose = view.base.pose
    assert np.allclose(pose[:2, 3], [1.0, -2.0], atol=0.02)
    # the yaw actuator has no ctrlrange: its commands must not be clipped to 0
    assert abs(np.arctan2(pose[1, 0], pose[0, 0]) - 2.5) < 0.02


def test_ik_reaches_fk_target_with_base_locked():
    cfg, _, _, view = _build()
    kin = LinearBotKinematics(cfg)
    q = view.get_qpos_dict()
    q_goal = {k: v.copy() for k, v in q.items()}
    q_goal["lift"] = np.array([0.6])
    q_goal["right_arm"] = np.array([0.3, 1.6, 1.2, -0.4, 0.3, 0.2])
    target = kin.fk(q_goal, view.base.pose)["right_gripper"]
    sol = kin.ik("right_gripper", target, ["lift", "right_arm"], q, view.base.pose)
    assert sol is not None
    assert np.allclose(sol["base"], q["base"])
    reached = kin.fk(sol, view.base.pose)["right_gripper"]
    assert np.linalg.norm(reached[:3, 3] - target[:3, 3]) < 1e-3
