"""Tests for the i2rt LinearBot (holonomic base + linear lift + two YAM arms)."""

import mujoco
import numpy as np
from mujoco import MjData, MjSpec, mjtGeom
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.robot_configs import LinearBotRobotConfig
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.robots.linearbot import LinearBotKinematics, LinearBotRobot


def _build(yaw: float = 0.0, **cfg_kw):
    cfg = LinearBotRobotConfig(**cfg_kw)
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


def test_linearbot_pi_policy_io():
    """Model vector <-> LinearBot action dict, no server needed."""
    from molmo_spaces.configs.policy_configs_baselines import LinearBotPiPolicyConfig
    from molmo_spaces.policy.learned_policy.linearbot_pi_policy import LinearBotPiPolicy

    assert LinearBotPiPolicyConfig().policy_cls is LinearBotPiPolicy
    pol = LinearBotPiPolicy.__new__(LinearBotPiPolicy)
    pol.grasping_type = "binary"
    pol.current_buffer_index = 0
    out = np.concatenate([np.arange(6) * 0.1, [1.0], np.arange(6) * -0.1, [0.0]])
    act = pol.model_output_to_action(out)
    assert set(act) == {"left_arm", "right_arm", "left_gripper", "right_gripper"}  # base, lift held
    assert np.allclose(act["left_gripper"], [-0.0475, -0.0475])  # open
    assert np.allclose(act["right_gripper"], [0.002, 0.002])  # closed
    assert pol._normalize_gripper(np.array([-0.0475, -0.0475])) == 1.0
    assert pol._normalize_gripper(np.array([0.002, 0.002])) == 0.0


def test_tcp_depth_moves_grasp_site_toward_wrist():
    sites = {}
    for depth in (0.0, 0.02):
        _, model, data, _ = _build(tcp_depth=depth)
        sid = model.site("robot_0/right_grasp_site").id
        sites[depth] = (data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy(),
                        data.xpos[model.body("robot_0/right_link_6").id].copy())
    (p0, r0, w0), (p1, _, w1) = sites[0.0], sites[0.02]
    assert np.allclose(w0, w1)
    assert np.allclose(p1 - p0, -0.02 * r0[:, 2], atol=1e-6)  # back along the approach axis
    assert np.linalg.norm(p1 - w1) < np.linalg.norm(p0 - w0)  # i.e. toward the wrist
