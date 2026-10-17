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


def test_linearbot_openpi_policy_io():
    """openpi ALOHA-layout obs / action mapping, no server needed."""
    from molmo_spaces.policy.learned_policy.linearbot_pi_policy import LinearBotOpenpiPolicy

    pol = LinearBotOpenpiPolicy.__new__(LinearBotOpenpiPolicy)
    pol.grasping_type, pol.grasping_threshold = "binary", 0.5
    pol.task = type("T", (), {"get_task_description": lambda self: "Pick up the Mug"})()
    img = np.zeros((360, 640, 3), np.uint8)
    obs = {"qpos": {"left_arm": [0.1] * 6, "left_gripper": [0.002, 0.002],
                    "right_arm": [0.2] * 6, "right_gripper": [-0.0475, -0.0475]},
           "head_camera": img, "left_wrist_camera": img, "right_wrist_camera": img}
    x = pol.obs_to_model_input([obs])
    assert x["state"].shape == (14,) and x["state"][6] == 0.0 and x["state"][13] == 1.0
    assert {k: v.shape for k, v in x["images"].items()} == {k: (3, 224, 224) for k in pol.CAMERAS}
    assert x["prompt"] == "pick up the mug"
    act = pol.model_output_to_action(np.r_[np.zeros(6), 0.9, np.ones(6), 0.1])
    assert np.allclose(act["left_gripper"], [-0.0475, -0.0475]) and np.allclose(act["right_gripper"], [0.002, 0.002])
    assert set(act) == {"left_arm", "left_gripper", "right_arm", "right_gripper"}


def test_base_body_frame_step_and_relative_lift():
    """Body-frame base command: 0.3 m forward + 0.1 m left at heading 2.5 rad lands where the
    robot faces; the lift steps up relative to where it is."""
    from molmo_spaces.controllers.joint_rel_pos import (
        BodyFrameRelPlanarController,
        JointRelPosController,
    )

    _, model, data, view = _build()
    pose = np.eye(4)  # place it the way task samplers do: through the base joints
    pose[:3, :3] = R.from_euler("z", 2.5).as_matrix()
    pose[:2, 3] = [1.0, -2.0]
    view.base.pose = pose
    mujoco.mj_forward(model, data)
    ctl = {mg: JointPosController(view.get_move_group(mg)) for mg in view.move_group_ids()}
    ctl["base"].set_target(view.base.noop_ctrl)

    def run(seconds):
        for _ in range(int(seconds / model.opt.timestep)):
            for mg, c in ctl.items():
                view.get_move_group(mg).ctrl = c.compute_ctrl_inputs()
            mujoco.mj_step(model, data)

    run(3.0)
    p0 = view.base.pose.copy()
    h0 = np.arctan2(p0[1, 0], p0[0, 0])
    ctl["base"] = BodyFrameRelPlanarController(view.get_move_group("base"))
    ctl["lift"] = JointRelPosController(view.get_move_group("lift"))
    lift0 = view.get_move_group("lift").joint_pos[0]
    ctl["base"].set_target(np.array([0.3, 0.1, 0.2]))
    ctl["lift"].set_target(np.array([0.15]))
    run(3.0)
    c, s = np.cos(h0), np.sin(h0)
    pose = view.base.pose
    assert np.allclose(pose[:2, 3], p0[:2, 3] + [0.3 * c - 0.1 * s, 0.3 * s + 0.1 * c], atol=0.02)
    turn = np.arctan2(pose[1, 0], pose[0, 0]) - h0
    assert abs((turn + np.pi) % (2 * np.pi) - np.pi - 0.2) < 0.02
    assert abs(view.get_move_group("lift").joint_pos[0] - (lift0 + 0.15)) < 0.02


def test_active_arm_left_reorders_grippers_and_ik_exclusion():
    cfg, _, _, view = _build(active_arm="left")
    assert view.get_gripper_movegroup_ids()[0] == "left_gripper"
    assert set(view.get_ik_excluded_movegroup_ids()) == {"base", "right_arm"}
    _, _, _, right = _build()
    assert right.get_gripper_movegroup_ids()[0] == "right_gripper"


def test_bimanual_ik_and_tracking():
    """Both arms to FK targets of a random reachable configuration, then tracked together by the
    joint controllers with the base held."""
    cfg, model, data, view = _build()
    kin = LinearBotKinematics(cfg)
    q = view.get_qpos_dict()
    rng = np.random.default_rng(3)
    goal = {k: np.array(v, dtype=float) for k, v in q.items()}
    for arm in ("left_arm", "right_arm"):
        lim = view.get_move_group(arm).joint_pos_limits
        goal[arm] = q[arm] + rng.uniform(-0.3, 0.3, size=len(q[arm]))
        goal[arm] = np.clip(goal[arm], lim[:, 0], lim[:, 1])
    fk = kin.fk(goal, view.base.pose)
    sol = kin.bimanual_ik(fk["left_gripper"], fk["right_gripper"], q, view.base.pose)
    assert sol is not None and np.allclose(sol["base"], q["base"])
    reached = kin.fk(sol, view.base.pose)
    for g in ("left_gripper", "right_gripper"):
        assert np.linalg.norm(reached[g][:3, 3] - fk[g][:3, 3]) < 1e-3
    ctl = {mg: JointPosController(view.get_move_group(mg)) for mg in view.move_group_ids()}
    for mg in ("lift", "left_arm", "right_arm"):
        ctl[mg].set_target(np.asarray(sol[mg]))
    ctl["base"].set_target(view.base.noop_ctrl)
    for _ in range(int(3.0 / model.opt.timestep)):
        for mg, c in ctl.items():
            view.get_move_group(mg).ctrl = c.compute_ctrl_inputs()
        mujoco.mj_step(model, data)
    for g in ("left_gripper", "right_gripper"):
        assert np.linalg.norm(view.get_move_group(g).leaf_frame_to_world[:3, 3] - fk[g][:3, 3]) < 0.02


def test_set_active_arm_switches_planner_arm_mid_episode():
    """BimanualPickPlannerPolicy hands over from the left to the right arm on one view; the
    cached gripper order and the IK exclusions must follow."""
    _, _, _, view = _build()
    assert view.get_gripper_movegroup_ids()[0] == "right_gripper"  # fills the cache
    view.set_active_arm("left")
    assert view.get_gripper_movegroup_ids()[0] == "left_gripper"
    assert set(view.get_ik_excluded_movegroup_ids()) == {"base", "right_arm"}
    view.set_active_arm("right", lock_lift=True)  # second hand: lift held where the first left it
    assert view.get_gripper_movegroup_ids()[0] == "right_gripper"
    assert set(view.get_ik_excluded_movegroup_ids()) == {"base", "left_arm", "lift"}


def test_ik_moves_base_within_bound_when_unlocked():
    """With the base unlocked (manipulation_moves_base) IK may shift it to reach, never past
    max_base_shift; a target needing more falls back to an arm-only solve."""
    cfg, _, _, view = _build()
    kin = LinearBotKinematics(cfg)
    q = view.get_qpos_dict()
    unlocked = ["base", "lift", "right_arm"]
    for shift, expect_base_move in ((0.08, True), (0.6, False)):
        moved = {k: np.array(v, dtype=float) for k, v in q.items()}
        moved["base"] = np.asarray(q["base"], float) + [shift, 0.0, 0.0]
        target = kin.fk(moved, view.base.pose)["right_gripper"]  # where the hand is after a base shift
        sol = kin.ik("right_gripper", target, unlocked, q, view.base.pose)
        if sol is None:
            assert not expect_base_move
            continue
        d = np.asarray(sol["base"]) - np.asarray(q["base"])
        assert np.linalg.norm(d[:2]) <= kin.max_base_shift + 1e-6
        if expect_base_move:  # the arm reaches from where the base is now ...
            now = {**sol, "base": np.asarray(q["base"], float)}
            reached = kin.fk(now, view.base.pose)["right_gripper"]
            assert np.linalg.norm(reached[:3, 3] - target[:3, 3]) < 1e-3
            assert np.linalg.norm(d[:2]) > 0.01  # ... while the base is sent toward the target


def test_batched_ik_matches_serial_and_holds_base():
    """LinearBotParallelKinematics (warp) solves the same reachable targets as LinearBotKinematics,
    for either arm, and never moves the base even when asked to unlock everything."""
    from molmo_spaces.robots.linearbot import LinearBotParallelKinematics

    cfg, _, _, view = _build()
    kin, par = LinearBotKinematics(cfg), LinearBotParallelKinematics(cfg)
    par.warmup_ik(8)  # the planner warms up first; it solves for the first frame group ("lift")
    q = view.get_qpos_dict()
    rng = np.random.default_rng(0)
    for side in ("right", "left"):
        targets = []
        for _ in range(12):
            g = {k: np.array(v, dtype=float) for k, v in q.items()}
            lim = view.get_move_group(f"{side}_arm").joint_pos_limits
            g[f"{side}_arm"] = rng.uniform(lim[:, 0], lim[:, 1])
            targets.append(kin.fk(g, view.base.pose)[f"{side}_gripper"])
        res = par.ik(f"{side}_gripper", np.stack(targets), None, [q] * 12, view.base.pose)
        solved = [(r, t) for r, t in zip(res, targets) if r is not None]
        assert len(solved) >= 10
        for r, t in solved:
            full = {**q, **r}
            assert np.allclose(full["base"], q["base"])
            assert np.linalg.norm(kin.fk(full, view.base.pose)[f"{side}_gripper"][:3, 3] - t[:3, 3]) < 1e-3
