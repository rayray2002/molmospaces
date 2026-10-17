"""Self-contained, controller-driven LinearBot lift of an open box by its end walls.

Run with ``python -m molmo_spaces.tasks.linearbot_bimanual_lift --render-dir DIR``.
No house, grasp database, attachment constraint, or object pose edits are used.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from molmo_spaces.configs.robot_configs import LinearBotRobotConfig
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.robots.linearbot import LinearBotKinematics, LinearBotRobot
from molmo_spaces.robots.robot_views.linearbot_view import LinearBotGripperGroup


def _build_scene(seed: int):
    cfg = LinearBotRobotConfig()
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1],
        rgba=[0.25, 0.29, 0.33, 1],
    )
    spec.worldbody.add_light(pos=[0.3, -1, 2.8], dir=[0, 0, -1])
    table = spec.worldbody.add_body(name="table", pos=[0.65, 0, 0])
    table.add_geom(
        name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, 0.775], size=[0.30, 0.50, 0.025], rgba=[0.55, 0.40, 0.25, 1],
    )
    for x in (-0.24, 0.24):
        for y in (-0.44, 0.44):
            table.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, y, 0.375],
                           size=[0.025, 0.025, 0.375], rgba=[0.3, 0.3, 0.3, 1])

    # A rigid open crate, 30 cm across the arms, 14 cm deep, 8 cm high, 0.5 kg.
    # Each hand pinches a 10 mm end wall; the complete 30 cm span exceeds a 99 mm gripper.
    rng = np.random.default_rng(seed)
    pos = np.array([0.50, 0, 0.84])
    pos[:2] += rng.uniform(-0.003, 0.003, 2)
    yaw = rng.uniform(-0.015, 0.015)
    box = spec.worldbody.add_body(
        name="box", pos=pos, quat=Rotation.from_euler("z", yaw).as_quat(scalar_first=True),
    )
    box.add_freejoint(name="box_free")
    parts = [("bottom", [0, 0, -0.036], [0.07, 0.15, 0.004])]
    parts += [(f"end_{s}", [0, s * 0.145, 0], [0.07, 0.005, 0.04]) for s in (-1, 1)]
    parts += [(f"side_{s}", [s * 0.065, 0, 0], [0.005, 0.14, 0.04]) for s in (-1, 1)]
    volumes = [8 * np.prod(size) for _, _, size in parts]
    for (name, p, size), volume in zip(parts, volumes):
        box.add_geom(
            name=f"box_{name}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=p, size=size,
            mass=0.5 * volume / sum(volumes), friction=[1, 0.005, 0.0001],
            rgba=[0.85, 0.35, 0.10, 1],
        )
    LinearBotRobot.add_robot_to_scene(cfg, spec, cfg.robot_namespace, [0, 0], [1, 0, 0, 0])
    LinearBotRobot.apply_control_overrides(spec, cfg)
    model = spec.compile()
    data = mujoco.MjData(model)
    view = cfg.robot_view_factory(data, cfg.robot_namespace)
    view.set_qpos_dict(cfg.init_qpos)
    for side in ("left", "right"):
        view.get_move_group(f"{side}_gripper").joint_pos = np.full(2, LinearBotGripperGroup.OPEN)
    mujoco.mj_forward(model, data)
    return cfg, model, data, view


def run_bimanual_lift(seed: int = 0, render_dir: str | Path | None = None) -> dict:
    """Plan both hands, pinch opposite end walls, and raise the shared lift by 10 cm.

    Success reads the simulated box displacement and actual box/pad contacts on BOTH
    fingers of BOTH hands, plus nonempty gaps, after a one-second final hold.
    Tracking errors are measured grasp-site position errors in metres.
    """
    started = time.perf_counter()
    cfg, model, data, view = _build_scene(seed)
    kin = LinearBotKinematics(cfg)
    controllers = {mg: JointPosController(view.get_move_group(mg)) for mg in view.move_group_ids()}
    controllers["base"].set_target(view.base.noop_ctrl)
    box_id = model.body("box").id
    box_geoms = set(np.flatnonzero(model.geom_bodyid == box_id))
    pad_geoms = {
        side: {finger: {i for i in range(model.ngeom)
                        if model.geom(i).name.startswith(f"{cfg.robot_namespace}{side}_{finger}_tip_collider")}
               for finger in ("left", "right")}
        for side in ("left", "right")
    }

    def contacts() -> dict:
        touching = set()
        for contact in data.contact:
            if contact.dist <= 0:
                if contact.geom1 in box_geoms:
                    touching.add(contact.geom2)
                if contact.geom2 in box_geoms:
                    touching.add(contact.geom1)
        return {side: {finger: bool(ids & touching) for finger, ids in pads.items()}
                for side, pads in pad_geoms.items()}

    renderer = None
    frames = []
    trace = []
    timings = {}
    phase = "settle"
    sample_steps = round(1 / (20 * model.opt.timestep))
    steps = 0
    if render_dir is not None:
        from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer

        renderer = MjOpenGLRenderer(model, width=640, height=480)

    def run(seconds: float, target: dict | None = None, ramp: bool = False) -> None:
        nonlocal steps
        t0 = time.perf_counter()
        initial = {mg: c.target.copy() for mg, c in controllers.items()}
        n = round(seconds / model.opt.timestep)
        for step in range(n):
            if target is not None:
                alpha = min(1.0, (step + 1) / n) if ramp else 1.0
                alpha = alpha * alpha * (3 - 2 * alpha)
                for mg, value in target.items():
                    controllers[mg].set_target((1 - alpha) * initial[mg] + alpha * np.asarray(value))
            for mg, controller in controllers.items():
                view.get_move_group(mg).ctrl = controller.compute_ctrl_inputs()
            mujoco.mj_step(model, data)
            steps += 1
            if steps % sample_steps == 0:
                mujoco.mj_forward(model, data)
                trace.append({"time": float(data.time), "phase": phase,
                              "box_z": float(data.xpos[box_id, 2]),
                              "gripper_gaps": {s: view.get_move_group(f"{s}_gripper").inter_finger_dist
                                               for s in ("left", "right")},
                              "contacts": contacts()})
                if renderer is not None:
                    renderer.update(data, camera=f"{cfg.robot_namespace}head_camera")
                    frames.append(renderer.render())
        timings[phase] = timings.get(phase, 0) + time.perf_counter() - t0

    failure = None
    tracking = {}
    targets = None
    try:
        run(0.5)
        start_z = float(data.xpos[box_id, 2])
        center = data.xpos[box_id].copy()
        box_rot = data.xmat[box_id].reshape(3, 3).copy()
        q = view.get_qpos_dict()

        def poses(outside: float, height: float) -> dict:
            result = {}
            for side, sign in (("left", 1), ("right", -1)):
                pose = np.eye(4)
                pose[:3, :3] = box_rot @ np.diag([-1, 1, -1])
                pose[:3, 3] = center + box_rot @ np.array([0, sign * (0.145 + outside), height])
                result[side] = pose
            return result

        for phase, outside, height, duration in (
            ("pregrasp", 0.07, 0.16, 2.0),
            ("above_walls", 0, 0.16, 0.8),
            ("grasp", 0, 0.015, 1.2),
        ):
            targets = poses(outside, height)
            t0 = time.perf_counter()
            solution = kin.bimanual_ik(targets["left"], targets["right"], q, view.base.pose,
                                      move_lift=phase == "pregrasp")
            timings[f"{phase}_ik"] = time.perf_counter() - t0
            if solution is None:
                failure = f"{phase} IK failed"
                break
            q = solution
            run(duration, {mg: q[mg] for mg in ("lift", "left_arm", "right_arm")}, ramp=True)
            run(0.4)
        if failure is None:
            phase = "close"
            run(0.7, {f"{s}_gripper": np.full(2, LinearBotGripperGroup.CLOSED)
                      for s in ("left", "right")}, ramp=True)
            run(0.4)
            phase = "lift"
            run(1.0, {"lift": np.asarray(q["lift"]) + 0.10}, ramp=True)
            phase = "hold"
            run(1.0)
            for side in ("left", "right"):
                actual = view.get_move_group(f"{side}_gripper").leaf_frame_to_world
                expected = targets[side][:3, 3] + [0, 0, 0.10]
                tracking[side] = float(np.linalg.norm(actual[:3, 3] - expected))
        mujoco.mj_forward(model, data)
        dz = float(data.xpos[box_id, 2] - start_z)
        gaps = {s: view.get_move_group(f"{s}_gripper").inter_finger_dist for s in ("left", "right")}
        final_contacts = contacts()
        success = (failure is None and dz >= 0.05 and all(gap > 0.003 for gap in gaps.values())
                   and all(all(pads.values()) for pads in final_contacts.values()))
        result = {"success": success, "seed": seed, "box_dz": dz,
                  "tracking_error": tracking, "gripper_gaps": gaps, "contacts": final_contacts,
                  "failure": failure, "timings": timings, "trace": trace,
                  "simulation_seconds": float(data.time)}
    finally:
        if renderer is not None:
            renderer.close()
    if render_dir is not None:
        from molmo_spaces.utils.save_utils import save_frames_to_mp4

        output = Path(render_dir)
        output.mkdir(parents=True, exist_ok=True)
        video = output / "head_camera.mp4"
        save_frames_to_mp4(frames, str(video), fps=20)
        result["video_path"] = str(video)
        repo = Path(__file__).resolve().parents[2]
        result["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        result["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip())
        result["argv"] = sys.argv
        result["wall_seconds"] = time.perf_counter() - started
        (output / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    result["timings"]["total"] = time.perf_counter() - started
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-dir", type=Path)
    args = parser.parse_args()
    episode = run_bimanual_lift(args.seed, args.render_dir)
    print(json.dumps({k: v for k, v in episode.items() if k != "trace"}, indent=2))
    raise SystemExit(0 if episode["success"] else 1)
