"""i2rt LinearBot: holonomic Flow Base + vertical linear lift + two YAM arms.

I2RT publishes no CAD/URDF for the LinearBot itself (only the YAM arms, grippers and desktop
stations), so the base, column and carriage here are boxes sized from what is published:
casters at +-0.20 m (flow_base_controller.py), 1.0 m lift stroke at up to 0.3 m/s, 95 kg net.
Everything else (base height, column position, arm spacing and mount height, head camera)
is a guess exposed in LinearBotRobotConfig -- measure the real robot and set it there.
"""

from typing import TYPE_CHECKING, cast

import mujoco
import numpy as np
from mujoco import MjData, MjSpec, mjtGeom
from scipy.spatial.transform import Rotation as R

from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.kinematics.parallel.dummy_parallel_kinematics import DummyParallelKinematics
from molmo_spaces.robots.bimanual_yam import BimanualYamRobot
from molmo_spaces.robots.robot_views.linearbot_view import LinearBotGripperGroup

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.configs.robot_configs import LinearBotRobotConfig


class LinearBotKinematics(MlSpacesKinematics):
    """IK tuned for the 6-DOF YAM arm on the lift.

    Measured on 60 reachable lift + right-arm targets (FK of random joint configs), solved from
    the home pose: the stock solver (damping 1e-12, full steps) solves 25/60; damping 1e-4 with
    half steps 45/60; plus up to 5 random restarts at >= 300 iterations 55/60. Restarts only run
    when the warm-started solve fails, so tracking along a trajectory stays continuous.
    """

    n_restarts: int = 5
    min_iter: int = 300

    def __init__(self, robot_config) -> None:
        super().__init__(robot_config)
        self._rng = np.random.default_rng(0)

    def ik(self, move_group_id, pose, unlocked_move_group_ids, q0, base_pose, rel_to_base=False,
           eps=1e-4, max_iter=1000, damping=None, dt=None):
        kw = dict(rel_to_base=rel_to_base, eps=eps, max_iter=max(max_iter, self.min_iter),
                  damping=1e-4, dt=0.5)
        res = super().ik(move_group_id, pose, unlocked_move_group_ids, q0, base_pose, **kw)
        unlocked = unlocked_move_group_ids or self._robot_view.move_group_ids()
        for _ in range(self.n_restarts):
            if res is not None:
                break
            seed = {k: np.array(v, dtype=float, copy=True) for k, v in q0.items()}
            for mg_id in unlocked:
                lim = self._robot_view.get_move_group(mg_id).joint_pos_limits
                if np.all(np.isfinite(lim)):  # leaves the planar base where it is
                    seed[mg_id] = self._rng.uniform(lim[:, 0], lim[:, 1])
            res = super().ik(move_group_id, pose, unlocked_move_group_ids, seed, base_pose, **kw)
        return res


class LinearBotRobot(BimanualYamRobot):
    def __init__(self, mj_data: MjData, config: "MlSpacesExpConfig") -> None:
        super().__init__(mj_data, config)
        self._kinematics = LinearBotKinematics(config.robot_config)
        self._parallel_kinematics = DummyParallelKinematics(config.robot_config, self._kinematics)
        base_controller_cls = {
            "holo_joint_planar_position": JointPosController,
        }[config.robot_config.command_mode["base"]]
        self._controllers["base"] = base_controller_cls(self._robot_view.get_move_group("base"))
        self._controllers["lift"] = JointPosController(self._robot_view.get_move_group("lift"))

    @staticmethod
    def robot_model_root_name() -> str:
        # gravcomp is applied below this body: the carriage and both arms, not the base
        return "carriage"

    @classmethod
    def add_robot_to_scene(
        cls,
        robot_config: "LinearBotRobotConfig",
        spec: MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
        strip_meshes: bool = False,
    ) -> None:
        cfg = cast("LinearBotRobotConfig", robot_config)
        pos = list(pos) + [0.0] if len(pos) == 2 else list(pos)
        init_rot = R.from_quat(quat, scalar_first=True)
        assert np.allclose(init_rot.as_euler("xyz")[:2], 0), "LinearBot must start upright"

        white = f"{prefix}linearbot_white"
        dark = f"{prefix}linearbot_dark"
        spec.add_material(name=white, rgba=[0.9, 0.9, 0.88, 1.0])
        spec.add_material(name=dark, rgba=[0.15, 0.15, 0.17, 1.0])

        base = spec.worldbody.add_body(name=f"{prefix}base", pos=pos, quat=quat)
        base.add_site(name=f"{prefix}base_site")
        bx, by, bz = cfg.base_size
        # keep the base box off the floor so the planar joints slide freely; 3 cm (the real base
        # rides on casters) also clears floors a little above z=0 (anysim's exported kitchens: 7 mm)
        clearance = 0.03
        base.add_geom(
            type=mjtGeom.mjGEOM_BOX,
            size=[bx / 2, by / 2, (bz - clearance) / 2],
            pos=[0, 0, clearance + (bz - clearance) / 2],
            material=white,
            mass=74.0,  # with column + carriage + arms: the published 95 kg
        )
        base.add_geom(  # dark top plate, visual only
            type=mjtGeom.mjGEOM_BOX,
            size=[bx / 2 + 0.005, by / 2 + 0.005, 0.005],
            pos=[0, 0, bz],
            material=dark,
            contype=0,
            conaffinity=0,
        )
        for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:  # casters, visual only
            base.add_geom(
                type=mjtGeom.mjGEOM_CYLINDER,
                size=[0.05, 0.02, 0],
                pos=[0.2 * sx, 0.2 * sy, 0.05],
                quat=[0.7071068, 0.7071068, 0, 0],
                material=dark,
                contype=0,
                conaffinity=0,
            )
        cx, cw, ch = cfg.column_x, cfg.column_width, cfg.column_height
        base.add_geom(
            type=mjtGeom.mjGEOM_BOX,
            size=[cw / 2, cw / 2, ch / 2],
            pos=[cx, 0, bz + ch / 2],
            material=white,
            mass=10.0,
        )
        hx, hz, pitch = cfg.head_camera_pose
        p = np.radians(pitch)
        base.add_camera(
            name=f"{prefix}head_camera",
            pos=[hx, 0, hz],
            # look along +x, pitched down; image x points to robot right (-y)
            xyaxes=[0, -1, 0, np.sin(p), 0, np.cos(p)],
            fovy=58.0,
        )

        carriage = base.add_body(name=f"{prefix}carriage", pos=[0, 0, cfg.arm_mount_height])
        carriage.add_joint(
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            name=f"{prefix}lift",
            axis=[0, 0, 1],
            range=[0.0, cfg.lift_stroke],
            damping=cfg.lift_control_params["damping"],
            armature=0.1,
        )
        # carriage plate in front of the column, the arms bolt to its top face
        plate_len = cfg.arm_mount_x - cx + 0.08
        carriage.add_geom(
            type=mjtGeom.mjGEOM_BOX,
            size=[plate_len / 2, cfg.arm_spacing / 2 + 0.06, 0.015],
            pos=[cx + plate_len / 2, 0, -0.015],
            material=dark,
            mass=cfg.carriage_mass,
        )

        for side, y in [("left", cfg.arm_spacing / 2), ("right", -cfg.arm_spacing / 2)]:
            arm_spec = cls._load_robot_spec(cfg, strip_meshes=strip_meshes)
            # The 4 mm tip pads tunnel through objects under default (soft) contact once the
            # finger servo is stiff enough to grip: a 20 mm block squeezed at kp 600+ ended with
            # the pads past each other. Robotiq-style pad contact (as in MolmoSpaces' 2f85.xml)
            # holds the true width at kp 2000 with 40 N.
            for g in arm_spec.geoms:
                if "tip_collider" in g.name:
                    g.solref[:] = [0.004, 1]
                    g.solimp[:3] = [0.95, 0.99, 0.001]
                    g.priority = 1
                    g.friction[0] = 1.0
            frame = carriage.add_frame(pos=[cfg.arm_mount_x, y, 0])
            frame.attach_body(arm_spec.body("arm"), f"{prefix}{side}_", "")
            for tip in ["left", "right"]:
                grip = spec.actuator(f"{prefix}{side}_gripper_{tip}_tip")
                grip.gainprm[0] = cfg.gripper_kp
                grip.biasprm[1:3] = [-cfg.gripper_kp, -cfg.gripper_kd]
                # yam_ai2.xml stops the fingers with the pads 4 mm apart, so forks, knives and
                # spoon handles were never squeezed; let them close until the pads meet
                grip.ctrlrange[1] = LinearBotGripperGroup.CLOSED
                spec.joint(f"{prefix}{side}_{tip}_finger").range[1] = LinearBotGripperGroup.CLOSED
            if cfg.tcp_depth:
                # planners put the grasp site on the grasp pose; moving it back along its own -z
                # (toward the wrist) sends the pads that much deeper around the object
                site = spec.site(f"{prefix}{side}_grasp_site")
                shift = np.zeros(3)
                mujoco.mju_rotVecQuat(shift, np.array([0.0, 0.0, -cfg.tcp_depth]), np.asarray(site.quat, float))
                site.pos = np.asarray(site.pos, float) + shift
            link6 = spec.body(f"{prefix}{side}_link_6")
            link6.add_camera(
                name=f"{prefix}{side}_wrist_camera",
                pos=cfg.wrist_camera_pos,
                quat=cfg.wrist_camera_quat,
                fovy=58.0,
            )

        spec.worldbody.add_site(name=f"{prefix}world")

        # planar base: two slide joints + one hinge, driven by site/joint position actuators
        # (same scheme as MobileFrankaRobot, so qpos of base_x/base_y are world coordinates)
        for gear_idx, jnt_name in enumerate(["base_x", "base_y"]):
            params = cfg.base_control_params[f"{jnt_name}_act"]
            axis = np.zeros(3)
            axis[gear_idx] = 1
            base.add_joint(
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                name=f"{prefix}{jnt_name}",
                axis=init_rot.inv().apply(axis),
                range=[-params["ctrlrange"], params["ctrlrange"]],
                ref=pos[gear_idx],
            )
            gear = [0] * 6
            gear[gear_idx] = 1
            act = spec.add_actuator(
                name=f"{prefix}{jnt_name}_act",
                target=f"{prefix}base_site",
                refsite=f"{prefix}world",
                trntype=mujoco.mjtTrn.mjTRN_SITE,
                biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                ctrlrange=[-params["ctrlrange"], params["ctrlrange"]],
                gear=gear,
            )
            act.gainprm[0] = params["kp"]
            act.biasprm[:3] = [0, -params["kp"], -params["kd"]]

        params = cfg.base_control_params["base_theta_act"]
        base.add_joint(
            type=mujoco.mjtJoint.mjJNT_HINGE,
            name=f"{prefix}base_theta",
            axis=[0, 0, 1],
            ref=init_rot.as_euler("xyz")[2],
        )
        act = spec.add_actuator(
            name=f"{prefix}base_theta_act",
            target=f"{prefix}base_theta",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        )
        act.gainprm[0] = params["kp"]
        act.biasprm[:3] = [0, -params["kp"], -params["kd"]]

        params = cfg.lift_control_params
        act = spec.add_actuator(
            name=f"{prefix}lift_act",
            target=f"{prefix}lift",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            ctrlrange=[0.0, cfg.lift_stroke],
            forcerange=[-params["max_force"], params["max_force"]],
            forcelimited=1,
        )
        act.gainprm[0] = params["kp"]
        act.biasprm[:3] = [0, -params["kp"], -params["kd"]]

    @classmethod
    def apply_control_overrides(cls, spec: MjSpec, robot_config: "LinearBotRobotConfig"):
        super().apply_control_overrides(spec, robot_config)
        if robot_config.gravcomp:
            spec.body(robot_config.robot_namespace + "carriage").gravcomp = 1.0

