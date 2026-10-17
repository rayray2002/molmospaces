# i2rt LinearBot in MolmoSpaces

Mobile base + 1 m lift + two YAM arms (95 mm linear grippers). Move groups: `base`, `lift`,
`right_arm`, `right_gripper`, `left_arm`, `left_gripper`. Code: `molmo_spaces/robots/linearbot.py`,
`robots/robot_views/linearbot_view.py`, `LinearBotRobotConfig` / `apply_linearbot_datagen` in
`configs/robot_configs.py`, `LinearBotCameraSystem` in `configs/camera_configs.py`.

Single-arm planners drive the active arm (`LinearBotRobotConfig.active_arm`, `run_pipeline.py
--arm right|left`, default right); the base and the other arm are excluded from their IK
(`LinearBotRobotView.get_ik_excluded_movegroup_ids`).

Environment: upstream main needs `molmospaces-resources==0.0.3a2` and `mink<1.0` (with mujoco 3.5).

## Base and lift control

`command_mode["base"]`:
- `holo_joint_planar_position` (default): absolute world [x, y, theta] targets (planners, navigation).
- `holo_joint_rel_planar_position`: world-frame steps [dx, dy, dtheta].
- `holo_body_rel_planar_position`: steps in the robot frame [forward, left, dtheta], the way a real
  holonomic base is driven (`BodyFrameRelPlanarController`).

`command_mode["lift"]`: `joint_position` (default) or `joint_rel_position`. Any action dict may
carry `"base"` / `"lift"`; groups it leaves out hold still. Navigation (`--task_type nav_to_obj`,
procthor-10k) drives the base with A*.

## Bimanual

Both arms are full move groups (`left_arm`, `left_gripper`, `right_arm`, `right_gripper`) in
actions, H5 recordings and the 14-dim policy layout. `LinearBotKinematics.bimanual_ik(left_pose,
right_pose, q0, base_pose)` solves both grasp sites at once (shared lift solved with the right
arm, base held); tested to track both targets under the joint controllers.

Bimanual datagen: `run_pipeline.py --robot linearbot --task_type bimanual_pick` ("Pick up the X
with the left hand and the Y with the right hand."). `BimanualPickTaskSampler` takes the pick
sampler's object + base placement for the left hand, then a second graspable object within the
right arm's reach (moving the nearest graspable one onto the same support, ~0.3 m in front of the
right arm, when none is in reach). `BimanualPickPlannerPolicy` runs the unmodified pick planner for
the left hand, holds it (arm, gripper, lift), then runs it for the right hand. Success: both objects
lifted, each held by its own hand only.

Coordinated two-hand lift (scripted, own table scene, no house):
`python -m molmo_spaces.tasks.linearbot_bimanual_lift [--seed N] [--render-dir DIR]` lifts an open
box by its end walls with both hands at once (20/20 seeds, box raised >= 9 cm).

## Planner data collection

```bash
cd scripts/datagen
python run_pipeline.py --robot linearbot --task_type pick           --house_inds 1 --samples_per_house 4 --seed 2
python run_pipeline.py --robot linearbot --task_type pick_and_place --house_inds 2 --samples_per_house 4 --seed 2
```

Output: `$ASSETS_DIR/datagen/<task>_planner_v1/<prefix>_<time>/house_<n>/` with
`trajectories_batch_1_of_1.h5` and head / left / right wrist MP4s (640x360).
`apply_linearbot_datagen(cfg)` puts the same rig on any planner datagen config (used by
`run_pipeline.py` and by downstream projects).

Planner success (executed episodes, ithor train houses 1-11; iTHOR train has no index 12+):

| | LinearBot (tcp_depth 0.02) | LinearBot (tcp_depth 0) | DROID |
|---|---|---|---|
| pick + pick-and-place, seeds 2-3 | 49/65 (75%) | 44/72 (61%) | 119/153 (78%) |

- `tcp_depth` = 0.02 (default): grasp site moved 2 cm toward the wrist; the YAM pads sit ~2 cm
  further back from it than the Robotiq pads the grasp libraries were made for, and closed above
  flat handles (spatula, fork). LinearBot vs DROID with it: Fisher p = 0.73.
- Yield is lower than DROID's: the 0.55 x 0.55 m floor base needs free floor within 0.55 m of the
  object (`base_pose_sampling_radius_range`), so cramped kitchens (ithor 3, 12) place few or no
  episodes. 0.7 m placed more but lost more (47% vs 65% pick-and-place).
- Failures shared with DROID: vases and statues; `Tomato_2`'s grasp library (all 29 grasps 7-9 cm
  off the object) is an asset defect.

## Pi policy inference

```bash
python run_pipeline.py --robot linearbot --policy pi --task_type pick
```

`LinearBotPiPolicy` (subclass of `BimanualYamPiPolicy`) talks to a LeRobot gRPC server
(`LinearBotPiPolicyConfig.remote_config`: host, port, policy_type, device; `checkpoint_path`).
It connects lazily on the first inference, so harnesses that never call `prepare_model()` work.

- State and action: 14-dim `[left arm 6, left grip, right arm 6, right grip]`, joint positions.
- Gripper: 1 = open, 0 = closed; mapped to both finger servos (ctrl -0.0475 .. 0.002).
- Cameras: `left_wrist_camera` -> `left`, `right_wrist_camera` -> `right`, `head_camera` -> `top`.
- Base and lift are not in the action dict, so they hold still.
- Checked with a stub server (`scripts/datagen/linearbot_fake_pi_check.py`): full episodes run,
  H5 + MP4s saved. Not checked against a real server or a trained checkpoint: none exists yet
  for this embodiment; a pi05 checkpoint must be trained on LinearBot (or bimanual YAM) data.

## openpi (the project's training / serving stack)

LinearBot's 14-dim vector is openpi's ALOHA layout. `LinearBotOpenpiPolicy` (`--policy openpi`)
sends `{"state", "images": {cam_high: head, cam_left_wrist, cam_right_wrist} 3x224x224,
"prompt"}` to `openpi scripts/serve_policy.py` and executes `actions[:chunk_size]`.

End-to-end check (2026-10-15): 147 LinearBot planner demos -> `h5_to_lerobot.py --robot linearbot`
(19,585 frames) -> openpi config `pi05_linearbot_lora` (ALOHA data config, adapt_to_pi=False,
LoRA from pi05_droid_jointpos) -> 500 training steps (loss 0.31 -> 0.25) -> `serve_policy.py` ->
`run_pipeline.py --robot linearbot --policy openpi --policy_port 8095` and
`eval_poc.py --robot linearbot --policy pi`: episodes run closed loop, the arm tracks the
model's targets (0.02-0.035 rad), grippers open and close. 500 steps is a smoke test, not a
trained policy: no task success is claimed. A full 4000-step LoRA (loss 0.03) on the same 147 demos picked 1/16 closed loop
(ithor 2/4/6/8, new seed); the project's Franka LoRAs trained the same way on ~100 demos score 0-3%.

## JSON benchmarks

Episode specs can name robot-MJCF cameras: `{"type": "mjcf", "name": "head_camera",
"mjcf_name": "head_camera", "fov": 58}` (`MjcfCameraSpec`). `JsonEvalTaskSampler` builds the
robot from `exp_config.robot_config`, so set `LinearBotRobotConfig()` there.

## Limits

- Dimensions other than caster spacing, stroke, lift speed and mass are estimates; grip force
  (finger kp 2000) is a guess. Measure the real robot and override the config fields.
- Open / close (`--task_type open|close`, ithor 1-8, seed 2): LinearBot 11/24 and 9/16, DROID 21/28
  and 6/26 (both differences p ~ 0.05). Open loses to DROID on pregrasp IK (69 vs 42 failures) and
  handles slipping out (41 vs 21): a 6-DOF arm on a locked base pulling handles. A closer base (0.45 m: 4/23) and a stronger
  grip (finger kp 5000: 8/24) were both worse; closing the gap needs base motion during opening. Navigation (`nav_to_obj`, procthor-10k) 8/8. Bimanual tasks are untested; the left arm
  never manipulates.
- No collision-aware whole-body planning (cuRobo not installed): the wrist can be blocked by
  shelf boards (logged as TCP tracking-error retries, as for DROID).
- Shared fixes (controller re-sync, `ik_locked_move_groups`) change behaviour for other robots;
  rerun a DROID baseline before using them for Franka work.
