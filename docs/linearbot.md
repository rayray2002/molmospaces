# i2rt LinearBot in MolmoSpaces

Mobile base + 1 m lift + two YAM arms (95 mm linear grippers). Move groups: `base`, `lift`,
`right_arm`, `right_gripper`, `left_arm`, `left_gripper`. Code: `molmo_spaces/robots/linearbot.py`,
`robots/robot_views/linearbot_view.py`, `LinearBotRobotConfig` / `apply_linearbot_datagen` in
`configs/robot_configs.py`, `LinearBotCameraSystem` in `configs/camera_configs.py`.

Planners use the right arm; the base and left arm are locked while manipulating (`ik_locked_move_groups`).

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

## JSON benchmarks

Episode specs can name robot-MJCF cameras: `{"type": "mjcf", "name": "head_camera",
"mjcf_name": "head_camera", "fov": 58}` (`MjcfCameraSpec`). `JsonEvalTaskSampler` builds the
robot from `exp_config.robot_config`, so set `LinearBotRobotConfig()` there.

## Limits

- Dimensions other than caster spacing, stroke, lift speed and mass are estimates; grip force
  (finger kp 2000) is a guess. Measure the real robot and override the config fields.
- Pick, pick-and-place only; open/close, navigation and bimanual tasks are untested; the left
  arm never manipulates.
- No collision-aware whole-body planning (cuRobo not installed): the wrist can be blocked by
  shelf boards (logged as TCP tracking-error retries, as for DROID).
- Shared fixes (controller re-sync, `ik_locked_move_groups`) change behaviour for other robots;
  rerun a DROID baseline before using them for Franka work.
