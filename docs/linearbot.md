# i2rt LinearBot in MolmoSpaces

Mobile base + 1 m lift + two YAM arms (95 mm linear grippers). Move groups: `base`, `lift`,
`right_arm`, `right_gripper`, `left_arm`, `left_gripper`. Code: `molmo_spaces/robots/linearbot.py`,
`robots/robot_views/linearbot_view.py`, `LinearBotRobotConfig` / `LinearBotCameraSystem` in `configs/`.

## Planner data collection (pick)

```bash
cd scripts/datagen
python run_pipeline.py --robot linearbot --task_type pick --house_inds 1 \
    --samples_per_house 4 --seed 2 --run_name_prefix lb
```

Output: `$ASSETS_DIR/datagen/pick_planner_v1/<prefix>_<time>/house_<n>/` with
`trajectories_batch_1_of_1.h5` and head / left / right wrist MP4s. The right arm picks; the base
and left arm are locked during manipulation. Reference rate (8 houses x 3 seeds): 57/81
executed successes (70.4%); placement limits yield (59% of requested episodes).

## Pi policy inference

```bash
python run_pipeline.py --robot linearbot --policy pi --task_type pick
```

`LinearBotPiPolicy` (subclass of `BimanualYamPiPolicy`) talks to a LeRobot gRPC server
(`LinearBotPiPolicyConfig.remote_config`: host, port, policy_type, device; `checkpoint_path`).

- State and action: 14-dim `[left arm 6, left grip, right arm 6, right grip]`, joint positions.
- Gripper: model value 1 = open, 0 = closed; mapped to both finger servos
  (ctrl -0.0475 open .. 0.002 closed). `grasping_type` binary or continuous.
- Cameras: `left_wrist_camera` -> `left`, `right_wrist_camera` -> `right`, `head_camera` -> `top`.
- Base and lift are not in the action dict, so they hold still.
- Checked with a stub server that returns the current state: a full 300-step episode runs,
  H5 + MP4s are saved. Not checked against a real server or a trained policy.

## Limits

- Dimensions other than caster spacing, stroke, lift speed and mass are estimates; grip force
  (finger kp 2000) is a guess. Measure the real robot and override the config fields.
- Pick only; pick-and-place, open/close and navigation are untested.
- No collision-aware whole-body planning (cuRobo not installed).
- Shared fixes (controller re-sync, `ik_locked_move_groups`) change behavior for other robots;
  rerun a DROID baseline before using them for Franka work.
