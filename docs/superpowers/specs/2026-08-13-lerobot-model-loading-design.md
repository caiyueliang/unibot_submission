# LeRobot Model Loading Design

## Goal

Extend `example/example_policy.py` so the UniBot policy service can load a
LeRobot PI0.5 checkpoint and run inference while keeping the README WebSocket
contract unchanged.

## Inputs

Startup configuration comes from environment variables:

- `UNIBOT_POLICY_PATH`: pretrained policy directory, for example
  `/data/ckpt/pi05_all/042000/pretrained_model/`.
- `UNIBOT_REPO_ID`: LeRobot dataset ID/path used to load metadata and stats.
- `UNIBOT_DEFAULT_TASK`: fallback task text if a request has no
  `observation.language`.
- `UNIBOT_CONTROL_SPACE`: defaults to `joint`.
- `UNIBOT_OBS_CHUNK_SIZE`: defaults to the policy `n_obs_steps`, expected `1`
  for the current PI0.5 checkpoint.
- `UNIBOT_ACTION_CHUNK_SIZE`: defaults to the policy chunk size, expected `50`
  for the current PI0.5 checkpoint.

If `UNIBOT_POLICY_PATH` is not set, the policy keeps the current safe fallback
behavior for local interface tests.

## Loading Flow

The implementation mirrors `unitree_lerobot/eval_robot/eval_g1.py` but excludes
robot hardware setup:

1. Load `PreTrainedConfig.from_pretrained(UNIBOT_POLICY_PATH)`.
2. Set `policy_cfg.pretrained_path` to `UNIBOT_POLICY_PATH`.
3. Load `LeRobotDataset(repo_id=UNIBOT_REPO_ID)` for metadata and stats.
4. Build the model with `make_policy(cfg=policy_cfg, ds_meta=dataset.meta)`.
5. Call `policy.eval()`.
6. If `policy_cfg.type == "pi05"` and `use_split_state_action` is enabled,
   adapt dataset stats with `adapt_pi05_stats`.
7. Build pre/post processors with `make_pre_post_processors`, overriding the
   device processor to the configured policy device.

## Inference Flow

Per request:

1. Convert README observation keys to dataset/model keys.
2. Select the newest frame from each observation chunk.
3. Convert images to torch tensors, preserving HWC `uint8` because the copied
   `predict_action` logic converts them to CHW float in `[0, 1]`.
4. Build 16-dimensional `observation.state` from left arm, right arm, left
   gripper, and right gripper.
5. Use request `observation.language` as the task. Fall back to
   `UNIBOT_DEFAULT_TASK` only when the request has no language key.
6. Run preprocessor, `policy.select_action`, and postprocessor under
   `torch.inference_mode()`.
7. Convert the returned 16-dimensional action to README joint keys:
   left arm, right arm, left gripper, right gripper. `action.pivot` remains a
   zero chunk because it is not part of the current PI0.5 16-dimensional action.

## Commands

Server startup:

```bash
conda activate lerobot_cyl
cd /home/wair/caiyueliang/unibot_submission
PYTHONPATH=/home/wair/caiyueliang/lerobot/src:/home/wair/caiyueliang/unitree_lerobot:/home/wair/yangsheng/xr_teleoperate/teleop/teleimager/src:$PYTHONPATH \
UNIBOT_SUBMISSION_TOKEN=<token> \
UNIBOT_POLICY_PATH=/data/ckpt/pi05_all/042000/pretrained_model/ \
UNIBOT_REPO_ID=/data/datasets/G1_Dex1_ArrangeTestTubes_3cams \
UNIBOT_CONTROL_SPACE=joint \
python example/run_server.py 8765
```

Local client validation:

```bash
conda activate lerobot_cyl
cd /home/wair/caiyueliang/unibot_submission
UNIBOT_SUBMISSION_TOKEN=<token> python example/run_client.py ws://127.0.0.1:8765
```

## Verification

Unit tests cover fallback behavior, startup configuration parsing, observation
conversion, and action-vector splitting. The local client still validates the
README action contract. Full checkpoint loading is verified by running the
server startup command in the `lerobot_cyl` environment.
