# ExamplePolicy Adapter Design

## Goal

Adapt `example/example_policy.py` so the service accepts the UniBot evaluator
format from `README.zh.md` and exposes observations/actions in the model feature
format described by
`/data/datasets/G1_Dex1_ArrangeTestTubes_3cams/meta/info.json`.

## Scope

Only `example/example_policy.py` changes. `example_policy_back.py`, the
WebSocket transport, and the local evaluator remain unchanged.

## Observation Mapping

The evaluator sends README observation keys. The policy converts them to the
model dataset keys before model inference:

- `observation.images.cam_left_high` -> `observation.images.head_stereo_left`
- `observation.images.cam_left_wrist` -> `observation.images.wrist_left`
- `observation.images.cam_right_wrist` -> `observation.images.wrist_right`

State keys that already match the dataset schema are passed through unchanged:
left/right arm, left/right gripper, lower body, and left/right EE pose in base.
`observation.language` remains a plain string.

## Action Mapping

The policy uses `joint` control by default because the dataset contains
`action.left_arm`, `action.right_arm`, grippers, and `action.pivot`, which match
the README joint action contract.

Final service responses always contain:

- `meta.token`
- `action.left_arm`
- `action.right_arm`
- `action.left_gripper`
- `action.right_gripper`
- `action.pivot`

All action arrays are returned as `np.float32` with shape
`(action_chunk_size, dim)`.

## Model Hook

`_predict_model_action(model_obs)` is the single replacement point for real
model inference. The default implementation is a safe fallback: hold the latest
observed arm and gripper positions and return zero pivot motion.

## Error Handling

Adapter helpers normalize model outputs:

- Single-frame arrays with shape `(dim,)` are repeated to the action chunk.
- Chunk arrays shorter than `action_chunk_size` are padded with the last frame.
- Chunk arrays longer than `action_chunk_size` are truncated.
- Missing actions fall back to latest observed state where possible, otherwise
  zeros with the correct shape.

## Verification

Run `example/run_server.py` and `example/run_client.py` with the same
`UNIBOT_SUBMISSION_TOKEN`. The client must validate token, action keys, shapes,
and dtypes for multiple steps.
