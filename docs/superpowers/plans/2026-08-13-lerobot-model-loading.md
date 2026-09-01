# LeRobot Model Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load a LeRobot PI0.5 checkpoint inside `ExamplePolicy` and return README-compliant UniBot actions.

**Architecture:** Keep the WebSocket transport unchanged. `ExamplePolicy` owns startup env parsing, optional model loading, request observation conversion, model inference, and final action normalization.

**Tech Stack:** Python, NumPy, PyTorch, LeRobot policy factory, LeRobot processor pipelines, unittest.

## Global Constraints

Only modify `example/example_policy.py` and focused adapter tests. Do not import robot hardware/image-client/control modules into the submission service. `repo_id` is a startup parameter. `task` comes from `obs["observation.language"]` with `UNIBOT_DEFAULT_TASK` as fallback. Missing `UNIBOT_POLICY_PATH` keeps safe fallback behavior.

---

### Task 1: Model Loading Configuration

**Files:**
- Modify: `example/example_policy.py`
- Modify: `example/test_example_policy_adapter.py`

**Interfaces:**
- Consumes: environment variables `UNIBOT_POLICY_PATH`, `UNIBOT_REPO_ID`, `UNIBOT_DEFAULT_TASK`
- Produces: `ExamplePolicy._model_enabled: bool`, `ExamplePolicy._load_model() -> None`

- [ ] **Step 1: Write the failing test**

```python
def test_requires_repo_id_when_policy_path_is_set(self):
    os.environ["UNIBOT_POLICY_PATH"] = "/tmp/model"
    os.environ.pop("UNIBOT_REPO_ID", None)
    with self.assertRaisesRegex(ValueError, "UNIBOT_REPO_ID"):
        ExamplePolicy()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest example.test_example_policy_adapter -v`
Expected: FAIL because `UNIBOT_POLICY_PATH` is ignored.

- [ ] **Step 3: Implement model loading configuration**

Read env vars in `__init__`; call `_load_model()` only when `UNIBOT_POLICY_PATH` is set; require `UNIBOT_REPO_ID` in that mode.

- [ ] **Step 4: Run tests**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest example.test_example_policy_adapter -v`
Expected: PASS.

### Task 2: Request Preparation and Action Splitting

**Files:**
- Modify: `example/example_policy.py`
- Modify: `example/test_example_policy_adapter.py`

**Interfaces:**
- Consumes: model observation dict from `_adapt_observation(obs)`
- Produces: `_prepare_policy_observation(model_obs: dict) -> tuple[dict, str]`, `_split_model_action_vector(action: object) -> dict`

- [ ] **Step 1: Write the failing test**

```python
def test_prepare_policy_observation_uses_request_language(self):
    model_obs = self.policy._adapt_observation(self.obs)
    prepared, task = self.policy._prepare_policy_observation(model_obs)
    self.assertEqual(task, "arrange the test tubes.")
    self.assertEqual(prepared["observation.state"].shape, (16,))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest example.test_example_policy_adapter -v`
Expected: FAIL because `_prepare_policy_observation` is missing.

- [ ] **Step 3: Implement request preparation and action splitting**

Use newest chunk frame, build 16D state from left/right arm and grippers, convert images to torch tensors when torch is available, split 16D or chunked action vectors into left/right arm and grippers.

- [ ] **Step 4: Run tests**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest example.test_example_policy_adapter -v`
Expected: PASS.

### Task 3: Real LeRobot Inference Hook and Commands

**Files:**
- Modify: `example/example_policy.py`
- Modify: `README.zh.md` or final response command snippets only

**Interfaces:**
- Consumes: `_policy`, `_preprocessor`, `_postprocessor`, `_device`
- Produces: `_predict_with_model(model_obs: dict) -> dict`

- [ ] **Step 1: Implement `_load_model`**

Mirror `eval_g1.py`: import LeRobot modules lazily, load config, dataset, policy, stats, and processors.

- [ ] **Step 2: Implement `_predict_with_model`**

Run preprocessor, `policy.select_action`, postprocessor under inference mode and split the output vector.

- [ ] **Step 3: Run fallback tests**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m unittest example.test_example_policy_adapter -v`
Expected: PASS.

- [ ] **Step 4: Run local client fallback validation**

Run server without `UNIBOT_POLICY_PATH`, then run client.
Expected: all client steps pass.

- [ ] **Step 5: Provide model server and client commands**

Include the `conda activate lerobot_cyl`, `PYTHONPATH`, `UNIBOT_POLICY_PATH`, `UNIBOT_REPO_ID`, and `run_client.py` commands in the final response.
