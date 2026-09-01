# ExamplePolicy Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt `example/example_policy.py` between the UniBot README service contract and the model dataset feature names.

**Architecture:** Keep the WebSocket service and local evaluator unchanged. Put observation key mapping, model inference hook, and action normalization inside `ExamplePolicy`.

**Tech Stack:** Python, NumPy, unittest, existing `PolicyService`/`RemotePolicy`.

## Global Constraints

Only modify `example/example_policy.py` and focused adapter tests. Preserve `example/example_policy_back.py`. Return README-compliant joint action keys with `np.float32` arrays shaped `(action_chunk_size, dim)`.

---

### Task 1: ExamplePolicy Adapter

**Files:**
- Modify: `example/example_policy.py`
- Create: `example/test_example_policy_adapter.py`

**Interfaces:**
- Consumes: `ExamplePolicy.get_action(obs: dict) -> dict`
- Produces: `_adapt_observation(obs: dict) -> dict`, `_predict_model_action(model_obs: dict) -> dict`, `_adapt_action(model_action: dict, obs: dict) -> dict`

- [ ] **Step 1: Write the failing test**

```python
def test_adapts_evaluator_observation_keys_to_dataset_keys(self):
    policy = ExamplePolicy()
    model_obs = policy._adapt_observation(self.obs)
    self.assertIn("observation.images.head_stereo_left", model_obs)
    self.assertIn("observation.images.wrist_left", model_obs)
    self.assertIn("observation.images.wrist_right", model_obs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest example.test_example_policy_adapter -v`
Expected: FAIL because `_adapt_observation` is not implemented yet.

- [ ] **Step 3: Write minimal implementation**

Implement `ExamplePolicy` helpers that map evaluator observation keys, expose `_predict_model_action`, normalize model actions to chunks, and return token plus joint action keys.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest example.test_example_policy_adapter -v`
Expected: PASS.

- [ ] **Step 5: Run end-to-end validation**

Run server with `UNIBOT_SUBMISSION_TOKEN=dev-token UNIBOT_CONTROL_SPACE=joint python example/run_server.py 8765`, then run client with `UNIBOT_SUBMISSION_TOKEN=dev-token python example/run_client.py`.
Expected: all client steps pass validation.
