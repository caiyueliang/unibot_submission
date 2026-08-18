# UniBot-V1 Challenge — Submission Template

> 中文版见 [README.zh.md](README.zh.md).

This document specifies the policy interface a submission must implement,
the data format exchanged across that interface, and the bundled reference
implementation.

---

## 1 · Policy interface

A submission consists of a policy class that exposes the three members
below. The class is instantiated on the participant machine and served by
`PolicyService`; the evaluator connects as a client and invokes
these members over the wire.

| Member | Kind | Purpose |
|---|---|---|
| `metadata`        | property → `dict` | Declares the control space, observation keys, temporal chunk sizes, and submission token. Transmitted once at handshake. |
| `get_action(obs)` | method            | Computes one action chunk for the supplied observation. Invoked at each evaluation step. |
| `reset()`         | method            | Clears per-episode state. Invoked at the start of each episode. |

### `metadata` schema

```python
{
    "control_space":     str,         # "joint" or "ee" — the action space (§3)
    "data_keys":         list[str],   # subset of the observation catalog (§2)
    "obs_chunk_size":    int >= 1,    # leading time axis of every observation
    "action_chunk_size": int >= 1,    # leading time axis of every action
    "token":             str,         # submission token issued by the organisers (§5)
}
```

All five fields are **required**. The evaluator reads them verbatim and does
**not** substitute a default for any missing field — a submission whose
metadata omits or malforms one of them is rejected on connect.

`control_space` selects the action space the policy operates in (§3) and is
**mutually exclusive** between `"joint"` and `"ee"`.

A minimal implementation is provided in
[`example/example_policy.py`](example/example_policy.py).

---

## 2 · Observation format

`obs` is a `dict` whose keys are drawn from the catalog below whose
values are `numpy.ndarray`. Each value carries a leading time axis of
length `obs_chunk_size` followed by the per-frame shape specified in the
table (§4).

| Key | Per-frame shape | dtype | Description |
|---|---|---|---|
| `observation.images.cam_left_high`   | `[480, 640, 3]` | `uint8`   | Left high-mounted camera, HWC, RGB |
| `observation.images.cam_right_high`  | `[480, 640, 3]` | `uint8`   | Right high-mounted camera, HWC, RGB |
| `observation.images.cam_left_wrist`  | `[480, 640, 3]` | `uint8`   | Left wrist camera, HWC, RGB |
| `observation.images.cam_right_wrist` | `[480, 640, 3]` | `uint8`   | Right wrist camera, HWC, RGB |
| `observation.state.left_arm`         | `[7]`           | `float32` | Left arm joint state |
| `observation.state.right_arm`        | `[7]`           | `float32` | Right arm joint state |
| `observation.state.left_ee_pose_gripper_base`  | `[6]` | `float32` | Left gripper end-effector pose in base frame: `xyz(3)` + `rpy(3)` |
| `observation.state.right_ee_pose_gripper_base` | `[6]` | `float32` | Right gripper end-effector pose in base frame: `xyz(3)` + `rpy(3)` |
| `observation.state.left_gripper`     | `[1]`           | `float32` | Left gripper opening |
| `observation.state.right_gripper`    | `[1]`           | `float32` | Right gripper opening |
| `observation.state.lower_body`       | `[15]`          | `float32` | Lower-body proprioception |
| `observation.language`               | scalar          | `str`     | Natural-language task instruction for the current episode |

Only the keys declared in `metadata.data_keys` are transmitted to the
policy. Declare the minimal subset required by the model.

Both the joint arm states (`observation.state.left_arm` / `right_arm`) and the
end-effector poses (`observation.state.*_ee_pose_gripper_base`) are available
in the catalog **regardless of `control_space`**. For the EE poses, `xyz` is position
(metres) and `rpy` is orientation as roll-pitch-yaw Euler angles (radians);
the layout matches the `action.*_ee_pose_gripper_base` actions (§3).

`observation.language` is the exception to the tensor layout above: its
value is a single Python `str` naming the task to perform (e.g.
`"move the block to the target position."`). It is **constant for the
episode** and carries **no leading time axis** (see §4).

---

## 3 · Action format

`get_action` returns a `dict` containing the submission token (§5) and the
action arrays for the **control space** the policy declared in
`metadata.control_space`. A policy operates in exactly one control space;
`"joint"` and `"ee"` are **mutually exclusive**. Each action value carries a
leading time axis of length `action_chunk_size` followed by the per-frame
shape (§4).

Action keys fall into two groups: **shared keys**, which must be provided under
both `joint` and `ee` and whose names / shapes / dtypes are identical across the
two (`meta.token`, the grippers, and `pivot`); and the **arm keys**, which
switch with `control_space` (`joint` → joint targets, `ee` → end-effector
poses). The three tables below list them.

**Shared keys — present in both control spaces**

| Key | Per-frame shape | dtype | Description |
|---|---|---|---|
| `meta.token`           | scalar | `str`     | Submission token issued by the organisers (§5) |
| `action.left_gripper`  | `[1]`  | `float32` | Target left gripper command |
| `action.right_gripper` | `[1]`  | `float32` | Target right gripper command |
| `action.pivot`         | `[7]`  | `float32` | Lower-body / pivot target |

**`control_space = "joint"`** — arm actions are target joint positions:

| Key | Per-frame shape | dtype | Description |
|---|---|---|---|
| `action.left_arm`  | `[7]` | `float32` | Target left arm joint position |
| `action.right_arm` | `[7]` | `float32` | Target right arm joint position |

**`control_space = "ee"`** — arm actions are target end-effector poses:

| Key | Per-frame shape | dtype | Description |
|---|---|---|---|
| `action.left_ee_pose_gripper_base`  | `[6]` | `float32` | Target left gripper EE pose in base frame: `xyz(3)` + `rpy(3)` |
| `action.right_ee_pose_gripper_base` | `[6]` | `float32` | Target right gripper EE pose in base frame: `xyz(3)` + `rpy(3)` |

For the EE poses, `xyz` is position (metres) and `rpy` is orientation as
roll-pitch-yaw Euler angles (radians) — the same layout as the
`observation.state.*_ee_pose_gripper_base` observations (§2).

---

## 4 · Leading time axis

Every observation tensor and every action tensor carries a leading time
axis whose length is governed by `metadata`.

```text
# obs_chunk_size = T_o
obs["observation.images.cam_left_high"].shape == (T_o, 480, 640, 3)
obs["observation.state.left_arm"].shape       == (T_o, 7)
obs["observation.state.lower_body"].shape     == (T_o, 15)

# action_chunk_size = T_a — shared keys (present under both joint and ee)
action["action.left_gripper"].shape  == (T_a, 1)
action["action.right_gripper"].shape == (T_a, 1)
action["action.pivot"].shape         == (T_a, 7)

# control_space == "joint": arms are joint targets
action["action.left_arm"].shape      == (T_a, 7)
action["action.right_arm"].shape     == (T_a, 7)

# control_space == "ee": arms are end-effector poses
action["action.left_ee_pose_gripper_base"].shape  == (T_a, 6)
action["action.right_ee_pose_gripper_base"].shape == (T_a, 6)
```

The axis is temporally ordered. For observations, index `0` is the oldest
frame and index `T_o − 1` is the most recent. For actions, index `0` is
executed first; successive indices are executed in order until the
evaluator re-queries `get_action`. The leading axis is present even when
the corresponding chunk size equals `1` — it is not squeezed.

`observation.language` is exempt from this rule: it is a single string,
not a tensor, and is delivered without a leading time axis.

---

## 5 · Submission token

Each accepted submission is issued a single token by the organisers. The
policy must carry it in **two places**:

1. the `token` field of `metadata` — checked once at handshake;
2. the reserved key `meta.token` in every `get_action` return value — checked
   at each step.

The evaluator rejects any handshake or action whose token is absent or does
not match the value issued for the submission.

The reference implementation reads the token from the
`UNIBOT_SUBMISSION_TOKEN` environment variable:

```python
import os
import numpy as np

class MyPolicy:
    def __init__(self):
        self._token = os.environ["UNIBOT_SUBMISSION_TOKEN"]

    @property
    def metadata(self):
        return {
            "control_space":     "joint",   # or "ee"
            "data_keys":         [...],      # subset of the catalog (§2)
            "obs_chunk_size":    2,
            "action_chunk_size": 8,
            "token":             self._token,
        }

    def get_action(self, obs):
        T = 8
        return {
            "meta.token":      self._token,
            "action.left_arm": np.zeros((T, 7), dtype=np.float32),
            # ... remaining action.* keys for the declared control_space, as in §3
        }
```

---

## 6 · Reference implementation & deployment

`example/` provides one runnable reference implementation, offered solely
as an example. Participants need only follow the `run_server.py` pattern:
wrap the policy in `PolicyService` and have it listen on a
designated port, at which point the evaluator connects.

| File | Role |
|---|---|
| [`example/example_policy.py`](example/example_policy.py) | A minimal policy exposing the §1 interface; it currently returns a constant zero action chunk. Participants may use it as a starting point and replace its body with their own model, or implement an equivalent class against the same interface. |
| [`example/example_env.py`](example/example_env.py)       | Local evaluation stub. Produces observations per §2 and validates returned actions with the same checks as the production evaluator. |
| [`example/run_server.py`](example/run_server.py)         | **Server-side entry template** — every submission is deployed by wrapping its policy in `PolicyService` and listening on a port, as demonstrated by this script. `UNIBOT_SUBMISSION_TOKEN` and `UNIBOT_CONTROL_SPACE` (`joint`/`ee`) must both be set. |
| [`example/run_client.py`](example/run_client.py)         | Local end-to-end verifier: it connects to the participant's server, steps through it, and validates each returned action. In production this client is run by the organisers; `UNIBOT_SUBMISSION_TOKEN` must match the server's token. |
| [`policy/`](policy)                                       | `PolicyService` / `RemotePolicy` and the msgpack/NumPy adapter. |
| [`requirements.txt`](requirements.txt)                   | Dependencies for the reference implementation: `numpy`, `msgpack`, `websockets`. |

### Environment

Install the dependencies into the Python environment used for the
submission:

```bash
pip install -r requirements.txt   # numpy, msgpack, websockets
```

### Workflow

Local verification is always performed through `run_server.py` and
`run_client.py`: one terminal serves the policy, while another emulates the
evaluator by connecting as a client. The token must match on both ends.

1. **Run the reference first.** Exercise the unmodified `ExamplePolicy`
   end-to-end and confirm a correct run (the client prints `token verified`
   at each step, with no `ACTION REJECTED`) to establish the expected
   behaviour of a full pass:
   ```bash
   # Terminal A
   UNIBOT_SUBMISSION_TOKEN=dev-token UNIBOT_CONTROL_SPACE=joint UNIBOT_SERVER_PORT=8765 python example/run_server.py

   # Terminal B
   UNIBOT_SUBMISSION_TOKEN=dev-token python example/run_client.py
   ```
2. **Integrate your model.** Implement the §1 interface and substitute
   it for `ExamplePolicy` in `run_server.py`; replace the zero tensors in
   `get_action` with your model's outputs and adjust `control_space`,
   `data_keys`, `obs_chunk_size`, and `action_chunk_size` as required.
3. **Run it again to confirm.** Repeat the two-terminal commands from step 1
   and verify that the client still passes at every step with your model in
   place; this confirms that the interface is correctly integrated.
4. **Deploy.** Write the token issued by the organisers into
   `UNIBOT_SUBMISSION_TOKEN`, start the server with `run_server.py`, and keep it
   online; the evaluator will connect on its own.

   Evaluation runs across **N test robots** in randomized order (the organisers
   announce the exact N separately), and the evaluator connects to one fixed
   port per robot. A submission must therefore expose **N consecutive ports**
   starting at `8765` (`8765`, `8766`, …, `8765 + N − 1`); **the number of
   exposed ports must exactly equal the number of robots N**, and each port must
   independently and reliably serve the entire interaction of its robot. How
   each port is implemented behind the scenes is left to the participant — two
   common approaches:

   **Approach 1 · One independent server per port (most robust).** Run N
   processes, each listening on one port and loading its own copy of the model;
   they are isolated and simplest to operate, at the cost of N copies of the
   model in GPU / host memory (expensive for large models). For **N = 3**, expose
   `8765`–`8767`, one instance each:

   ```bash
   UNIBOT_SUBMISSION_TOKEN=<token> UNIBOT_CONTROL_SPACE=<joint|ee> UNIBOT_SERVER_PORT=8765 python example/run_server.py
   UNIBOT_SUBMISSION_TOKEN=<token> UNIBOT_CONTROL_SPACE=<joint|ee> UNIBOT_SERVER_PORT=8766 python example/run_server.py
   UNIBOT_SUBMISSION_TOKEN=<token> UNIBOT_CONTROL_SPACE=<joint|ee> UNIBOT_SERVER_PORT=8767 python example/run_server.py
   ```

   `UNIBOT_SERVER_PORT` is the port that instance listens on. These instances may
   run on one machine or be spread across several, but all N ports must be
   reachable from outside.

   **Approach 2 · Share one model across ports (resource-saving).** When the
   model is too heavy to hold N copies, several ports can share a single model,
   but the participant must implement request dispatch on the server side (N
   lightweight endpoints each listen on one port and forward requests to a shared
   inference backend). Two things to watch:

   - **Concurrency:** different ports may be called at the same time, so the
     shared backend must serve concurrent requests without blocking one another
     and inflating latency;
   - **State isolation:** each port is a separate robot running a separate
     episode, so the `reset` and history state of the ports must be kept strictly
     isolated and never shared.

---

## 7 · Interaction quality and result credibility

During evaluation the policy is queried over the network at every step, and
the timing and content of each `get_action` response are recorded. A
submission is expected to sustain smooth and consistent interaction timing
and to keep its returned values free of unnecessary anomalies.

When assessing the credibility of a result, the organisers may take into
account the full range of interaction metrics collected during evaluation,
including per-step response latency and its variability, together with the
statistical regularity of the returned actions, and these metrics may be
presented alongside the published result. Participants are accordingly
advised to keep per-step inference latency stable, to ensure returned actions
are well-formed and free of spurious outliers, and to avoid any behaviour that
introduces irregular interaction patterns.
