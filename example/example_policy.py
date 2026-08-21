"""Reference policy: emits zero actions in the declared control space.
Replace the body of get_action with your model. Served by run_server.py.

    UNIBOT_SUBMISSION_TOKEN=<token> UNIBOT_CONTROL_SPACE=joint|ee
"""

import os

import numpy as np

CONTROL_SPACES = ("joint", "ee")


class ExamplePolicy:
    """Stub policy; token and control_space are read from env vars."""

    ACTION_CHUNK_SIZE = 8
    # Per-key temporal stacking: obs_delta_indices[key][i] is frame i's offset
    # from the current step, so a key's observation carries len(value) stacked
    # frames. Its keys also select which observations are sent — declare only
    # the observations the model consumes. Offsets are <= 0 and strictly
    # increasing, so index 0 is the oldest frame and the last index is the
    # newest; e.g. [-4, -2, 0] stacks the frames at t-4, t-2 and the current
    # step t, and [0] is a single current frame. Non-contiguous spacing such as
    # [-10, -5, -2, 0] is allowed. `observation.language` is a scalar string
    # with no time axis, so its offset must be exactly [0].
    OBS_DELTA_INDICES = {
        "observation.language":                         [0],
        "observation.images.cam_left_high":             [-10, -5, -2, 0],
        "observation.images.cam_left_wrist":            [-10, -5, -2, 0],
        "observation.images.cam_right_wrist":           [-10, -5, -2, 0],
        "observation.state.left_arm":                   [-4, -2, 0],
        "observation.state.right_arm":                  [-4, -2, 0],
        "observation.state.left_ee_pose_gripper_base":  [-4, -2, 0],
        "observation.state.right_ee_pose_gripper_base": [-4, -2, 0],
        "observation.state.left_gripper":               [-4, -2, 0],
        "observation.state.right_gripper":              [-4, -2, 0],
        "observation.state.lower_body":                 [-4, -2, 0],
    }
    # Ask the client to resize these image keys to [height, width] before
    # sending them; each key must also appear in OBS_DELTA_INDICES. Image keys
    # left out of this map are delivered at their native catalog resolution.
    IMAGE_RESIZE = {
        "observation.images.cam_left_high":  [240, 320],
        "observation.images.cam_left_wrist": [128, 128],
    }

    def __init__(self):
        """Read token and control_space from the environment; both required."""
        self._token = os.environ.get("UNIBOT_SUBMISSION_TOKEN")
        self._control_space = os.environ.get("UNIBOT_CONTROL_SPACE")
        self._step = 0
        if self._token is None:
            raise ValueError("UNIBOT_SUBMISSION_TOKEN is required")
        if self._control_space not in CONTROL_SPACES:
            raise ValueError(f"unknown control_space {self._control_space!r}; expected one of {CONTROL_SPACES}")

    @property
    def metadata(self):
        """Contract sent to the evaluator at handshake."""
        return {
            "control_space":     self._control_space,
            "obs_delta_indices": {k: list(v) for k, v in self.OBS_DELTA_INDICES.items()},
            "image_resize":      {k: list(v) for k, v in self.IMAGE_RESIZE.items()},
            "action_chunk_size": self.ACTION_CHUNK_SIZE,
            "token":             self._token,
        }

    def get_action(self, obs):
        """Return one zero action chunk for the active control space."""
        self._step += 1
        T = self.ACTION_CHUNK_SIZE

        action = {
            "meta.token":           self._token,
            "action.left_gripper":  np.zeros((T, 1), dtype=np.float32),
            "action.right_gripper": np.zeros((T, 1), dtype=np.float32),
            "action.pivot":         np.zeros((T, 7), dtype=np.float32),
        }
        if self._control_space == "joint":
            action["action.left_arm"]  = np.zeros((T, 7), dtype=np.float32)
            action["action.right_arm"] = np.zeros((T, 7), dtype=np.float32)
        else:
            action["action.left_ee_pose_gripper_base"]  = np.zeros((T, 6), dtype=np.float32)
            action["action.right_ee_pose_gripper_base"] = np.zeros((T, 6), dtype=np.float32)
        return action

    def reset(self):
        """Clear per-episode state."""
        self._step = 0
        return {"ok": True}
