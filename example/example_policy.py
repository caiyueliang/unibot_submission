"""Reference policy: emits zero actions in the declared control space.
Replace the body of get_action with your model. Served by run_server.py.

    UNIBOT_SUBMISSION_TOKEN=<token> UNIBOT_CONTROL_SPACE=joint|ee
"""

import os

import numpy as np

CONTROL_SPACES = ("joint", "ee")


class ExamplePolicy:
    """Stub policy; token and control_space are read from env vars."""

    OBS_CHUNK_SIZE = 2
    ACTION_CHUNK_SIZE = 8
    DATA_KEYS = (
        "observation.language",
        "observation.images.cam_left_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
        "observation.state.left_arm",
        "observation.state.right_arm",
        "observation.state.left_ee_pose_gripper_base",
        "observation.state.right_ee_pose_gripper_base",
        "observation.state.left_gripper",
        "observation.state.right_gripper",
        "observation.state.lower_body",
    )

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
            "data_keys":         list(self.DATA_KEYS),
            "obs_chunk_size":    self.OBS_CHUNK_SIZE,
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
