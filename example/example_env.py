"""Reference evaluation env: builds stub observations from a policy's metadata
and validates the actions it returns. Used by run_client.py, not run directly."""

from __future__ import annotations

import hmac
import os
import warnings
from typing import Mapping

import numpy as np

# Per-frame (shape, dtype) for each observation key.
OBSERVATION_SPEC: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "observation.images.cam_left_high":   ((480, 640, 3), np.dtype("uint8")),
    "observation.images.cam_right_high":  ((480, 640, 3), np.dtype("uint8")),
    "observation.images.cam_left_wrist":  ((480, 640, 3), np.dtype("uint8")),
    "observation.images.cam_right_wrist": ((480, 640, 3), np.dtype("uint8")),
    "observation.state.left_arm":         ((7,),          np.dtype("float32")),
    "observation.state.right_arm":        ((7,),          np.dtype("float32")),
    "observation.state.left_ee_pose_gripper_base":  ((6,), np.dtype("float32")),
    "observation.state.right_ee_pose_gripper_base": ((6,), np.dtype("float32")),
    "observation.state.left_gripper":     ((1,),          np.dtype("float32")),
    "observation.state.right_gripper":    ((1,),          np.dtype("float32")),
    "observation.state.lower_body":       ((15,),         np.dtype("float32")),
    "observation.language":               (None,           np.dtype("str")),
}

# Per-frame shape of each action key, one dict per control space.
JOINT_ACTION_SPEC = {
    "action.left_arm":  (7,),
    "action.right_arm": (7,),
    "action.left_gripper":  (1,),
    "action.right_gripper": (1,),
    "action.pivot":         (7,),
    "meta.token":            None,
}

EE_ACTION_SPEC = {
    "action.left_ee_pose_gripper_base":  (6,),
    "action.right_ee_pose_gripper_base": (6,),
    "action.left_gripper":  (1,),
    "action.right_gripper": (1,),
    "action.pivot":         (7,),
    "meta.token":            None,
}

CONTROL_SPACES = ("joint", "ee")
DEFAULT_LANGUAGE = "move the block to the target position."  # placeholder task instruction


class ActionError(ValueError):
    """Raised when a policy's `get_action` response violates the spec."""


class ExampleEnv:
    """Validates a policy's metadata and actions; serves stub observations."""

    def __init__(self, metadata: dict):
        """Validate the metadata and pick the action spec for its control space."""
        self.policy_metadata = metadata
        self._expected_token = os.environ.get("UNIBOT_SUBMISSION_TOKEN")
        if self._expected_token is None:
            raise ValueError("UNIBOT_SUBMISSION_TOKEN is required")
        self.validate_token(self.policy_metadata["token"])

        self.obs_delta_indices = self.policy_metadata["obs_delta_indices"]
        self.image_resize = self.policy_metadata["image_resize"]
        self.action_chunk_size = self.policy_metadata["action_chunk_size"]
        self.control_space = self.policy_metadata["control_space"]

        self.validate_obs_delta_indices(self.obs_delta_indices)
        self.validate_image_resize(self.image_resize, self.obs_delta_indices)
        if not (type(self.action_chunk_size) == int and self.action_chunk_size > 0):
            raise ActionError(f"action_chunk_size must be a positive int, got {self.action_chunk_size!r}")
        if self.control_space not in CONTROL_SPACES:
            raise ActionError(f"unknown control_space {self.control_space!r}; expected one of {CONTROL_SPACES}")
        self.action_spec = JOINT_ACTION_SPEC if self.control_space == "joint" else EE_ACTION_SPEC

    @staticmethod
    def validate_obs_delta_indices(obs_delta_indices):
        """Check the per-key temporal stacking spec (§2).

        Each value is the list of frame offsets from the current step: all
        offsets must be non-positive ints in strictly increasing order, so
        index 0 is the oldest frame and the last index is the newest. The last
        offset need not be 0, but a key whose last offset is not 0 does not
        receive the most recent observation, so a warning is emitted for it.
        `observation.language` is a scalar string with no time axis, so its
        offset list must be exactly [0].
        """
        if not isinstance(obs_delta_indices, Mapping) or len(obs_delta_indices) == 0:
            raise ActionError("obs_delta_indices must be a non-empty mapping of key -> frame offsets")
        for key, offsets in obs_delta_indices.items():
            if key not in OBSERVATION_SPEC:
                raise ActionError(f"unknown observation key {key!r}; expected one of {OBSERVATION_SPEC.keys()}")
            if not (isinstance(offsets, (list, tuple)) and len(offsets) > 0):
                raise ActionError(f"obs_delta_indices[{key!r}] must be a non-empty list of frame offsets, got {offsets!r}")
            if not all(type(o) == int for o in offsets):
                raise ActionError(f"obs_delta_indices[{key!r}] offsets must all be ints, got {offsets!r}")
            if not all(o <= 0 for o in offsets):
                raise ActionError(f"obs_delta_indices[{key!r}] offsets must all be <= 0, got {offsets!r}")
            if not all(a < b for a, b in zip(offsets, offsets[1:])):
                raise ActionError(f"obs_delta_indices[{key!r}] offsets must be strictly increasing, got {offsets!r}")
            if key == "observation.language" and list(offsets) != [0]:
                raise ActionError(f"observation.language carries no time axis; its offsets must be [0], got {offsets!r}")
            if offsets[-1] != 0:
                warnings.warn(
                    f"obs_delta_indices[{key!r}] ends at offset {offsets[-1]} (not 0); "
                    f"this key will not use the most recent observation",
                    stacklevel=2,
                )


    @staticmethod
    def validate_image_resize(image_resize, obs_delta_indices):
        """Check the per-image resize map (§2).

        Maps an image key to the [height, width] the client should resize that
        image to before sending it. Every key must be an image key that also
        appears in obs_delta_indices, and each value is a [height, width] pair of
        positive ints. The map may be empty — images left out keep their native
        catalog resolution.
        """
        if not isinstance(image_resize, Mapping):
            raise ActionError(f"image_resize must be a mapping of image key -> [height, width], got {image_resize!r}")
        for key, size in image_resize.items():
            if not key.startswith("observation.images."):
                raise ActionError(f"image_resize key {key!r} is not an image key")
            if key not in OBSERVATION_SPEC:
                raise ActionError(f"unknown image key {key!r}; expected one of {OBSERVATION_SPEC.keys()}")
            if key not in obs_delta_indices:
                raise ActionError(f"image_resize key {key!r} must also appear in obs_delta_indices")
            if not (isinstance(size, (list, tuple)) and len(size) == 2):
                raise ActionError(f"image_resize[{key!r}] must be a [height, width] pair, got {size!r}")
            if not all(type(s) == int and s > 0 for s in size):
                raise ActionError(f"image_resize[{key!r}] must be two positive ints, got {size!r}")



    def validate_token(self, token: str):
        """Reject a token that doesn't match UNIBOT_SUBMISSION_TOKEN."""
        if not hmac.compare_digest(token, self._expected_token):
            raise ActionError("invalid meta.token: does not match UNIBOT_SUBMISSION_TOKEN")

    def reset(self):
        """Start an episode; return the first observation."""
        return self.get_observation()

    def step(self, action: dict):
        """Validate one action; return the next observation."""
        self.validate_token(action["meta.token"])

        for action_key in self.action_spec:
            if action_key not in action:
                raise ActionError(f"missing action key {action_key!r}; expected one of {self.action_spec.keys()}")

        for action_key in action:
            if action_key not in self.action_spec:
                raise ActionError(f"unknown action key {action_key!r}; expected one of {self.action_spec.keys()}")
            if action_key != "meta.token":
                if not isinstance(action[action_key], np.ndarray):
                    raise ActionError(f"action[{action_key}] must be a numpy.ndarray, got {type(action[action_key]).__name__}")
                if action[action_key].shape != (self.action_chunk_size, *self.action_spec[action_key]):
                    raise ActionError(f"action[{action_key}].shape {action[action_key].shape} != expected {(self.action_chunk_size, *self.action_spec[action_key])}")
                if action[action_key].dtype != np.float32:
                    raise ActionError(f"action[{action_key}].dtype {action[action_key].dtype} != np.float32")
        return self.get_observation()

    def get_observation(self):
        """Build a spec-shaped stub observation (zeros; language is a string).

        Each key's leading time axis has one entry per declared frame offset,
        so its length is len(obs_delta_indices[key]). Image keys listed in
        image_resize use the requested [height, width] instead of the native
        catalog resolution.
        """
        observation = {}
        for key, offsets in self.obs_delta_indices.items():
            if key == "observation.language":
                observation[key] = DEFAULT_LANGUAGE
                continue
            frame_shape, dtype = OBSERVATION_SPEC[key]
            if key in self.image_resize:
                height, width = self.image_resize[key]
                frame_shape = (height, width, frame_shape[2])
            shape = (len(offsets),) + frame_shape
            observation[key] = np.zeros(shape, dtype=dtype)
        return observation
