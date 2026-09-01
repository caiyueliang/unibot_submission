"""参考评测环境。

这个文件不会直接启动真正的机器人环境，而是根据 policy.metadata 构造一份形状
正确的假观测（stub observation），再校验 policy 返回的动作是否满足提交规范。
它主要由 ``run_client.py`` 调用，用来在本地快速检查服务端策略是否能通过接口约束。
"""

from __future__ import annotations

import hmac
import os
import warnings
from typing import Mapping

import numpy as np

# 每个 observation key 对应“单帧”的形状和 dtype。
# 实际传给 policy 时会在最前面再加 obs_chunk_size 这个时间维度。
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

# 每个 action key 对应“单步动作”的形状；实际返回时还要加 action_chunk_size 时间维度。
# joint 和 ee 是两套不同控制空间，因此各自有一份动作规格。
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
DEFAULT_LANGUAGE = "move the block to the target position."  # 占位任务指令文本


class ActionError(ValueError):
    """当 policy 的 ``get_action`` 返回结果不符合规范时抛出。"""


class ExampleEnv:
    """校验 policy metadata 和动作输出，并生成假观测供本地联调使用。"""

    def __init__(self, metadata: dict):
        """校验 metadata，并根据 control_space 选择后续动作校验规则。"""
        self.policy_metadata = metadata
        # token 用于确认服务端 policy 和本地评测端使用的是同一份提交凭证。
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
        # 后续 step(action) 会使用这份 action_spec 校验动作 key、shape、dtype。
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
        """校验 token 是否与环境变量 ``UNIBOT_SUBMISSION_TOKEN`` 完全一致。"""
        # hmac.compare_digest 可以避免普通字符串比较中的时序侧信道差异。
        if not hmac.compare_digest(token, self._expected_token):
            raise ActionError("invalid meta.token: does not match UNIBOT_SUBMISSION_TOKEN")

    def reset(self):
        """开始一个 episode，并返回第一帧/第一段观测。"""
        return self.get_observation()

    def step(self, action: dict):
        """校验一段动作 chunk；若合法则返回下一段假观测。"""
        self.validate_token(action["meta.token"])

        # 先检查必需动作 key 是否全部存在，避免后面访问时出现 KeyError。
        for action_key in self.action_spec:
            if action_key not in action:
                raise ActionError(f"missing action key {action_key!r}; expected one of {self.action_spec.keys()}")

        # 再检查是否包含未知 key，以及每个数组的类型、shape、dtype 是否符合规范。
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
                # 语言指令是字符串，不需要 numpy 数组，也没有时间维 shape。
                observation[key] = DEFAULT_LANGUAGE
                continue
            frame_shape, dtype = OBSERVATION_SPEC[key]
            if key in self.image_resize:
                height, width = self.image_resize[key]
                frame_shape = (height, width, frame_shape[2])
            shape = (len(offsets),) + frame_shape
            observation[key] = np.zeros(shape, dtype=dtype)
        return observation
