"""参考评测环境。

这个文件不会直接启动真正的机器人环境，而是根据 policy.metadata 构造一份形状
正确的假观测（stub observation），再校验 policy 返回的动作是否满足提交规范。
它主要由 ``run_client.py`` 调用，用来在本地快速检查服务端策略是否能通过接口约束。
"""

from __future__ import annotations

import hmac
import os
from typing import Iterable, Mapping

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

        # metadata 是服务端在 WebSocket 握手时发来的接口契约。
        self.data_keys = self.policy_metadata["data_keys"]
        self.obs_chunk_size = self.policy_metadata["obs_chunk_size"]
        self.action_chunk_size = self.policy_metadata["action_chunk_size"]
        self.control_space = self.policy_metadata["control_space"]

        # 逐项检查服务端声明的观测 key 是否在评测环境支持范围内。
        for key in self.data_keys:
            if key not in OBSERVATION_SPEC:
                raise ActionError(f"unknown observation key {key!r}; expected one of {OBSERVATION_SPEC.keys()}")
        if len(self.data_keys) == 0:
            raise ActionError("data_keys must be a non-empty list")
        if not (type(self.obs_chunk_size) == int and self.obs_chunk_size > 0):
            raise ActionError(f"obs_chunk_size must be a positive int, got {self.obs_chunk_size!r}")
        if not (type(self.action_chunk_size) == int and self.action_chunk_size > 0):
            raise ActionError(f"action_chunk_size must be a positive int, got {self.action_chunk_size!r}")
        if self.control_space not in CONTROL_SPACES:
            raise ActionError(f"unknown control_space {self.control_space!r}; expected one of {CONTROL_SPACES}")
        # 后续 step(action) 会使用这份 action_spec 校验动作 key、shape、dtype。
        self.action_spec = JOINT_ACTION_SPEC if self.control_space == "joint" else EE_ACTION_SPEC

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
        """按 metadata 中声明的 data_keys 构造形状正确的假观测。"""
        observation = {}
        for key in self.data_keys:
            if key == "observation.language":
                # 语言指令是字符串，不需要 numpy 数组，也没有时间维 shape。
                observation[key] = DEFAULT_LANGUAGE
                continue
            frame_shape, dtype = OBSERVATION_SPEC[key]
            # 在单帧 shape 前加 obs_chunk_size，形成输入给 policy 的时间窗口。
            shape = (self.obs_chunk_size,) + frame_shape
            observation[key] = np.zeros(shape, dtype=dtype)
        return observation
