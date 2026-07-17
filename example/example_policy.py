"""参考策略示例：按声明的控制空间返回全零动作。

真正提交时，通常只需要把 ``get_action`` 里的全零动作替换成自己的模型推理逻辑；
其余 metadata / reset / token 处理可以作为接口模板参考。本策略由
``run_server.py`` 创建并通过 WebSocket 对外提供服务。

    UNIBOT_SUBMISSION_TOKEN=<token> UNIBOT_CONTROL_SPACE=joint|ee
"""

import os

import numpy as np

CONTROL_SPACES = ("joint", "ee")


class ExamplePolicy:
    """最小可运行的 policy；token 和 control_space 从环境变量读取。"""

    # policy 每次接收的观测 chunk 长度。评测环境会按这个长度构造 observation。
    OBS_CHUNK_SIZE = 2
    # policy 每次返回的动作 chunk 长度。所有动作数组第一维都必须等于这个值。
    ACTION_CHUNK_SIZE = 8
    # policy 需要的观测 key 列表。服务端会把这份声明放进 metadata 发给客户端。
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
        """从环境变量读取提交 token 和控制空间；二者都是必填项。"""
        self._token = os.environ.get("UNIBOT_SUBMISSION_TOKEN")
        self._control_space = os.environ.get("UNIBOT_CONTROL_SPACE")
        # 示例内部状态：记录 get_action 被调用次数。真实模型可替换成自己的 episode 状态。
        self._step = 0
        if self._token is None:
            raise ValueError("UNIBOT_SUBMISSION_TOKEN is required")
        if self._control_space not in CONTROL_SPACES:
            raise ValueError(f"unknown control_space {self._control_space!r}; expected one of {CONTROL_SPACES}")

    @property
    def metadata(self):
        """WebSocket 握手时发送给评测端的数据契约。"""
        return {
            "control_space":     self._control_space,
            "data_keys":         list(self.DATA_KEYS),
            "obs_chunk_size":    self.OBS_CHUNK_SIZE,
            "action_chunk_size": self.ACTION_CHUNK_SIZE,
            "token":             self._token,
        }

    def get_action(self, obs):
        """根据当前控制空间返回一段动作 chunk。

        参数 ``obs`` 是评测端传来的观测字典，key 与 ``DATA_KEYS`` 对应。这里为了示例
        没有使用观测内容，而是直接返回全零动作；接入真实模型时应在这里做预处理、
        模型推理和后处理。
        """
        self._step += 1
        T = self.ACTION_CHUNK_SIZE

        # 两种控制空间都需要返回 gripper、pivot 和 meta.token。
        action = {
            "meta.token":           self._token,
            "action.left_gripper":  np.zeros((T, 1), dtype=np.float32),
            "action.right_gripper": np.zeros((T, 1), dtype=np.float32),
            "action.pivot":         np.zeros((T, 7), dtype=np.float32),
        }
        if self._control_space == "joint":
            # joint 控制空间：左右机械臂动作是 7 维关节量。
            action["action.left_arm"]  = np.zeros((T, 7), dtype=np.float32)
            action["action.right_arm"] = np.zeros((T, 7), dtype=np.float32)
        else:
            # ee 控制空间：左右末端执行器 pose 是 6 维位姿量。
            action["action.left_ee_pose_gripper_base"]  = np.zeros((T, 6), dtype=np.float32)
            action["action.right_ee_pose_gripper_base"] = np.zeros((T, 6), dtype=np.float32)
        return action

    def reset(self):
        """清空每个 episode 级别的状态。"""
        self._step = 0
        return {"ok": True}
