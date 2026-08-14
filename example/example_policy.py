"""模型服务适配示例：在评测接口和模型训练格式之间转换数据。

评测端使用 ``README.zh.md`` 中定义的 observation/action key；训练数据集
``G1_Dex1_ArrangeTestTubes_3cams`` 使用另一组相机 key。本文件把这层差异收在
``ExamplePolicy`` 内部：服务 metadata 和返回动作仍完全遵守 README，真实模型只需要
接收 ``_adapt_observation`` 产出的 dataset-style observation，并从
``_predict_model_action`` 返回 dataset-style action。

    UNIBOT_SUBMISSION_TOKEN=<token> UNIBOT_CONTROL_SPACE=joint|ee
"""

import os
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

CONTROL_SPACES = ("joint", "ee")


class ExamplePolicy:
    """可运行的 policy 适配层；token 和 control_space 从环境变量读取。"""

    # policy 每次接收的观测 chunk 长度。评测环境会按这个长度构造 observation。
    OBS_CHUNK_SIZE = int(os.environ.get("UNIBOT_OBS_CHUNK_SIZE", "1"))
    # policy 每次返回的动作 chunk 长度。所有动作数组第一维都必须等于这个值。
    ACTION_CHUNK_SIZE = int(os.environ.get("UNIBOT_ACTION_CHUNK_SIZE", "1"))
    # policy 需要的观测 key 列表。服务端会把这份声明放进 metadata 发给客户端。
    DATA_KEYS = (
        "observation.language",                          # 当前 episode 的自然语言任务指令，标量 str。
        "observation.images.cam_left_high",              # 左上方相机图像，单帧形状 [480, 640, 3]，HWC，RGB，uint8。
        "observation.images.cam_left_wrist",             # 左手腕相机图像，单帧形状 [480, 640, 3]，HWC，RGB，uint8。
        "observation.images.cam_right_wrist",            # 右手腕相机图像，单帧形状 [480, 640, 3]，HWC，RGB，uint8。
        "observation.state.left_arm",                    # 左臂 7 维关节状态，float32。
        "observation.state.right_arm",                   # 右臂 7 维关节状态，float32。
        # "observation.state.left_ee_pose_gripper_base",   # PI0.5 split state/action 训练未使用。
        # "observation.state.right_ee_pose_gripper_base",  # PI0.5 split state/action 训练未使用。
        "observation.state.left_gripper",                # 左夹爪 1 维开度，float32。
        "observation.state.right_gripper",               # 右夹爪 1 维开度，float32。
        # "observation.state.lower_body",                  # PI0.5 split state/action 训练未使用。
    )
    # README 评测 key -> 数据集 / 模型训练 key。
    OBSERVATION_KEY_MAP = {
        "observation.images.cam_left_high": "observation.images.head_stereo_left",
        "observation.images.cam_left_wrist": "observation.images.wrist_left",
        "observation.images.cam_right_wrist": "observation.images.wrist_right",
    }

    def __init__(self):
        """从环境变量读取提交 token 和控制空间；二者都是必填项。"""
        self._token = os.environ.get("UNIBOT_SUBMISSION_TOKEN")
        self._control_space = os.environ.get("UNIBOT_CONTROL_SPACE", "joint")
        self._policy_path = os.environ.get("UNIBOT_POLICY_PATH")
        self._repo_id = os.environ.get("UNIBOT_REPO_ID")
        self._default_task = os.environ.get("UNIBOT_DEFAULT_TASK", "")
        self._policy = None
        self._preprocessor = None
        self._postprocessor = None
        self._device = None
        self._use_amp = False
        self._model_enabled = False
        self.OBS_CHUNK_SIZE = int(os.environ.get("UNIBOT_OBS_CHUNK_SIZE", str(self.OBS_CHUNK_SIZE)))
        self.ACTION_CHUNK_SIZE = int(os.environ.get("UNIBOT_ACTION_CHUNK_SIZE", str(self.ACTION_CHUNK_SIZE)))
        # 示例内部状态：记录 get_action 被调用次数。真实模型可替换成自己的 episode 状态。
        self._step = 0
        if self._token is None:
            raise ValueError("UNIBOT_SUBMISSION_TOKEN is required")
        if self._control_space not in CONTROL_SPACES:
            raise ValueError(f"unknown control_space {self._control_space!r}; expected one of {CONTROL_SPACES}")
        if self._policy_path is not None:
            if not self._repo_id:
                raise ValueError("UNIBOT_REPO_ID is required when UNIBOT_POLICY_PATH is set")
            self._load_model()

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

        参数 ``obs`` 是评测端传来的观测字典，key 与 ``DATA_KEYS`` 对应。接入真实模型
        时，把 ``_predict_model_action`` 替换成自己的推理逻辑即可；该方法前后会负责
        key 映射、shape/dtype 归一化，以及 token 注入。
        """
        self._step += 1
        model_obs = self._adapt_observation(obs)
        model_action = self._predict_model_action(model_obs)
        return self._adapt_action(model_action, obs)

    def _adapt_observation(self, obs):
        """把 README observation key 转成模型训练时使用的 dataset feature key。"""
        model_obs = {}
        for key, value in obs.items():
            model_key = self.OBSERVATION_KEY_MAP.get(key, key)
            model_obs[model_key] = value
        return model_obs

    def _predict_model_action(self, model_obs):
        """模型推理入口。

        默认实现用于本地联调：保持最新一帧双臂和夹爪状态，pivot 置零。接真实模型时，
        在这里读取 ``model_obs`` 并返回同名 action dict。
        """
        if self._model_enabled:
            return self._predict_with_model(model_obs)
        return {
            "action.left_arm": self._latest_or_zeros(model_obs, "observation.state.left_arm", 7),
            "action.right_arm": self._latest_or_zeros(model_obs, "observation.state.right_arm", 7),
            "action.left_gripper": self._latest_or_zeros(model_obs, "observation.state.left_gripper", 1),
            "action.right_gripper": self._latest_or_zeros(model_obs, "observation.state.right_gripper", 1),
            "action.pivot": np.zeros((7,), dtype=np.float32),
            "action.left_ee_pose_gripper_base": np.zeros((6,), dtype=np.float32),
            "action.right_ee_pose_gripper_base": np.zeros((6,), dtype=np.float32),
        }

    def _load_model(self):
        """按 ``eval_g1.py`` 的最小链路加载 LeRobot policy 和 processors。"""
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.policies.pi05.dataset_adapter import adapt_pi05_stats
        from lerobot.processor.rename_processor import rename_stats
        from lerobot.utils.utils import get_safe_torch_device

        policy_cfg = PreTrainedConfig.from_pretrained(self._policy_path)
        policy_cfg.pretrained_path = self._policy_path
        self.OBS_CHUNK_SIZE = int(os.environ.get("UNIBOT_OBS_CHUNK_SIZE", getattr(policy_cfg, "n_obs_steps", 1)))
        self.ACTION_CHUNK_SIZE = int(
            os.environ.get(
                "UNIBOT_ACTION_CHUNK_SIZE",
                getattr(policy_cfg, "n_action_steps", getattr(policy_cfg, "chunk_size", self.ACTION_CHUNK_SIZE)),
            )
        )

        dataset_meta = self._load_dataset_meta(self._repo_id)
        self._policy = make_policy(cfg=policy_cfg, ds_meta=dataset_meta)
        self._policy.eval()

        use_pi05_split_state_action = policy_cfg.type == "pi05" and getattr(
            policy_cfg,
            "use_split_state_action",
            False,
        )
        dataset_stats = adapt_pi05_stats(dataset_meta.stats) if use_pi05_split_state_action else dataset_meta.stats
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_cfg.pretrained_path,
            dataset_stats=rename_stats(dataset_stats, {}),
            preprocessor_overrides={
                "device_processor": {"device": policy_cfg.device},
                "rename_observations_processor": {"rename_map": {}},
            },
        )
        self._device = get_safe_torch_device(policy_cfg.device, log=True)
        self._use_amp = bool(getattr(policy_cfg, "use_amp", False))
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        self._model_enabled = True

    @staticmethod
    def _load_dataset_meta(repo_id):
        """只读取 LeRobot meta/info.json 和 stats.json，避免加载数据帧/episodes。"""
        meta_dir = Path(repo_id) / "meta"
        with (meta_dir / "info.json").open("r", encoding="utf-8") as f:
            info = json.load(f)
        with (meta_dir / "stats.json").open("r", encoding="utf-8") as f:
            stats = json.load(f)
        return SimpleNamespace(features=info["features"], stats=stats)

    def _predict_with_model(self, model_obs):
        """运行 LeRobot policy，并拆成 dataset-style action dict。"""
        import torch
        from contextlib import nullcontext

        prepared_obs, task = self._prepare_policy_observation(model_obs)
        observation = dict(prepared_obs)
        with (
            torch.inference_mode(),
            torch.autocast(device_type=self._device.type) if self._device.type == "cuda" and self._use_amp else nullcontext(),
        ):
            for name in list(observation):
                value = observation[name]
                if not hasattr(value, "unsqueeze"):
                    continue
                if "images" in name:
                    value = value.type(torch.float32) / 255
                    value = value.permute(2, 0, 1).contiguous()
                observation[name] = value.unsqueeze(0).to(self._device)
            observation["task"] = task
            observation["robot_type"] = ""
            observation = self._preprocessor(observation)
            action = self._policy.select_action(observation)
            action = self._postprocessor(action)
            action = action.squeeze(0).to("cpu")
        return self._split_model_action_vector(action)

    def _prepare_policy_observation(self, model_obs):
        """取最新帧并补充 PI0.5 需要的 16 维 ``observation.state``。"""
        prepared = {}
        for key, value in model_obs.items():
            if key == "observation.language":
                continue
            latest = self._latest_frame(value)
            if key.startswith("observation.images."):
                prepared[key] = self._to_torch_tensor(latest)
            else:
                prepared[key] = self._to_torch_tensor(np.asarray(latest, dtype=np.float32))
        prepared["observation.state"] = np.concatenate(
            (
                self._latest_or_zeros(model_obs, "observation.state.left_arm", 7),
                self._latest_or_zeros(model_obs, "observation.state.right_arm", 7),
                self._latest_or_zeros(model_obs, "observation.state.left_gripper", 1),
                self._latest_or_zeros(model_obs, "observation.state.right_gripper", 1),
            ),
            axis=0,
        ).astype(np.float32)
        prepared["observation.state"] = self._to_torch_tensor(prepared["observation.state"])
        task = model_obs.get("observation.language") or self._default_task
        return prepared, task

    def _split_model_action_vector(self, action):
        """把 PI0.5 16 维动作向量或动作 chunk 拆回 split action keys。"""
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()
        array = np.asarray(action, dtype=np.float32)
        if array.ndim == 1:
            if array.shape[0] < 16:
                raise ValueError(f"expected at least 16 action values, got shape {array.shape}")
            return {
                "action.left_arm": array[0:7],
                "action.right_arm": array[7:14],
                "action.left_gripper": array[14:15],
                "action.right_gripper": array[15:16],
                "action.pivot": np.zeros((7,), dtype=np.float32),
            }
        if array.ndim == 2:
            if array.shape[1] < 16:
                raise ValueError(f"expected at least 16 action values per frame, got shape {array.shape}")
            return {
                "action.left_arm": array[:, 0:7],
                "action.right_arm": array[:, 7:14],
                "action.left_gripper": array[:, 14:15],
                "action.right_gripper": array[:, 15:16],
                "action.pivot": np.zeros((array.shape[0], 7), dtype=np.float32),
            }
        raise ValueError(f"unsupported model action shape {array.shape}")

    def _adapt_action(self, model_action, obs):
        """把模型输出整理成 README 要求的动作 dict。"""
        action = {
            "meta.token": self._token,
            "action.left_gripper": self._chunk_action(
                model_action.get("action.left_gripper"),
                1,
                fallback=self._latest_or_zeros(obs, "observation.state.left_gripper", 1),
            ),
            "action.right_gripper": self._chunk_action(
                model_action.get("action.right_gripper"),
                1,
                fallback=self._latest_or_zeros(obs, "observation.state.right_gripper", 1),
            ),
            "action.pivot": self._chunk_action(model_action.get("action.pivot"), 7),
        }
        if self._control_space == "joint":
            action["action.left_arm"] = self._chunk_action(
                model_action.get("action.left_arm"),
                7,
                fallback=self._latest_or_zeros(obs, "observation.state.left_arm", 7),
            )
            action["action.right_arm"] = self._chunk_action(
                model_action.get("action.right_arm"),
                7,
                fallback=self._latest_or_zeros(obs, "observation.state.right_arm", 7),
            )
        else:
            action["action.left_ee_pose_gripper_base"] = self._chunk_action(
                model_action.get("action.left_ee_pose_gripper_base"),
                6,
            )
            action["action.right_ee_pose_gripper_base"] = self._chunk_action(
                model_action.get("action.right_ee_pose_gripper_base"),
                6,
            )
        return action

    def _latest_or_zeros(self, values, key, dim):
        """取 observation chunk 最新帧；缺失时返回单帧零动作。"""
        value = values.get(key)
        if value is None:
            return np.zeros((dim,), dtype=np.float32)
        array = np.asarray(value, dtype=np.float32)
        if array.shape == (dim,):
            return array
        if array.ndim >= 2 and array.shape[-1] == dim:
            return array[-1]
        return np.zeros((dim,), dtype=np.float32)

    @staticmethod
    def _latest_frame(value):
        """取带时间维 observation 的最新帧；字符串等非数组值原样返回。"""
        array = np.asarray(value)
        if array.ndim >= 1:
            return array[-1]
        return value

    @staticmethod
    def _to_torch_tensor(value):
        """延迟导入 torch，把 numpy/Python 值转为 tensor。"""
        import torch

        return torch.as_tensor(value)

    def _chunk_action(self, value, dim, fallback=None):
        """把单帧或 chunk 动作归一化为 ``(ACTION_CHUNK_SIZE, dim)``。"""
        if value is None:
            value = np.zeros((dim,), dtype=np.float32) if fallback is None else fallback
        array = np.asarray(value, dtype=np.float32)
        if array.shape == (dim,):
            return np.repeat(array[None, :], self.ACTION_CHUNK_SIZE, axis=0)
        if array.ndim != 2 or array.shape[1] != dim:
            raise ValueError(f"action shape {array.shape} cannot be adapted to (*, {dim})")
        if array.shape[0] == self.ACTION_CHUNK_SIZE:
            return array
        if array.shape[0] > self.ACTION_CHUNK_SIZE:
            return array[: self.ACTION_CHUNK_SIZE]
        pad_count = self.ACTION_CHUNK_SIZE - array.shape[0]
        pad = np.repeat(array[-1:, :], pad_count, axis=0)
        return np.concatenate([array, pad], axis=0)

    def reset(self):
        """清空每个 episode 级别的状态。"""
        self._step = 0
        return {"ok": True}
