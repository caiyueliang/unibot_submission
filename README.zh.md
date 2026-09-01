# UniBot-V1 挑战赛 — 提交模板

> English version: [README.md](README.md)

本文档规定提交所必须实现的策略接口、跨接口传递的数据格式，并附带一份参考实现。

---

## 1 · 策略接口

提交由一个 policy 类构成，该类需暴露下列三个成员。类在参赛者机器上实例化、由 `PolicyService` 对外提供服务；评测器以 client 身份连接并通过该接口进行调用。

| 成员 | 类别 | 说明 |
|---|---|---|
| `metadata`        | 属性 → `dict` | 策略配置：控制空间、观测 key 与时序叠帧、图像 resize、动作 chunk 长度、提交 token。 |
| `get_action(obs)` | 方法           | 在给定观测下产出一个动作 chunk。 |
| `reset()`         | 方法           | 清空 episode 内部状态。每个 episode 开始时调用一次。 |

### `metadata` 结构

```python
{
    "control_space":     str,             # "joint" 或 "ee"，动作空间（§3）
    "obs_delta_indices": dict[str, list], # 选中的观测 key -> 各帧偏移列表（§2）
    "image_resize":      dict[str, list], # 图像 key -> resize 到的 [高, 宽]（§2）
    "action_chunk_size": int >= 1,        # 每个动作前置时序维度的长度（§3）
    "token":             str,             # 组委会发放的提交 token（§4）
}
```

评测器按下文各字段所述的约定读取 metadata。请按每个字段的要求填写，未按要求填写可能导致交互错误。

- **`control_space`** —— policy 所处的动作空间，取 `"joint"` 或 `"ee"`，二者**互斥**；它决定 `get_action` 必须返回哪一组双臂动作 key（§3）。
- **`obs_delta_indices`** —— 同时承担两件事：**其 key 决定发送哪些观测**（只有声明的 key 才会被传输），**其 value 为一个帧偏移列表，声明该 key 的时序叠帧方式**（§2）。
- **`image_resize`** —— 返回的 observation 中各图像的分辨率 `[高, 宽]`（§2）。
- **`action_chunk_size`** —— `int` ≥ 1：`get_action` 返回的每个动作数组前置时序维度的长度（§3）。
- **`token`** —— 组委会发放的提交 token，握手时校验一次（§4）。

最小实现见 [`example/example_policy.py`](example/example_policy.py)。

---

## 2 · 观测格式

观测（`obs`）由评测端经网络发给策略，供其决策。`obs` 是一个 `dict`，可选 key 的范围及单帧形状、数据类型、说明如下。

### 观测目录

| Key | 单帧形状 | 数据类型 | 说明 |
|---|---|---|---|
| `observation.images.cam_left_high`   | `[480, 640, 3]` | `numpy.uint8`   | 左上方相机，HWC，RGB |
| `observation.images.cam_right_high`  | `[480, 640, 3]` | `numpy.uint8`   | 右上方相机，HWC，RGB |
| `observation.images.cam_left_wrist`  | `[480, 640, 3]` | `numpy.uint8`   | 左手腕相机，HWC，RGB |
| `observation.images.cam_right_wrist` | `[480, 640, 3]` | `numpy.uint8`   | 右手腕相机，HWC，RGB |
| `observation.state.left_arm`         | `[7]`           | `numpy.float32` | 左臂关节状态 |
| `observation.state.right_arm`        | `[7]`           | `numpy.float32` | 右臂关节状态 |
| `observation.state.left_ee_pose_gripper_base`  | `[6]` | `numpy.float32` | 左夹爪末端在 base 坐标系下的位姿：`xyz(3)` + `rpy(3)` |
| `observation.state.right_ee_pose_gripper_base` | `[6]` | `numpy.float32` | 右夹爪末端在 base 坐标系下的位姿：`xyz(3)` + `rpy(3)` |
| `observation.state.left_gripper`     | `[1]`           | `numpy.float32` | 左夹爪开度 |
| `observation.state.right_gripper`    | `[1]`           | `numpy.float32` | 右夹爪开度 |
| `observation.state.lower_body`       | `[15]`          | `numpy.float32` | 下身本体感知 |
| `observation.language`               | 标量            | `str`           | 当前 episode 的自然语言任务指令 |

**注意：** `observation.language` 为单个 `str`，描述当前 episode 的任务（如 `"move the block to the target position."`），全程不变、不带时序维度；下文关于时序叠帧与形状的约定均不适用于 `observation.language`。

与 `obs` 相关的 metadata 字段：

- **`obs_delta_indices`**：决定下发哪些 key，以及各 key 如何时序叠帧；
- **`image_resize`**：决定各图像的分辨率。

### `obs_delta_indices` — 选取观测与时序叠帧

`obs_delta_indices` 同时决定两件事：

1. **选哪些观测**：只传输其中声明的 key，请只声明模型需要的最小子集；
2. **如何叠帧**：每个 value 是相对当前步的**帧偏移**列表。

偏移含义：`0` 表示当前步，负数表示过去若干步；列表按时间从旧到新排列，长度即叠帧数。例如：

```python
"observation.state.left_arm": [-10, -5, -2, 0]  # t-10、t-5、t-2、t，共 4 帧
```

基于此配置，返回的 `obs` 中该数组形状为 `(4, 7)`，即 `(叠帧数, …单帧形状…)`；即使不需要叠帧，也请保留长度为 `1` 的时序维度列表。偏移须为 **≤ 0** 的 `int`，且索引列表严格递增；历史帧不足时，缺的帧使用第一帧补齐。`observation.language` 不带时序维度，其偏移列表须恰为 `[0]`。

### `image_resize` — 图像分辨率

图像原生分辨率为 `[480, 640]`（高、宽），与开源数据保持一致。`image_resize` 可为需要缩放的图像 key 指定目标 `[高, 宽]`，评测端发送前按此缩放，策略直接收到目标分辨率。例如：

```python
{"observation.images.cam_left_high": [240, 320]}
```

value 为 `[高, 宽]`（正整数）；评测端调用 `cv2.resize(frame, (宽, 高))` 缩放，缩放后单帧为 `[高, 宽, 3]`。key 须已出现在 `obs_delta_indices` 中。未列出或 `image_resize` 为空 `{}` 时，保持原生分辨率。

---

## 3 · 动作格式

动作由策略经网络返回给评测端，供其执行。`get_action` 的返回值是一个 `dict`，所有 key 的范围及单帧形状、数据类型、说明如下。

### 动作目录

| Key | 单帧形状 | 数据类型 | 控制模式 | 说明 |
|---|---|---|---|---|
| `meta.token`           | 标量 | `str`           | 共用 | 组委会为该提交发放的 token（§4） |
| `action.left_gripper`  | `[1]`  | `numpy.float32` | 共用 | 左夹爪目标指令 |
| `action.right_gripper` | `[1]`  | `numpy.float32` | 共用 | 右夹爪目标指令 |
| `action.pivot`         | `[7]`  | `numpy.float32` | 共用 | 下身 / pivot 目标 |
| `action.left_arm`  | `[7]` | `numpy.float32` | `joint` | 左臂关节目标位置 |
| `action.right_arm` | `[7]` | `numpy.float32` | `joint` | 右臂关节目标位置 |
| `action.left_ee_pose_gripper_base`  | `[6]` | `numpy.float32` | `ee` | 左夹爪末端在 base 下的目标位姿：`xyz(3)` + `rpy(3)` |
| `action.right_ee_pose_gripper_base` | `[6]` | `numpy.float32` | `ee` | 右夹爪末端在 base 下的目标位姿：`xyz(3)` + `rpy(3)` |

**注意：** `meta.token` 为单个 `str`（§4），不带时序维度；下文关于动作步数与形状的约定均不适用于 `meta.token`。

与动作相关的 metadata 字段：

- **`control_space`**：控制模式；
- **`action_chunk_size`**：每次 `get_action` 返回的动作步数。

### `control_space` — 控制模式

取 `"joint"` 或 `"ee"`，二者**互斥**。两种模式均须返回表中「共用」的 key，并按所选模式额外返回对应双臂 key：

- **`"joint"`**：`action.left_arm`、`action.right_arm`；
- **`"ee"`**：`action.left_ee_pose_gripper_base`、`action.right_ee_pose_gripper_base`。

### `action_chunk_size` — 动作步数

`action_chunk_size`（记为 `T_a`，`int` ≥ 1）为每次返回的动作步数。各动作数组形状为 `(T_a, …单帧形状…)`，索引 `0` 最先执行，随后依次执行，直至评测端再次调用 `get_action`。即使只返回一步，也请保留长度为 `1` 的时序维度。

---

## 4 · 提交 token

每个通过审核的提交由组委会发放唯一的 token。policy 需在**两处**携带该 token：

1. `metadata` 的 `token` 字段，握手时校验一次；
2. `get_action` 每次返回值中的保留 key `meta.token`，每步校验。

评测器将拒绝 token 缺失、或其值与所发放 token 不一致的握手或动作。

参考实现从环境变量 `UNIBOT_SUBMISSION_TOKEN` 读取 token：

```python
import os
import numpy as np

class MyPolicy:
    def __init__(self):
        self._token = os.environ["UNIBOT_SUBMISSION_TOKEN"]

    @property
    def metadata(self):
        return {
            "control_space":     "joint",   # 或 "ee"
            "obs_delta_indices": {               # 观测 key -> 各帧偏移列表（§2）
                "observation.language":       [0],
                "observation.images.cam_left_high": [0],
                "observation.state.left_arm": [-4, -2, 0],
                # ... 模型消费的其余观测 key
            },
            "image_resize": {                # 图像 key -> [高, 宽]（§2），可为空
                "observation.images.cam_left_high": [240, 320],
            },
            "action_chunk_size": 8,
            "token":             self._token,
        }

    def get_action(self, obs):
        T = 8
        return {
            "meta.token":      self._token,
            "action.left_arm": np.zeros((T, 7), dtype=np.float32),
            # ... 所声明 control_space 对应的其余 action.* key,形状同 §3
        }
```

---

## 5 · 参考实现与提交方式

`example/` 提供一份可运行的参考实现，仅作为实现示例。参赛者只需参照 `run_server.py` 的方式，将自有策略对象封装入 `PolicyService` 并监听指定端口对外提供服务，评测器即可连入。

| 文件 | 作用 |
|---|---|
| [`example/example_policy.py`](example/example_policy.py) | 一份最小可运行的策略，暴露 §1 接口，当前返回全零动作 chunk。参赛者可在此基础上替换为自有模型，亦可依据相同接口自行实现。 |
| [`example/example_env.py`](example/example_env.py)       | 本地评测桩，按 §2 规格产出观测，并对返回的动作执行与生产评测器一致的校验。 |
| [`example/run_server.py`](example/run_server.py)         | **服务端入口范例**，任何提交均按此方式将策略对象交由 `PolicyService` 托管并监听端口。`UNIBOT_SUBMISSION_TOKEN` 与 `UNIBOT_CONTROL_SPACE`（`joint`/`ee`）均须设置。 |
| [`example/run_client.py`](example/run_client.py)         | 本地端到端验证工具，模拟评测器连入参赛者的 server 并逐步校验返回的动作。生产环境下由组委会运行，`UNIBOT_SUBMISSION_TOKEN` 须与 server 的 token 一致。 |
| [`policy/`](policy)                                       | `PolicyService` / `RemotePolicy` 与 msgpack-NumPy 适配。 |
| [`requirements.txt`](requirements.txt)                   | 参考实现的依赖：`numpy`、`msgpack`、`websockets`、`Pillow`。 |

### 环境准备

在用于本次提交的 Python 环境中安装依赖：

```bash
pip install -r requirements.txt   # numpy、msgpack、websockets、Pillow
```

### Docker 镜像

仓库根目录提供 `Dockerfile`，镜像默认启动推理服务，不包含模型权重和数据集。构建：

```bash
docker build -t twr.wair.ac.cn/taichu-studio/unibot_submission:1.0.0 .
```

默认构建只安装服务协议和本地 fallback policy 所需依赖，保证镜像可以直接启动服务。
如果需要在镜像内提前安装 PI0.5/LeRobot 推理依赖，可使用：

```bash
docker build \
  --build-arg INSTALL_INFERENCE_DEPS=true \
  -t twr.wair.ac.cn/taichu-studio/unibot_submission:1.0.0 .
```

运行时挂载模型、数据集和 LeRobot 源码，并通过环境变量指定路径：

```bash
docker run --rm --gpus all \
  -p 8765:8765 \
  -e UNIBOT_SUBMISSION_TOKEN="123456" \
  -e UNIBOT_POLICY_PATH=/models/pi05 \
  -e UNIBOT_REPO_ID=/datasets/G1_Dex1_ArrangeTestTubes_3cams \
  -e UNIBOT_CONTROL_SPACE=joint \
  -e UNIBOT_SERVER_PORT=8765 \
  -v /data/ckpt/pi05_all/042000/pretrained_model:/models/pi05:ro \
  -v /data/datasets/G1_Dex1_ArrangeTestTubes_3cams:/datasets/G1_Dex1_ArrangeTestTubes_3cams:ro \
  -v /home/wair/caiyueliang/lerobot:/opt/lerobot:ro \
  -v /home/wair/caiyueliang/unitree_lerobot:/opt/unitree_lerobot:ro \
  -v /home/wair/yangsheng/xr_teleoperate/teleop/teleimager:/opt/teleimager:ro \
  -v unibot_hf_cache:/cache/huggingface \
  twr.wair.ac.cn/taichu-studio/unibot_submission:1.0.0
```

同一个镜像也包含本地验证 client：

```bash
docker run --rm --network host \
  -e UNIBOT_SUBMISSION_TOKEN="123456" \
  twr.wair.ac.cn/taichu-studio/unibot_submission:1.0.0 \
  python client.py ws://127.0.0.1:8765
```

### 工作流程

本地验证统一通过 `run_server.py` + `run_client.py` 进行：一个终端启动 server，另一个终端以 client 模拟评测器连入。两端的 token 须保持一致。

1. **先运行参考实现。** 使用未经改动的 `ExamplePolicy` 完整运行一次，确认端到端输出正常（client 每步打印 `token verified`，且无 `ACTION REJECTED`），以了解完整的评测流程：
   ```bash
   # 终端 A
   UNIBOT_SUBMISSION_TOKEN=dev-token UNIBOT_CONTROL_SPACE=joint UNIBOT_SERVER_PORT=8765 python example/run_server.py

   # 终端 B
   UNIBOT_SUBMISSION_TOKEN=dev-token python example/run_client.py
   ```
2. **接入自有模型。** 按 §1 接口实现策略类，替换 `run_server.py` 中实例化的 `ExamplePolicy`；将 `get_action` 中的零向量替换为模型输出，并按需调整 `control_space`、`obs_delta_indices`、`image_resize`、`action_chunk_size`。
3. **再次运行以确认接入无误。** 重复第 1 步的两个终端命令，确认接入自有模型后 client 仍逐步通过校验，即表明接口对接正确。
4. **正式部署。** 将组委会下发的 token 写入 `UNIBOT_SUBMISSION_TOKEN`，按 `run_server.py` 的方式启动 server 并保持在线，评测器将主动连入。

   评测在 **N 台测试机器人**上以随机顺序进行（N 的具体数量由组委会另行通知），评测端为每台机器人固定连接一个端口进行交互。因此提交必须对外暴露 **N 个连续端口**，从 `8765` 起（`8765`、`8766`、…、`8765 + N − 1`），**端口数量须与机器人数量 N 严格一致**，且每个端口都要能独立、稳定地完成其对应机器人的整条交互流程。至于每个端口背后如何实现，由参赛者自行选择，下面给出两种常见做法：

   **方式一 · 每端口独立 server（最稳定）。** N 个端口各跑一个进程、各自加载一份模型，互不影响、实现最简单；代价是需要 N 份模型的显存 / 内存，模型较重时成本高。以 **N = 3** 为例，开放 `8765`–`8767`，各启动一个实例：

   ```bash
   UNIBOT_SUBMISSION_TOKEN=<token> UNIBOT_CONTROL_SPACE=<joint|ee> UNIBOT_SERVER_PORT=8765 python example/run_server.py
   UNIBOT_SUBMISSION_TOKEN=<token> UNIBOT_CONTROL_SPACE=<joint|ee> UNIBOT_SERVER_PORT=8766 python example/run_server.py
   UNIBOT_SUBMISSION_TOKEN=<token> UNIBOT_CONTROL_SPACE=<joint|ee> UNIBOT_SERVER_PORT=8767 python example/run_server.py
   ```

   `UNIBOT_SERVER_PORT` 即该实例监听的端口。这些实例可同机部署，也可分布于多台机器，但须保证全部 N 个端口均可从外部访问。

   **方式二 · 多端口共用模型（省资源）。** 模型较重、显存 / 内存装不下 N 份时，可让多个端口共用同一份模型，但需参赛者自行在 server 端实现请求分发（N 个轻量端点各监听一个端口，再把请求转发给共享的推理后端）。此时须注意两点：

   - **并发：** 不同端口可能被同时调用，共享后端需能并发处理，避免相互阻塞、拉高时延；
   - **状态隔离：** 每个端口对应一台独立机器人的独立 episode，各端口的 `reset` 与历史状态必须相互隔离，不能串用。

---

## 6 · 交互质量与成绩可信度

评测全程通过网络对策略逐步发起调用，每次 `get_action` 响应的时延与内容均会被记录。提交方应保持交互时延平滑、稳定，并使返回结果尽量避免不必要的异常值。

组委会在评估成绩可信度时，可能综合评测过程中采集的各项交互数据，包括每步响应时延及其波动，以及返回动作的统计规律性，且上述指标可能与公布的成绩一同展示。请参赛者据此保持每步推理时延稳定，确保返回动作格式规范、无异常离群值，并避免任何导致交互模式不规律的行为。
