# UniBot-V1 挑战赛 — 提交模板

> English version: [README.md](README.md)

本文档规定提交所必须实现的策略接口、跨接口传递的数据格式，并附带一份参考实现。

---

## 1 · 策略接口

提交由一个 policy 类构成，该类需暴露下列三个成员。类在参赛者机器上实例化、由 `PolicyService` 对外提供服务；评测器以 client 身份连接并通过该接口进行调用。

| 成员 | 类别 | 说明 |
|---|---|---|
| `metadata`        | 属性 → `dict` | 声明控制空间、观测 key、时序 chunk 长度与提交 token。握手阶段发送一次。 |
| `get_action(obs)` | 方法           | 在给定观测下产出一个动作 chunk。评测过程中每步调用一次。 |
| `reset()`         | 方法           | 清空 episode 内部状态。每个 episode 开始时调用一次。 |

### `metadata` 结构

```python
{
    "control_space":     str,         # "joint" 或 "ee"，动作空间（§3）
    "data_keys":         list[str],   # §2 观测目录的子集
    "obs_chunk_size":    int >= 1,    # 每个观测张量前置时序维度的长度
    "action_chunk_size": int >= 1,    # 每个动作张量前置时序维度的长度
    "token":             str,         # 组委会发放的提交 token（§5）
}
```

以上五个字段**均为必填**。评测器按原样读取，且**不会**为任何缺失字段填入默认值，metadata 缺失或格式不符任一字段的提交，将在连接时被直接拒绝。

`control_space` 指定 policy 所处的动作空间（§3），在 `"joint"` 与 `"ee"` 之间**互斥**。

最小实现见 [`example/example_policy.py`](example/example_policy.py)。

---

## 2 · 观测格式

`obs` 是一个 `dict`，其 key 取自下表所列目录，value 为 `numpy.ndarray`。每个 value 在表中所列的单帧形状之前再附加一个长度为 `obs_chunk_size` 的前置时序维度（详见 §4）。

| Key | 单帧形状 | dtype | 说明 |
|---|---|---|---|
| `observation.images.cam_left_high`   | `[480, 640, 3]` | `uint8`   | 左上方相机，HWC，RGB |
| `observation.images.cam_right_high`  | `[480, 640, 3]` | `uint8`   | 右上方相机，HWC，RGB |
| `observation.images.cam_left_wrist`  | `[480, 640, 3]` | `uint8`   | 左手腕相机，HWC，RGB |
| `observation.images.cam_right_wrist` | `[480, 640, 3]` | `uint8`   | 右手腕相机，HWC，RGB |
| `observation.state.left_arm`         | `[7]`           | `float32` | 左臂关节状态 |
| `observation.state.right_arm`        | `[7]`           | `float32` | 右臂关节状态 |
| `observation.state.left_ee_pose_gripper_base`  | `[6]` | `float32` | 左夹爪末端在 base 坐标系下的位姿：`xyz(3)` + `rpy(3)` |
| `observation.state.right_ee_pose_gripper_base` | `[6]` | `float32` | 右夹爪末端在 base 坐标系下的位姿：`xyz(3)` + `rpy(3)` |
| `observation.state.left_gripper`     | `[1]`           | `float32` | 左夹爪开度 |
| `observation.state.right_gripper`    | `[1]`           | `float32` | 右夹爪开度 |
| `observation.state.lower_body`       | `[15]`          | `float32` | 下身本体感知 |
| `observation.language`               | 标量            | `str`     | 当前 episode 的自然语言任务指令 |

仅 `metadata.data_keys` 中声明的 key 会传递给 policy。请仅声明模型实际需要的最小子集。

关节臂状态（`observation.state.left_arm` / `right_arm`）与末端位姿（`observation.state.*_ee_pose_gripper_base`）**无论 `control_space` 为何都在目录中提供**。末端位姿中，`xyz` 为位置（米），`rpy` 为 roll-pitch-yaw 欧拉角（弧度）表示的姿态；其布局与 `action.*_ee_pose_gripper_base` 动作（§3）一致。

`observation.language` 是上述张量布局的例外：其 value 为单个 Python `str`，描述需要执行的任务（例如 `"move the block to the target position."`）。该指令在**整个 episode 内保持不变**，且**不带前置时序维度**（见 §4）。

---

## 3 · 动作格式

`get_action` 返回一个 `dict`，其中需包含提交 token（§5）以及策略在 `metadata.control_space` 中声明的**控制空间**所对应的动作数组。policy 只处于其中一个控制空间；`"joint"` 与 `"ee"` **互斥**。每个动作 value 在表中所列的单帧形状之前再附加一个长度为 `action_chunk_size` 的前置时序维度（详见 §4）。

动作 key 分为两类：一类是 `joint` 与 `ee` 下都必须提供、且键名 / 形状 / dtype 完全一致的**公共 key**（`meta.token` 与夹爪、pivot）；另一类是随 `control_space` 二选一的**双臂 key**（`joint` 用关节目标，`ee` 用末端位姿）。下面三张表分别列出。

**公共 key，两种控制空间下都存在**

| Key | 单帧形状 | dtype | 说明 |
|---|---|---|---|
| `meta.token`           | 标量 | `str`     | 组委会为该提交发放的 token（§5） |
| `action.left_gripper`  | `[1]`  | `float32` | 左夹爪指令 |
| `action.right_gripper` | `[1]`  | `float32` | 右夹爪指令 |
| `action.pivot`         | `[7]`  | `float32` | 下身 / pivot 目标 |

**`control_space = "joint"`**，双臂动作为目标关节位置：

| Key | 单帧形状 | dtype | 说明 |
|---|---|---|---|
| `action.left_arm`  | `[7]` | `float32` | 左臂关节目标位置 |
| `action.right_arm` | `[7]` | `float32` | 右臂关节目标位置 |

**`control_space = "ee"`**，双臂动作为目标末端位姿：

| Key | 单帧形状 | dtype | 说明 |
|---|---|---|---|
| `action.left_ee_pose_gripper_base`  | `[6]` | `float32` | 左夹爪末端在 base 坐标系下的目标位姿：`xyz(3)` + `rpy(3)` |
| `action.right_ee_pose_gripper_base` | `[6]` | `float32` | 右夹爪末端在 base 坐标系下的目标位姿：`xyz(3)` + `rpy(3)` |

末端位姿中，`xyz` 为位置（米），`rpy` 为 roll-pitch-yaw 欧拉角（弧度）表示的姿态，与 `observation.state.*_ee_pose_gripper_base` 观测（§2）的布局一致。

---

## 4 · 前置时序维度

所有观测张量与动作张量均带有一个前置时序维度，其长度由 `metadata` 中的对应字段决定。

```text
# obs_chunk_size = T_o
obs["observation.images.cam_left_high"].shape == (T_o, 480, 640, 3)
obs["observation.state.left_arm"].shape       == (T_o, 7)
obs["observation.state.lower_body"].shape     == (T_o, 15)

# action_chunk_size = T_a，公共 key（joint 与 ee 都有）
action["action.left_gripper"].shape  == (T_a, 1)
action["action.right_gripper"].shape == (T_a, 1)
action["action.pivot"].shape         == (T_a, 7)

# control_space == "joint"：双臂为关节目标
action["action.left_arm"].shape      == (T_a, 7)
action["action.right_arm"].shape     == (T_a, 7)

# control_space == "ee"：双臂为末端位姿
action["action.left_ee_pose_gripper_base"].shape  == (T_a, 6)
action["action.right_ee_pose_gripper_base"].shape == (T_a, 6)
```

该维度按时间顺序排列。对观测而言，索引 `0` 为最旧帧，索引 `T_o − 1` 为最新帧；对动作而言，索引 `0` 为下一个执行的动作，评测器顺序执行各索引直至再次调用 `get_action`。当对应 chunk size 为 `1` 时，前置维度仍然存在，不会被省略。

`observation.language` 不受此规则约束：它是单个字符串而非张量，传递时不带前置时序维度。

---

## 5 · 提交 token

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
            "data_keys":         [...],      # §2 目录的子集
            "obs_chunk_size":    2,
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

## 6 · 参考实现与提交方式

`example/` 提供一份可运行的参考实现，仅作为实现示例。参赛者只需参照 `run_server.py` 的方式，将自有策略对象封装入 `PolicyService` 并监听指定端口对外提供服务，评测器即可连入。

| 文件 | 作用 |
|---|---|
| [`example/example_policy.py`](example/example_policy.py) | 一份最小可运行的策略，暴露 §1 接口，当前返回全零动作 chunk。参赛者可在此基础上替换为自有模型，亦可依据相同接口自行实现。 |
| [`example/example_env.py`](example/example_env.py)       | 本地评测桩，按 §2 规格产出观测，并对返回的动作执行与生产评测器一致的校验。 |
| [`example/run_server.py`](example/run_server.py)         | **服务端入口范例**，任何提交均按此方式将策略对象交由 `PolicyService` 托管并监听端口。`UNIBOT_SUBMISSION_TOKEN` 与 `UNIBOT_CONTROL_SPACE`（`joint`/`ee`）均须设置。 |
| [`example/run_client.py`](example/run_client.py)         | 本地端到端验证工具，模拟评测器连入参赛者的 server 并逐步校验返回的动作。生产环境下由组委会运行，`UNIBOT_SUBMISSION_TOKEN` 须与 server 的 token 一致。 |
| [`policy/`](policy)                                       | `PolicyService` / `RemotePolicy` 与 msgpack-NumPy 适配。 |
| [`requirements.txt`](requirements.txt)                   | 参考实现的依赖：`numpy`、`msgpack`、`websockets`。 |

### 环境准备

在用于本次提交的 Python 环境中安装依赖：

```bash
pip install -r requirements.txt   # numpy、msgpack、websockets
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
2. **接入自有模型。** 按 §1 接口实现策略类，替换 `run_server.py` 中实例化的 `ExamplePolicy`；将 `get_action` 中的零向量替换为模型输出，并按需调整 `control_space`、`data_keys`、`obs_chunk_size`、`action_chunk_size`。
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

## 7 · 交互质量与成绩可信度

评测全程通过网络对策略逐步发起调用，每次 `get_action` 响应的时延与内容均会被记录。提交方应保持交互时延平滑、稳定，并使返回结果尽量避免不必要的异常值。

组委会在评估成绩可信度时，可能综合评测过程中采集的各项交互数据，包括每步响应时延及其波动，以及返回动作的统计规律性，且上述指标可能与公布的成绩一同展示。请参赛者据此保持每步推理时延稳定，确保返回动作格式规范、无异常离群值，并避免任何导致交互模式不规律的行为。
