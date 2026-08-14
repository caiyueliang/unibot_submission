"""评测端示例脚本。

它会连接到一个已经启动的 policy 服务端，读取服务端 metadata，创建
``ExampleEnv``，然后循环执行 ``get_action -> env.step``。``ExampleEnv`` 会校验
每次返回的动作是否满足 shape / dtype / token 等提交规范。
注意：本地环境变量 ``UNIBOT_SUBMISSION_TOKEN`` 必须与服务端一致。

    UNIBOT_SUBMISSION_TOKEN=<token> python example/run_client.py [ws://host:port]
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# 把仓库根目录加入 import path，确保可以导入 policy 包。
sys.path.insert(0, os.path.dirname(HERE))
# 把当前 example 目录加入 import path，确保可以导入同目录下的 example_env。
sys.path.insert(0, HERE)

from example_env import ActionError, ExampleEnv
from policy.web_policy import RemotePolicy


def main(uri: str = "ws://127.0.0.1:8765", n_steps: int = 5) -> None:
    """连接服务端，并执行 n_steps 次 ``get_action -> env.step`` 校验循环。"""
    print(f"Connecting to {uri} ...")
    client = RemotePolicy(host=uri)
    # metadata 是服务端握手时发来的接口声明，后续环境校验会完全依赖它。
    meta = client.metadata

    env = ExampleEnv(meta)
    print("[server metadata]")
    print(f"  control_space     = {meta['control_space']}")
    print(f"  data_keys         = {meta['data_keys']}")
    print(f"  obs_chunk_size    = {meta['obs_chunk_size']}")
    print(f"  action_chunk_size = {meta['action_chunk_size']}")
    print()

    # 先通知服务端重置内部状态，再让本地假环境生成第一段 observation。
    client.reset()
    print(f"client.reset() -> server reset done")
    obs = env.reset()
    print(f"env.reset() -> first observation: {obs}")
    for i in range(n_steps):
        # 发送 observation 到远端 policy，拿回一段 action chunk。

        # print(f"[{i}/{n_steps}] obs: {obs}")
        action = client.get_action(obs)
        # print(f"[{i}/{n_steps}] action: {action}; obs: {obs}")
        try:
            # env.step 会严格检查 action 的 key、shape、dtype 和 token。
            obs = env.step(action)
        except ActionError as e:
            print(f"[step {i}] ACTION REJECTED: {e}")
            sys.exit(1)
        # 打印动作 key 和 shape，方便本地确认当前控制空间输出是否符合预期。
        action_keys = [k for k in action if not k.startswith("meta.")]
        shapes = ", ".join(f"{k}={tuple(action[k].shape)}" for k in action_keys)
        
        print(f"[step {i}] OK — token verified, {len(action_keys)} action keys: {shapes}")

    print("\nAll steps passed validation.")


if __name__ == "__main__":
    # 命令行第一个参数可覆盖默认连接地址；没有传参时连接本机 8765 端口。
    uri = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8765"
    main(uri=uri)
