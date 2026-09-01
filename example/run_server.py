"""启动 WebSocket 服务端，把 ``ExamplePolicy`` 暴露给评测端连接。

运行后服务端会监听指定端口，客户端（例如 ``run_client.py`` 或正式评测器）
可以通过 ``ws://host:port`` 连接并调用 policy。

    UNIBOT_SUBMISSION_TOKEN=<token> UNIBOT_CONTROL_SPACE=joint|ee UNIBOT_SERVER_PORT=8765 python example/run_server.py
"""

import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# 把仓库根目录加入 import path，确保可以导入 policy 包。
sys.path.insert(0, os.path.dirname(HERE))
# 把当前 example 目录加入 import path，确保可以导入同目录下的 example_policy。
sys.path.insert(0, HERE)

from example_policy import ExamplePolicy
from policy.web_policy import PolicyService

DEFAULT_PORT = 8765
PORT_ENV_VAR = "UNIBOT_SERVER_PORT"


def get_server_port() -> int:
    """从环境变量读取服务端监听端口；未设置时使用默认端口。"""
    value = os.environ.get(PORT_ENV_VAR, str(DEFAULT_PORT))
    try:
        port = int(value)
    except ValueError as exc:
        raise ValueError(f"{PORT_ENV_VAR} must be an integer, got {value!r}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{PORT_ENV_VAR} must be between 1 and 65535, got {port}")
    return port


def main(host: str = "0.0.0.0", port: int = DEFAULT_PORT) -> None:
    """创建 policy 实例，并在 host:port 上持续提供服务。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [server] %(message)s")
    policy = ExamplePolicy()
    # 打印 metadata 方便本地确认配置，但不要把 token 打到日志里。
    safe_meta = {k: v for k, v in policy.metadata.items() if k != "token"}
    print(f"Serving ExamplePolicy on ws://{host}:{port}")
    print("  metadata =", safe_meta)
    PolicyService(policy, host=host, port=port).run_forever()


if __name__ == "__main__":
    main(port=get_server_port())
