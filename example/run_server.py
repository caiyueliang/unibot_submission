"""启动 WebSocket 服务端，把 ``ExamplePolicy`` 暴露给评测端连接。

运行后服务端会监听指定端口，客户端（例如 ``run_client.py`` 或正式评测器）
可以通过 ``ws://host:port`` 连接并调用 policy。

    UNIBOT_SUBMISSION_TOKEN=<token> UNIBOT_CONTROL_SPACE=joint|ee python example/run_server.py [port]
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


def main(host: str = "0.0.0.0", port: int = 8765) -> None:
    """创建 policy 实例，并在 host:port 上持续提供服务。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [server] %(message)s")
    policy = ExamplePolicy()
    # 打印 metadata 方便本地确认配置，但不要把 token 打到日志里。
    safe_meta = {k: v for k, v in policy.metadata.items() if k != "token"}
    print(f"Serving ExamplePolicy on ws://{host}:{port}")
    print("  metadata =", safe_meta)
    PolicyService(policy, host=host, port=port).run_forever()


if __name__ == "__main__":
    # 命令行第一个参数可覆盖默认端口；没有传参时使用 8765。
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    main(port=port)
