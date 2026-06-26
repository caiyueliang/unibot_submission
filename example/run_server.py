"""Serve ExamplePolicy over websocket for the evaluator to connect to.

    UNIBOT_SUBMISSION_TOKEN=<token> UNIBOT_CONTROL_SPACE=joint|ee python example/run_server.py [port]
"""

import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # repo root, for the `policy` package
sys.path.insert(0, HERE)                   # this directory, for sibling modules

from example_policy import ExamplePolicy
from policy.web_policy import PolicyService


def main(host: str = "0.0.0.0", port: int = 8765) -> None:
    """Build the policy and serve it on host:port."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [server] %(message)s")
    policy = ExamplePolicy()
    safe_meta = {k: v for k, v in policy.metadata.items() if k != "token"}
    print(f"Serving ExamplePolicy on ws://{host}:{port}")
    print("  metadata =", safe_meta)
    PolicyService(policy, host=host, port=port).run_forever()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    main(port=port)
