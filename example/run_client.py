"""Evaluator-side runner: connect to a server and step it through ExampleEnv,
which validates every action. UNIBOT_SUBMISSION_TOKEN must match the server's.

    UNIBOT_SUBMISSION_TOKEN=<token> python example/run_client.py [ws://host:port]
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # repo root, for the `policy` package
sys.path.insert(0, HERE)                   # this directory, for sibling modules

from example_env import ActionError, ExampleEnv
from policy.web_policy import RemotePolicy


def main(uri: str = "ws://127.0.0.1:8765", n_steps: int = 5) -> None:
    """Connect to the server and run n_steps of get_action → env.step."""
    print(f"Connecting to {uri} ...")
    client = RemotePolicy(host=uri)
    meta = client.metadata

    env = ExampleEnv(meta)
    print("[server metadata]")
    print(f"  control_space     = {meta['control_space']}")
    print(f"  action_chunk_size = {meta['action_chunk_size']}")
    print(f"  image_resize      = {meta['image_resize']}")
    print(f"  obs_delta_indices =")
    for key, offsets in meta["obs_delta_indices"].items():
        print(f"      {key}: {offsets}")
    print()

    client.reset()
    obs = env.reset()
    latencies_ms = []
    for i in range(n_steps):
        t0 = time.perf_counter()
        action = client.get_action(obs)
        dt_ms = (time.perf_counter() - t0) * 1e3
        latencies_ms.append(dt_ms)
        try:
            obs = env.step(action)
        except ActionError as e:
            print(f"[step {i}] ACTION REJECTED: {e}")
            sys.exit(1)
        action_keys = [k for k in action if not k.startswith("meta.")]
        shapes = ", ".join(f"{k}={tuple(action[k].shape)}" for k in action_keys)
        print(f"[step {i}] OK — token verified, {dt_ms:.1f} ms, {len(action_keys)} action keys: {shapes}")
    print("\nAll steps passed validation.")
    n = len(latencies_ms)
    if n:
        print(
            f"get_action latency over {n} steps: "
            f"mean {sum(latencies_ms) / n:.1f} ms, "
            f"min {min(latencies_ms):.1f} ms, "
            f"max {max(latencies_ms):.1f} ms"
        )


if __name__ == "__main__":
    uri = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8765"
    main(uri=uri)
