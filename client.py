"""Convenience entrypoint for the bundled validation client."""

import sys

from example.run_client import main


if __name__ == "__main__":
    uri = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8765"
    main(uri=uri)
