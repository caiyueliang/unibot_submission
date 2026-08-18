import os
import unittest
from unittest.mock import patch

from example import run_server


class RunServerConfigTest(unittest.TestCase):
    def test_reads_port_from_environment(self):
        with patch.dict(os.environ, {"UNIBOT_SERVER_PORT": "8999"}):
            self.assertEqual(run_server.get_server_port(), 8999)

    def test_uses_default_port_when_environment_is_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(run_server.get_server_port(), 8765)

    def test_rejects_non_integer_environment_port(self):
        with patch.dict(os.environ, {"UNIBOT_SERVER_PORT": "not-a-port"}):
            with self.assertRaisesRegex(ValueError, "UNIBOT_SERVER_PORT"):
                run_server.get_server_port()


if __name__ == "__main__":
    unittest.main()
