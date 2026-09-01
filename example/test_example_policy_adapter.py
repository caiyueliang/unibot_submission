import os
import unittest

import numpy as np

from example.example_policy import ExamplePolicy


class ExamplePolicyAdapterTest(unittest.TestCase):
    def setUp(self):
        os.environ["UNIBOT_SUBMISSION_TOKEN"] = "dev-token"
        os.environ["UNIBOT_CONTROL_SPACE"] = "joint"
        os.environ.pop("UNIBOT_POLICY_PATH", None)
        os.environ.pop("UNIBOT_REPO_ID", None)
        os.environ.pop("UNIBOT_DEFAULT_TASK", None)
        self.policy = ExamplePolicy()
        T = self.policy.OBS_CHUNK_SIZE
        self.obs = {
            "observation.language": "arrange the test tubes.",
            "observation.images.cam_left_high": np.ones((T, 480, 640, 3), dtype=np.uint8),
            "observation.images.cam_left_wrist": np.full((T, 480, 640, 3), 2, dtype=np.uint8),
            "observation.images.cam_right_wrist": np.full((T, 480, 640, 3), 3, dtype=np.uint8),
            "observation.state.left_arm": np.ones((T, 7), dtype=np.float32),
            "observation.state.right_arm": np.full((T, 7), 2.0, dtype=np.float32),
            "observation.state.left_gripper": np.full((T, 1), 0.4, dtype=np.float32),
            "observation.state.right_gripper": np.full((T, 1), 0.6, dtype=np.float32),
        }

    def test_metadata_declares_only_pi05_runtime_observation_keys(self):
        data_keys = set(self.policy.metadata["obs_delta_indices"])

        self.assertNotIn("observation.state.left_ee_pose_gripper_base", data_keys)
        self.assertNotIn("observation.state.right_ee_pose_gripper_base", data_keys)
        self.assertNotIn("observation.state.lower_body", data_keys)
        self.assertEqual(data_keys, set(self.obs))

    def test_adapts_evaluator_observation_keys_to_dataset_keys(self):
        model_obs = self.policy._adapt_observation(self.obs)

        self.assertIs(model_obs["observation.images.head_stereo_left"], self.obs["observation.images.cam_left_high"])
        self.assertIs(model_obs["observation.images.wrist_left"], self.obs["observation.images.cam_left_wrist"])
        self.assertIs(model_obs["observation.images.wrist_right"], self.obs["observation.images.cam_right_wrist"])
        self.assertEqual(model_obs["observation.language"], "arrange the test tubes.")

    def test_get_action_returns_readme_joint_contract(self):
        action = self.policy.get_action(self.obs)

        self.assertEqual(action["meta.token"], "dev-token")
        self.assertEqual(set(action), {
            "meta.token",
            "action.left_arm",
            "action.right_arm",
            "action.left_gripper",
            "action.right_gripper",
            "action.pivot",
        })
        for key, dim in {
            "action.left_arm": 7,
            "action.right_arm": 7,
            "action.left_gripper": 1,
            "action.right_gripper": 1,
            "action.pivot": 7,
        }.items():
            self.assertEqual(action[key].shape, (self.policy.ACTION_CHUNK_SIZE, dim))
            self.assertEqual(action[key].dtype, np.float32)

        np.testing.assert_allclose(action["action.left_arm"][0], self.obs["observation.state.left_arm"][-1])
        np.testing.assert_allclose(action["action.right_arm"][0], self.obs["observation.state.right_arm"][-1])

    def test_requires_repo_id_when_policy_path_is_set(self):
        os.environ["UNIBOT_POLICY_PATH"] = "/tmp/pretrained_model"
        os.environ.pop("UNIBOT_REPO_ID", None)

        with self.assertRaisesRegex(ValueError, "UNIBOT_REPO_ID"):
            ExamplePolicy()

    def test_prepare_policy_observation_uses_request_language(self):
        model_obs = self.policy._adapt_observation(self.obs)

        prepared, task = self.policy._prepare_policy_observation(model_obs)

        self.assertEqual(task, "arrange the test tubes.")
        self.assertEqual(prepared["observation.state"].shape, (16,))
        np.testing.assert_allclose(prepared["observation.state"][:7], self.obs["observation.state.left_arm"][-1])
        np.testing.assert_allclose(prepared["observation.state"][7:14], self.obs["observation.state.right_arm"][-1])
        np.testing.assert_allclose(prepared["observation.state"][14:15], self.obs["observation.state.left_gripper"][-1])
        np.testing.assert_allclose(prepared["observation.state"][15:16], self.obs["observation.state.right_gripper"][-1])

    def test_split_model_action_vector_returns_dataset_action_keys(self):
        vector = np.arange(16, dtype=np.float32)

        action = self.policy._split_model_action_vector(vector)

        np.testing.assert_allclose(action["action.left_arm"], np.arange(0, 7, dtype=np.float32))
        np.testing.assert_allclose(action["action.right_arm"], np.arange(7, 14, dtype=np.float32))
        np.testing.assert_allclose(action["action.left_gripper"], np.array([14], dtype=np.float32))
        np.testing.assert_allclose(action["action.right_gripper"], np.array([15], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
