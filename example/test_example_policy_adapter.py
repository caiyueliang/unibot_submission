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
        os.environ.pop("UNIBOT_ACTION_CHUNK_SIZE", None)
        os.environ.pop("UNIBOT_ENABLE_ACTION_SAFETY", None)
        os.environ.pop("UNIBOT_MAX_JOINT_DELTA", None)
        os.environ.pop("UNIBOT_MAX_GRIPPER_DELTA", None)
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

    def test_get_action_clips_large_joint_and_gripper_deltas(self):
        os.environ["UNIBOT_ACTION_CHUNK_SIZE"] = "3"
        os.environ["UNIBOT_MAX_JOINT_DELTA"] = "0.1"
        os.environ["UNIBOT_MAX_GRIPPER_DELTA"] = "0.2"

        class JumpPolicy(ExamplePolicy):
            def _predict_model_action(self, model_obs):
                return {
                    "action.left_arm": np.array(
                        [
                            [1.0, -1.0, 0.05, 0.0, 0.0, 0.0, 0.0],
                            [1.0, -1.0, 0.25, 0.0, 0.0, 0.0, 0.0],
                            [-1.0, 1.0, 0.40, 0.0, 0.0, 0.0, 0.0],
                        ],
                        dtype=np.float32,
                    ),
                    "action.right_arm": np.array(
                        [
                            [-1.0, 1.0, -0.05, 0.0, 0.0, 0.0, 0.0],
                            [-1.0, 1.0, -0.25, 0.0, 0.0, 0.0, 0.0],
                            [1.0, -1.0, -0.40, 0.0, 0.0, 0.0, 0.0],
                        ],
                        dtype=np.float32,
                    ),
                    "action.left_gripper": np.array([[2.0], [2.0], [0.0]], dtype=np.float32),
                    "action.right_gripper": np.array([[-2.0], [-2.0], [0.0]], dtype=np.float32),
                    "action.pivot": np.ones((3, 7), dtype=np.float32),
                }

        policy = JumpPolicy()
        obs = dict(self.obs)
        obs["observation.state.left_arm"] = np.zeros((1, 7), dtype=np.float32)
        obs["observation.state.right_arm"] = np.zeros((1, 7), dtype=np.float32)
        obs["observation.state.left_gripper"] = np.zeros((1, 1), dtype=np.float32)
        obs["observation.state.right_gripper"] = np.zeros((1, 1), dtype=np.float32)

        action = policy.get_action(obs)

        np.testing.assert_allclose(
            action["action.left_arm"][:, :3],
            np.array([[0.1, -0.1, 0.05], [0.2, -0.2, 0.15], [0.1, -0.1, 0.25]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            action["action.right_arm"][:, :3],
            np.array([[-0.1, 0.1, -0.05], [-0.2, 0.2, -0.15], [-0.1, 0.1, -0.25]], dtype=np.float32),
        )
        np.testing.assert_allclose(action["action.left_gripper"], np.array([[0.2], [0.4], [0.2]], dtype=np.float32))
        np.testing.assert_allclose(action["action.right_gripper"], np.array([[-0.2], [-0.4], [-0.2]], dtype=np.float32))
        np.testing.assert_allclose(action["action.pivot"], np.ones((3, 7), dtype=np.float32))

    def test_action_safety_clip_can_be_disabled(self):
        os.environ["UNIBOT_ACTION_CHUNK_SIZE"] = "1"
        os.environ["UNIBOT_ENABLE_ACTION_SAFETY"] = "false"

        class JumpPolicy(ExamplePolicy):
            def _predict_model_action(self, model_obs):
                return {
                    "action.left_arm": np.full((1, 7), 2.0, dtype=np.float32),
                    "action.right_arm": np.full((1, 7), -2.0, dtype=np.float32),
                    "action.left_gripper": np.array([[2.0]], dtype=np.float32),
                    "action.right_gripper": np.array([[-2.0]], dtype=np.float32),
                    "action.pivot": np.zeros((1, 7), dtype=np.float32),
                }

        policy = JumpPolicy()
        obs = dict(self.obs)
        obs["observation.state.left_arm"] = np.zeros((1, 7), dtype=np.float32)
        obs["observation.state.right_arm"] = np.zeros((1, 7), dtype=np.float32)
        obs["observation.state.left_gripper"] = np.zeros((1, 1), dtype=np.float32)
        obs["observation.state.right_gripper"] = np.zeros((1, 1), dtype=np.float32)

        action = policy.get_action(obs)

        np.testing.assert_allclose(action["action.left_arm"], np.full((1, 7), 2.0, dtype=np.float32))
        np.testing.assert_allclose(action["action.right_arm"], np.full((1, 7), -2.0, dtype=np.float32))

    def test_real_sample_action_safety_clip_output(self):
        class RealSamplePolicy(ExamplePolicy):
            def _predict_model_action(self, model_obs):
                return {
                    "action.left_gripper": np.array([[4.030442714691162]], dtype=np.float32),
                    "action.right_gripper": np.array([[2.6583504676818848]], dtype=np.float32),
                    "action.pivot": np.zeros((1, 7), dtype=np.float32),
                    "action.left_arm": np.array(
                        [[
                            -0.08698493242263794,
                            0.48088252544403076,
                            -0.042104050517082214,
                            -0.4055706262588501,
                            -0.37599074840545654,
                            0.2420746237039566,
                            -0.7427308559417725,
                        ]],
                        dtype=np.float32,
                    ),
                    "action.right_arm": np.array(
                        [[
                            -1.0320369005203247,
                            -0.8419457077980042,
                            0.31322041153907776,
                            0.5841830372810364,
                            -0.21033702790737152,
                            0.5849776864051819,
                            1.2433329820632935,
                        ]],
                        dtype=np.float32,
                    ),
                }

        obs = {
            "observation.language": "Place the test tubes neatly back into the test tube rack.",
            "observation.state.left_arm": np.array(
                [[
                    0.31828904151916504,
                    0.38898399472236633,
                    0.2698008716106415,
                    0.038217693567276,
                    -0.3668491244316101,
                    -0.11712183058261871,
                    -0.698979914188385,
                ]],
                dtype=np.float32,
            ),
            "observation.state.right_arm": np.array(
                [[
                    0.3583642840385437,
                    -0.3616359829902649,
                    -0.31354328989982605,
                    0.04164518415927887,
                    0.364656001329422,
                    -0.12902216613292694,
                    0.70981365442276,
                ]],
                dtype=np.float32,
            ),
            "observation.state.left_gripper": np.array([[4.20970344543457]], dtype=np.float32),
            "observation.state.right_gripper": np.array([[4.173340320587158]], dtype=np.float32),
        }

        raw_action = RealSamplePolicy()._predict_model_action({})
        action = RealSamplePolicy().get_action(obs)

        print("\n[real sample action safety clip]")
        for action_key, obs_key in (
            ("action.left_arm", "observation.state.left_arm"),
            ("action.right_arm", "observation.state.right_arm"),
            ("action.left_gripper", "observation.state.left_gripper"),
            ("action.right_gripper", "observation.state.right_gripper"),
        ):
            raw_delta = raw_action[action_key][0] - obs[obs_key][0]
            clipped_delta = action[action_key][0] - obs[obs_key][0]
            print(f"{action_key} raw_action={raw_action[action_key]}")
            print(f"{action_key} raw_delta={raw_delta}")
            print(f"{action_key} clipped_action={action[action_key]}")
            print(f"{action_key} clipped_delta={clipped_delta}")
            print(f"{action_key} max_abs_clipped_delta={np.max(np.abs(clipped_delta)):.6f}")

        self.assertLessEqual(np.max(np.abs(action["action.left_arm"][0] - obs["observation.state.left_arm"][0])), 0.120001)
        self.assertLessEqual(np.max(np.abs(action["action.right_arm"][0] - obs["observation.state.right_arm"][0])), 0.120001)
        self.assertLessEqual(np.max(np.abs(action["action.left_gripper"][0] - obs["observation.state.left_gripper"][0])), 0.200001)
        self.assertLessEqual(np.max(np.abs(action["action.right_gripper"][0] - obs["observation.state.right_gripper"][0])), 0.200001)

    def test_rejects_invalid_action_safety_environment(self):
        os.environ["UNIBOT_ENABLE_ACTION_SAFETY"] = "maybe"
        with self.assertRaisesRegex(ValueError, "UNIBOT_ENABLE_ACTION_SAFETY"):
            ExamplePolicy()

        os.environ["UNIBOT_ENABLE_ACTION_SAFETY"] = "true"
        os.environ["UNIBOT_MAX_JOINT_DELTA"] = "-0.1"
        with self.assertRaisesRegex(ValueError, "UNIBOT_MAX_JOINT_DELTA"):
            ExamplePolicy()

    def test_get_action_logs_every_30_frames_without_images(self):
        with self.assertLogs(level="INFO") as logs:
            self.policy.get_action(self.obs)
        first_log = "\n".join(logs.output)
        self.assertIn("step=0", first_log)
        self.assertIn("inference_ms=", first_log)
        self.assertIn("observation.state.left_arm", first_log)
        self.assertIn("action.left_arm", first_log)
        self.assertNotIn("observation.images", first_log)
        self.assertNotIn("dev-token", first_log)

        with self.assertNoLogs(level="INFO"):
            self.policy.get_action(self.obs)

        self.policy._step = 30
        with self.assertLogs(level="INFO") as logs:
            self.policy.get_action(self.obs)
        self.assertIn("step=30", "\n".join(logs.output))

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
