import unittest
import random
from unittest.mock import patch

import numpy as np

from src.lib.core import (
    ActorCriticModel,
    create_episode_paths,
    observations_for_targets,
    path_point_is_free,
    pid_expert_actions,
    plan_navigation_path,
    plan_rrt_path,
    random_position,
)


class Planning3DTests(unittest.TestCase):
    def setUp(self):
        self.wall = {
            "x_min": 0.4, "x_max": 0.6,
            "y_min": -0.15, "y_max": 0.15,
            "z_min": 0.0, "z_max": 0.1,
        }
        self.bounds = {
            "x_min": -0.15, "x_max": 1.05,
            "y_min": -0.32, "y_max": 0.32,
            "z_min": 0.02, "z_max": 0.15,
        }

    def test_height_changes_obstacle_collision(self):
        self.assertFalse(path_point_is_free(
            [0.5, 0.0, 0.11], self.wall, self.bounds, clearance=0.02
        ))
        self.assertTrue(path_point_is_free(
            [0.5, 0.0, 0.15], self.wall, self.bounds, clearance=0.02
        ))

    def test_surface_path_goes_over_obstacle(self):
        path = plan_rrt_path(
            np.array([0.0, 0.0, 0.15]),
            np.array([0.8, 0.0, 0.15]),
            self.wall, self.bounds,
        )
        self.assertIsNotNone(path)
        self.assertTrue(all(len(point) == 3 for point in path))
        self.assertTrue(any(0.4 <= point[0] <= 0.6 for point in path))

    def test_policy_observation_and_action_are_3d(self):
        state = np.array([[0.0, 0.0, 0.15, 0.0, 0.0, 0.0]])
        target = np.array([[0.8, 0.0, 0.15]])
        self.assertEqual(observations_for_targets(state, target).shape, (1, 9))
        self.assertEqual(pid_expert_actions(state, target).shape, (1, 3))
        model = ActorCriticModel(9, 8, 8, 8, 3)
        self.assertEqual(tuple(model.actor.fc_mu.weight.shape), (3, 8))

    def test_underwater_route_uses_full_volume(self):
        random.seed(1)
        goal = np.array([0.9, -0.22, 0.06])
        path = plan_navigation_path(
            np.array([0.0, 0.0, 0.15]), goal, self.wall, self.bounds
        )
        intermediate = np.asarray(path[1:-1])
        self.assertTrue(np.any(intermediate[:, 2] < 0.14))
        self.assertTrue(np.any(np.linalg.norm(
            intermediate[:, :2]-goal[:2], axis=1
        ) > 0.05))

    def test_random_spawns_cover_safe_depths(self):
        samples = np.asarray([
            random_position(
                self.wall, self.bounds, np.array([0.9, -0.22, 0.06]),
                safety_margin=0.025,
            )
            for _ in range(200)
        ])
        self.assertLess(samples[:, 2].min(), 0.05)
        self.assertGreater(samples[:, 2].max(), 0.13)
        inside_obstacle = (
            (samples[:, 0] >= self.wall["x_min"]-0.025)
            & (samples[:, 0] <= self.wall["x_max"]+0.025)
            & (samples[:, 1] >= self.wall["y_min"]-0.025)
            & (samples[:, 1] <= self.wall["y_max"]+0.025)
            & (samples[:, 2] <= self.wall["z_max"]+0.025)
        )
        self.assertFalse(inside_obstacle.any())

    def test_failed_rrt_retries_without_resampling_spawn(self):
        states = np.array([[0.0, 0.0, 0.05, 0.0, 0.0, 0.0]])
        original = states.copy()
        goal = np.array([0.9, -0.22, 0.06])
        planned_path = [states[0, :3].copy(), goal]
        with patch(
            "src.lib.core.plan_rrt_path", side_effect=[None, planned_path]
        ) as planner:
            paths = create_episode_paths(
                states, goal, self.wall, self.bounds, planning_retries=2,
            )

        self.assertEqual(planner.call_count, 2)
        np.testing.assert_allclose(states, original)
        np.testing.assert_allclose(paths[0], planned_path)


if __name__ == "__main__":
    unittest.main()
