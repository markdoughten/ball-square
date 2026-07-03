import unittest

import numpy as np

from src.lib.core import (
    ActorCriticModel,
    observations_for_targets,
    path_point_is_free,
    pid_expert_actions,
    plan_rrt_path,
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


if __name__ == "__main__":
    unittest.main()
