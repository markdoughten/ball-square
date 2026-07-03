import unittest

import numpy as np

from src.lib.fluid import FluidDynamics, FluidRegion, NavierStokesField


class FluidDynamicsTests(unittest.TestCase):
    def test_submersion_fraction(self):
        region = FluidRegion()
        self.assertEqual(region.submerged_fraction([-0.3, 0.0], 0.02), 0.0)
        self.assertAlmostEqual(region.submerged_fraction([0.8, 0.0], 0.02), 1.0)
        self.assertAlmostEqual(region.submerged_fraction([-0.15, 0.0], 0.02), 0.5)

    def test_drag_opposes_relative_motion(self):
        fluid = FluidDynamics(current_velocity=np.zeros(2))
        force = fluid.force([0.8, 0.0], [0.2, 0.0], radius=0.02, dt=0.1)
        self.assertLess(force[0], 0.0)
        self.assertAlmostEqual(force[1], 0.0)

    def test_current_pushes_stationary_ball(self):
        fluid = FluidDynamics(current_velocity=np.array([-0.1, 0.0]))
        force = fluid.force([0.8, 0.0], [0.0, 0.0], radius=0.02, dt=0.1)
        self.assertLess(force[0], 0.0)

    def test_archimedes_submerged_volume(self):
        fluid = FluidDynamics()
        sphere_volume = 4.0 / 3.0 * np.pi * 0.02**3
        self.assertAlmostEqual(fluid.submerged_volume(0.05, 0.02), sphere_volume)
        self.assertEqual(fluid.submerged_volume(0.2, 0.02), 0.0)
        # At the surface the sphere is half submerged and buoyancy equals weight.
        self.assertAlmostEqual(
            fluid.buoyancy_force(5.0, 0.15, 0.02, 1.0, 9.81), 49.05
        )
        # Full submersion supplies positive lift toward the surface.
        self.assertAlmostEqual(
            fluid.buoyancy_force(5.0, 0.05, 0.02, 1.0, 9.81), 98.1
        )

    def test_navier_stokes_field_stays_finite(self):
        field = NavierStokesField(FluidRegion(), [-0.025, 0.0], nx=20, ny=12)
        field.step(0.01)
        self.assertTrue(np.isfinite(field.velocity).all())
        self.assertTrue(np.isfinite(field.pressure).all())
        self.assertTrue(np.all(field.velocity[field.solid] == 0.0))

    def test_moving_sphere_couples_back_to_fluid(self):
        field = NavierStokesField(
            FluidRegion(), [0.0, 0.0, 0.0], nx=24, ny=16, nz=10
        )
        position = np.array([0.2, 0.0, 0.08])
        body_velocity = np.array([0.1, 0.0, 0.0])
        field.set_moving_sphere(position, 0.04, body_velocity)
        field._apply_boundaries(field.velocity)
        self.assertTrue(np.any(field.moving_solid))
        np.testing.assert_allclose(
            field.velocity[field.moving_solid],
            np.broadcast_to(body_velocity, field.velocity[field.moving_solid].shape),
        )
        before = field.velocity.copy()
        field.add_reaction_force(position, 0.04, [-1.0, 0.0, 0.0], 0.01, 997.0)
        self.assertGreater(np.linalg.norm(field.velocity-before), 0.0)

    def test_turbulence_changes_flow_field(self):
        turbulent = NavierStokesField(
            FluidRegion(), [-0.025, 0.0], nx=20, ny=12
        )
        laminar = NavierStokesField(
            FluidRegion(), [-0.025, 0.0], nx=20, ny=12,
            turbulence_intensity=0.0, vorticity_confinement=0.0,
        )
        turbulent.step(0.01)
        laminar.step(0.01)
        self.assertGreater(
            np.linalg.norm(turbulent.velocity-laminar.velocity), 0.0
        )
        self.assertEqual(turbulent.vorticity().shape, (10, 12, 20, 3))
        self.assertTrue(np.isfinite(turbulent.vorticity()).all())

    def test_turbulence_seed_controls_random_realization(self):
        first = NavierStokesField(FluidRegion(), [-0.025, 0.0], seed=7)
        repeated = NavierStokesField(FluidRegion(), [-0.025, 0.0], seed=7)
        different = NavierStokesField(FluidRegion(), [-0.025, 0.0], seed=8)
        for _ in range(3):
            first.step(0.01)
            repeated.step(0.01)
            different.step(0.01)
        np.testing.assert_allclose(first.velocity, repeated.velocity)
        self.assertFalse(np.allclose(first.velocity, different.velocity))

    def test_current_direction_preserves_speed(self):
        fluid = FluidDynamics(current_velocity=np.array([-0.025, 0.0]))
        fluid.set_current_direction([0.0, 1.0])
        np.testing.assert_allclose(fluid.target_current_velocity, [0.0, 0.025, 0.0])
        original = fluid.current_velocity.copy()
        fluid.advance_current_direction(0.1)
        self.assertFalse(np.allclose(fluid.current_velocity, original))
        self.assertFalse(np.allclose(
            fluid.current_velocity, fluid.target_current_velocity
        ))
        self.assertAlmostEqual(np.linalg.norm(fluid.current_velocity), 0.025)

    def test_current_eventually_reaches_target_direction(self):
        fluid = FluidDynamics(current_velocity=np.array([-0.025, 0.0]))
        fluid.set_current_direction([0.0, 1.0])
        for _ in range(20):
            fluid.advance_current_direction(0.1)
        np.testing.assert_allclose(
            fluid.current_velocity, fluid.target_current_velocity, atol=1e-10
        )


if __name__ == "__main__":
    unittest.main()
