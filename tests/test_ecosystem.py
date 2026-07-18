import unittest

import numpy as np
import torch

from src.lib.ecosystem import (
    Creature,
    Ecosystem,
    EcosystemConfig,
    EcosystemController,
    FoodResource,
    MorphologyGenome,
    aggregate_energy_rankings,
    combine_energy_groups,
    ecosystem_mujoco_xml,
    energy_by_shape_orientation,
    inside_obstacle,
    plot_aggregate_energy_rankings,
    plot_energy_by_shape_orientation,
    render_ecosystem_snapshot,
    run_repeated_ecosystem_simulations,
    run_long_ecosystem_simulation,
    sync_ecosystem_to_mujoco,
)


def constant_controller(action):
    controller = EcosystemController()
    with torch.no_grad():
        for parameter in controller.parameters():
            parameter.zero_()
        controller.network[-1].bias.copy_(
            torch.atanh(torch.as_tensor(action, dtype=torch.float32) * 0.95)
        )
    return controller


def parameter_vector(model):
    return torch.cat([parameter.detach().flatten() for parameter in model.parameters()])


class EcosystemTests(unittest.TestCase):
    def test_creature_eats_food_for_energy(self):
        ecosystem = Ecosystem(
            EcosystemConfig(initial_creatures=0, food_count=0),
            seed=1,
        )
        creature = Creature(
            id=1,
            genome=MorphologyGenome(mass=1.0),
            controller=constant_controller([0.0, 0.0, 0.0, 0.0]),
            position=np.array([0.2, 0.0, 0.08]),
            energy=50.0,
        )
        ecosystem.creatures = [creature]
        ecosystem.food = [
            FoodResource(position=creature.position.copy(), energy=25.0)
        ]

        ecosystem.step()

        self.assertGreater(creature.energy, 70.0)
        self.assertEqual(creature.food_eaten, 1)
        self.assertEqual(len(ecosystem.food), 0)

    def test_predator_can_eat_nearby_prey(self):
        config = EcosystemConfig(
            initial_creatures=0,
            food_count=0,
            attack_threshold=0.1,
        )
        ecosystem = Ecosystem(config, seed=2)
        predator = Creature(
            id=1,
            genome=MorphologyGenome(mass=1.0, aggression=1.0, defense=1.0),
            controller=constant_controller([0.0, 0.0, 0.0, 1.0]),
            position=np.array([0.2, 0.0, 0.08]),
            energy=50.0,
        )
        prey = Creature(
            id=2,
            genome=MorphologyGenome(mass=1.0, aggression=0.0, defense=0.0),
            controller=constant_controller([0.0, 0.0, 0.0, 0.0]),
            position=np.array([0.22, 0.0, 0.08]),
            energy=40.0,
        )
        ecosystem.creatures = [predator, prey]

        ecosystem.step()

        self.assertEqual([creature.id for creature in ecosystem.creatures], [1])
        self.assertEqual(predator.prey_eaten, 1)
        self.assertGreater(predator.energy, 70.0)

    def test_reproduction_mutates_child_body_and_controller(self):
        config = EcosystemConfig(
            initial_creatures=0,
            food_count=0,
            reproduction_threshold=100.0,
            child_energy=40.0,
            controller_mutation_rate=1.0,
            controller_mutation_strength=0.05,
            controller_reset_rate=0.0,
        )
        ecosystem = Ecosystem(config, seed=3)
        parent = Creature(
            id=1,
            genome=MorphologyGenome(mass=2.0, mutation_scale=0.2),
            controller=constant_controller([0.0, 0.0, 0.0, 0.0]),
            position=np.array([0.2, 0.0, 0.08]),
            energy=150.0,
        )
        parent_parameters = parameter_vector(parent.controller).clone()
        ecosystem.creatures = [parent]
        ecosystem.next_creature_id = 2

        children = ecosystem.reproduce()

        self.assertEqual(len(children), 1)
        self.assertEqual(len(ecosystem.creatures), 2)
        self.assertLess(parent.energy, 150.0)
        self.assertNotEqual(children[0].genome, parent.genome)
        self.assertFalse(torch.allclose(
            parameter_vector(children[0].controller),
            parent_parameters,
        ))

    def test_mate_reproduction_requires_two_nearby_eligible_creatures(self):
        config = EcosystemConfig(
            initial_creatures=0,
            food_count=0,
            reproduction_threshold=100.0,
            child_energy=35.0,
            mate_energy_cost=20.0,
            require_mate_for_reproduction=True,
            mating_distance=0.1,
            fixed_creature_radius=0.02,
            controller_mutation_rate=1.0,
            controller_mutation_strength=0.05,
            controller_reset_rate=0.0,
        )
        ecosystem = Ecosystem(config, seed=8)
        first = Creature(
            id=1,
            genome=MorphologyGenome(size=(0.02, 0.02, 0.02), mass=2.0),
            controller=constant_controller([0.0, 0.0, 0.0, 0.0]),
            position=np.array([0.2, 0.0, 0.08]),
            energy=130.0,
        )
        second = Creature(
            id=2,
            genome=MorphologyGenome(size=(0.02, 0.02, 0.02), mass=3.0),
            controller=constant_controller([0.0, 0.0, 0.0, 0.0]),
            position=np.array([0.24, 0.0, 0.08]),
            energy=125.0,
        )
        ecosystem.creatures = [first, second]
        ecosystem.next_creature_id = 3

        children = ecosystem.reproduce()

        self.assertEqual(len(children), 1)
        self.assertEqual(len(ecosystem.creatures), 3)
        self.assertAlmostEqual(children[0].genome.radius, 0.02)
        self.assertLess(first.energy, 130.0)
        self.assertLess(second.energy, 125.0)

    def test_mate_reproduction_does_not_happen_when_partner_is_far(self):
        config = EcosystemConfig(
            initial_creatures=0,
            food_count=0,
            reproduction_threshold=100.0,
            require_mate_for_reproduction=True,
            mating_distance=0.03,
        )
        ecosystem = Ecosystem(config, seed=9)
        ecosystem.creatures = [
            Creature(
                id=1,
                genome=MorphologyGenome(),
                controller=constant_controller([0.0, 0.0, 0.0, 0.0]),
                position=np.array([0.2, 0.0, 0.08]),
                energy=130.0,
            ),
            Creature(
                id=2,
                genome=MorphologyGenome(),
                controller=constant_controller([0.0, 0.0, 0.0, 0.0]),
                position=np.array([0.4, 0.0, 0.08]),
                energy=130.0,
            ),
        ]

        self.assertEqual(ecosystem.reproduce(), [])

    def test_ten_step_run_keeps_population_valid(self):
        ecosystem = Ecosystem(
            EcosystemConfig(initial_creatures=4, food_count=6, max_population=12),
            seed=4,
        )

        history = ecosystem.run(10)

        self.assertEqual(len(history), 10)
        self.assertLessEqual(history[-1]["population"], 12)
        self.assertEqual(history[-1]["food"], 6)
        self.assertGreaterEqual(history[-1]["population"], 0)

    def test_random_spawns_and_food_avoid_obstacle(self):
        ecosystem = Ecosystem(
            EcosystemConfig(initial_creatures=20, food_count=20),
            seed=20,
        )

        self.assertFalse(any(
            inside_obstacle(creature.position, creature.genome.radius)
            for creature in ecosystem.creatures
        ))
        self.assertFalse(any(
            inside_obstacle(food.position, food.radius)
            for food in ecosystem.food
        ))

    def test_creature_is_pushed_out_of_obstacle(self):
        ecosystem = Ecosystem(
            EcosystemConfig(initial_creatures=0, food_count=0),
            seed=21,
        )
        creature = Creature(
            id=1,
            genome=MorphologyGenome(size=(0.02, 0.02, 0.02)),
            controller=constant_controller([0.0, 0.0, 0.0, 0.0]),
            position=np.array([0.5, 0.0, 0.05]),
            velocity=np.zeros(3),
        )

        ecosystem.keep_inside_region(creature)

        self.assertFalse(inside_obstacle(creature.position, creature.genome.radius))

    def test_creature_dies_of_old_age(self):
        ecosystem = Ecosystem(
            EcosystemConfig(initial_creatures=0, food_count=0, max_age=3),
            seed=23,
        )
        creature = Creature(
            id=1,
            genome=MorphologyGenome(),
            controller=constant_controller([0.0, 0.0, 0.0, 0.0]),
            position=np.array([0.2, 0.0, 0.12]),
            energy=100.0,
            age=4,
        )
        ecosystem.creatures = [creature]

        ecosystem.remove_dead()

        self.assertEqual(ecosystem.creatures, [])
        self.assertEqual(ecosystem.summary()["old_age_deaths"], 1)
        self.assertEqual(ecosystem.summary()["starvation_deaths"], 0)

    def test_old_age_increases_energy_cost_after_threshold(self):
        ecosystem = Ecosystem(
            EcosystemConfig(
                initial_creatures=0,
                food_count=0,
                old_age_start=1,
                max_age=10,
                old_age_metabolism_rate=1.0,
                metabolism_rate=0.0,
                movement_cost=0.0,
            ),
            seed=24,
        )
        creature = Creature(
            id=1,
            genome=MorphologyGenome(mass=2.0),
            controller=constant_controller([0.0, 0.0, 0.0, 0.0]),
            position=np.array([0.2, 0.0, 0.12]),
            energy=100.0,
            age=6,
        )
        ecosystem.creatures = [creature]

        ecosystem.move_creature(creature, np.zeros(4))

        self.assertLess(creature.energy, 100.0)

    def test_long_run_helper_returns_history(self):
        _, history = run_long_ecosystem_simulation(
            steps=5,
            seed=5,
            config=EcosystemConfig(initial_creatures=3, food_count=4),
            report_interval=10,
        )

        self.assertEqual(len(history), 5)
        self.assertIn("population", history[-1])

    def test_render_snapshot_draws_without_display(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ecosystem = Ecosystem(
            EcosystemConfig(initial_creatures=2, food_count=3),
            seed=6,
        )
        figure, axis = plt.subplots()
        try:
            render_ecosystem_snapshot(ecosystem, axis, step=1)
            self.assertGreater(len(axis.collections), 0)
        finally:
            plt.close(figure)

    def test_ecosystem_mujoco_xml_compiles_and_syncs(self):
        import mujoco

        config = EcosystemConfig(
            initial_creatures=2,
            food_count=3,
            max_population=4,
            use_navier_stokes=True,
        )
        ecosystem = Ecosystem(config, seed=7)
        model = mujoco.MjModel.from_xml_string(
            ecosystem_mujoco_xml(config.max_population, config.food_count)
        )
        data = mujoco.MjData(model)

        sync_ecosystem_to_mujoco(ecosystem, model, data)

        geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "creature_sphere_0"
        )
        self.assertGreater(model.geom_size[geom_id, 0], 0.0)
        self.assertIsNotNone(ecosystem.fluid_field)

    def test_configured_creature_and_food_sizes_match_original_markers(self):
        config = EcosystemConfig(
            initial_creatures=2,
            food_count=3,
            fixed_creature_radius=0.02,
            food_radius=0.01,
        )
        ecosystem = Ecosystem(config, seed=10)

        self.assertTrue(all(
            creature.genome.radius == 0.02
            for creature in ecosystem.creatures
        ))
        self.assertTrue(all(food.radius == 0.01 for food in ecosystem.food))

    def test_start_nearby_pair_uses_energy_range_and_mating_distance(self):
        config = EcosystemConfig(
            initial_creatures=2,
            initial_energy_range=(125.0, 125.0),
            start_nearby_pair=True,
            mating_distance=0.12,
        )
        ecosystem = Ecosystem(config, seed=14)

        distance = np.linalg.norm(
            ecosystem.creatures[0].position - ecosystem.creatures[1].position
        )
        self.assertLessEqual(distance, config.mating_distance)
        self.assertEqual(ecosystem.creatures[0].energy, 125.0)
        self.assertEqual(ecosystem.creatures[1].energy, 125.0)

    def test_guided_hungry_creature_steers_toward_food(self):
        config = EcosystemConfig(
            initial_creatures=0,
            food_count=0,
            guided_behavior=True,
            guidance_strength=1.0,
        )
        ecosystem = Ecosystem(config, seed=16)
        creature = Creature(
            id=1,
            genome=MorphologyGenome(),
            controller=constant_controller([0.0, 0.0, 0.0, 0.0]),
            position=np.array([0.2, 0.0, 0.08]),
            energy=50.0,
        )
        ecosystem.creatures = [creature]
        ecosystem.food = [
            FoodResource(position=np.array([0.4, 0.0, 0.08]))
        ]

        action = ecosystem.controller_action(creature)

        self.assertGreater(action[0], 0.8)
        self.assertAlmostEqual(action[1], 0.0, delta=0.2)

    def test_guided_ready_creature_steers_toward_mate(self):
        config = EcosystemConfig(
            initial_creatures=0,
            food_count=0,
            guided_behavior=True,
            guidance_strength=1.0,
            reproduction_threshold=100.0,
        )
        ecosystem = Ecosystem(config, seed=17)
        first = Creature(
            id=1,
            genome=MorphologyGenome(),
            controller=constant_controller([0.0, 0.0, 0.0, 0.0]),
            position=np.array([0.2, 0.0, 0.08]),
            energy=120.0,
        )
        second = Creature(
            id=2,
            genome=MorphologyGenome(),
            controller=constant_controller([0.0, 0.0, 0.0, 0.0]),
            position=np.array([0.4, 0.0, 0.08]),
            energy=120.0,
        )
        ecosystem.creatures = [first, second]
        ecosystem.food = [
            FoodResource(position=np.array([0.0, 0.0, 0.08]))
        ]

        action = ecosystem.controller_action(first)

        self.assertGreater(action[0], 0.8)

    def test_fixed_radius_still_allows_shape_mutation(self):
        config = EcosystemConfig(
            initial_creatures=0,
            food_count=0,
            fixed_creature_radius=0.02,
            shape_mutation_rate=1.0,
        )
        ecosystem = Ecosystem(config, seed=11)
        genome = MorphologyGenome(
            shape_type="ellipsoid",
            size=(0.05, 0.02, 0.01),
        )

        normalized = ecosystem.normalize_genome(genome)

        self.assertEqual(normalized.shape_type, "ellipsoid")
        self.assertAlmostEqual(normalized.radius, 0.02)
        self.assertNotEqual(normalized.size, (0.02, 0.02, 0.02))

    def test_shape_specific_drag_depends_on_orientation(self):
        aligned = MorphologyGenome(
            shape_type="capsule",
            size=(0.02, 0.01, 0.01),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )
        broadside = MorphologyGenome(
            shape_type="capsule",
            size=(0.02, 0.01, 0.01),
            orientation=(0.70710678, 0.0, 0.0, 0.70710678),
        )

        aligned_area, aligned_cd = aligned.drag_properties([1.0, 0.0, 0.0])
        broadside_area, broadside_cd = broadside.drag_properties([1.0, 0.0, 0.0])

        self.assertLess(aligned_area * aligned_cd, broadside_area * broadside_cd)

    def test_sphere_drag_is_orientation_invariant(self):
        first = MorphologyGenome(
            shape_type="sphere",
            size=(0.02, 0.02, 0.02),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )
        second = MorphologyGenome(
            shape_type="sphere",
            size=(0.02, 0.02, 0.02),
            orientation=(0.70710678, 0.0, 0.0, 0.70710678),
        )

        self.assertEqual(
            first.drag_properties([1.0, 0.0, 0.0]),
            second.drag_properties([1.0, 0.0, 0.0]),
        )

    def test_reward_guided_mutation_selects_best_candidate(self):
        config = EcosystemConfig(
            initial_creatures=0,
            food_count=0,
            fixed_creature_radius=0.02,
            reward_guided_mutation=True,
            mutation_candidates=2,
        )
        ecosystem = Ecosystem(config, seed=13)
        weak = MorphologyGenome(
            shape_type="sphere",
            size=(0.02, 0.02, 0.02),
            mass=10.0,
            motor_force=5.0,
            sensor_range=0.05,
            defense=0.0,
            aggression=1.0,
        )
        strong = MorphologyGenome(
            shape_type="ellipsoid",
            size=(0.02, 0.01, 0.01),
            mass=1.0,
            motor_force=60.0,
            sensor_range=0.7,
            defense=1.0,
            aggression=0.0,
        )

        original_mutated = MorphologyGenome.mutated
        try:
            candidates = iter([weak, strong])
            MorphologyGenome.mutated = lambda self, rng, shape_mutation_rate=0.03: next(candidates)
            selected = ecosystem.mutate_genome(MorphologyGenome())
        finally:
            MorphologyGenome.mutated = original_mutated

        self.assertEqual(selected.shape_type, "ellipsoid")
        self.assertAlmostEqual(selected.radius, 0.02)

    def test_reward_guided_mutation_evaluates_shape_alternatives(self):
        config = EcosystemConfig(
            initial_creatures=0,
            food_count=0,
            fixed_creature_radius=0.02,
            reward_guided_mutation=True,
            mutation_candidates=1,
            shape_mutation_rate=0.0,
        )
        ecosystem = Ecosystem(config, seed=15)

        selected = ecosystem.mutate_genome(MorphologyGenome())

        self.assertIn(selected.shape_type, {"ellipsoid", "capsule"})
        self.assertNotEqual(selected.shape_type, "sphere")
        self.assertAlmostEqual(selected.radius, 0.02)

    def test_sync_shows_only_active_mutated_shape(self):
        import mujoco

        config = EcosystemConfig(
            initial_creatures=0,
            food_count=0,
            max_population=1,
            fixed_creature_radius=0.02,
        )
        ecosystem = Ecosystem(config, seed=12)
        ecosystem.creatures = [
            Creature(
                id=1,
                genome=MorphologyGenome(
                    shape_type="ellipsoid",
                    size=(0.02, 0.01, 0.008),
                ),
                controller=constant_controller([0.0, 0.0, 0.0, 0.0]),
                position=np.array([0.2, 0.0, 0.08]),
            )
        ]
        model = mujoco.MjModel.from_xml_string(ecosystem_mujoco_xml(1, 0))
        data = mujoco.MjData(model)

        sync_ecosystem_to_mujoco(ecosystem, model, data)

        sphere = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "creature_sphere_0"
        )
        ellipsoid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "creature_ellipsoid_0"
        )
        capsule = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "creature_capsule_0"
        )
        self.assertEqual(model.geom_rgba[sphere, 3], 0.0)
        self.assertGreater(model.geom_rgba[ellipsoid, 3], 0.0)
        self.assertEqual(model.geom_rgba[capsule, 3], 0.0)

    def test_sync_applies_mutated_orientation(self):
        import mujoco

        config = EcosystemConfig(
            initial_creatures=0,
            food_count=0,
            max_population=1,
        )
        ecosystem = Ecosystem(config, seed=18)
        orientation = (0.5, 0.5, 0.5, 0.5)
        ecosystem.creatures = [
            Creature(
                id=1,
                genome=MorphologyGenome(
                    shape_type="ellipsoid",
                    size=(0.02, 0.01, 0.008),
                    orientation=orientation,
                ),
                controller=constant_controller([0.0, 0.0, 0.0, 0.0]),
                position=np.array([0.2, 0.0, 0.08]),
            )
        ]
        model = mujoco.MjModel.from_xml_string(ecosystem_mujoco_xml(1, 0))
        data = mujoco.MjData(model)

        sync_ecosystem_to_mujoco(ecosystem, model, data)

        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "creature_joint_0"
        )
        qpos_address = model.jnt_qposadr[joint_id]
        np.testing.assert_allclose(
            data.qpos[qpos_address + 3:qpos_address + 7],
            orientation,
        )

    def test_guided_hungry_predator_steers_toward_prey(self):
        config = EcosystemConfig(
            initial_creatures=0,
            food_count=0,
            guided_behavior=True,
            guidance_strength=1.0,
            reproduction_threshold=100.0,
            predation_enabled=True,
            predator_aggression_threshold=0.7,
            predator_hunger_threshold=1.2,
        )
        ecosystem = Ecosystem(config, seed=19)
        predator = Creature(
            id=1,
            genome=MorphologyGenome(aggression=1.0),
            controller=constant_controller([0.0, 0.0, 0.0, 0.0]),
            position=np.array([0.2, 0.0, 0.08]),
            energy=90.0,
        )
        prey = Creature(
            id=2,
            genome=MorphologyGenome(aggression=0.0),
            controller=constant_controller([0.0, 0.0, 0.0, 0.0]),
            position=np.array([0.4, 0.0, 0.08]),
            energy=50.0,
        )
        ecosystem.creatures = [predator, prey]

        action = ecosystem.controller_action(predator)

        self.assertGreater(action[0], 0.8)
        self.assertGreater(action[3], 0.8)

    def test_energy_grouping_and_plot_by_shape_orientation(self):
        import os
        import tempfile

        config = EcosystemConfig(initial_creatures=0, food_count=0)
        ecosystem = Ecosystem(config, seed=22)
        ecosystem.creatures = [
            Creature(
                id=1,
                genome=MorphologyGenome(
                    shape_type="sphere",
                    orientation=(1.0, 0.0, 0.0, 0.0),
                ),
                controller=constant_controller([0.0, 0.0, 0.0, 0.0]),
                position=np.array([0.2, 0.0, 0.08]),
                energy=10.0,
            ),
            Creature(
                id=2,
                genome=MorphologyGenome(
                    shape_type="ellipsoid",
                    orientation=(0.5, 0.5, 0.5, 0.5),
                ),
                controller=constant_controller([0.0, 0.0, 0.0, 0.0]),
                position=np.array([0.3, 0.0, 0.08]),
                energy=20.0,
            ),
        ]

        groups = energy_by_shape_orientation(ecosystem)
        self.assertEqual(sum(len(values) for values in groups.values()), 2)
        with tempfile.TemporaryDirectory() as directory:
            output_path = os.path.join(directory, "energy.png")
            plot_energy_by_shape_orientation(ecosystem, output_path)
            self.assertTrue(os.path.exists(output_path))

    def test_aggregate_energy_ranking_across_runs(self):
        import os
        import tempfile

        first = Ecosystem(EcosystemConfig(initial_creatures=0, food_count=0), seed=25)
        second = Ecosystem(EcosystemConfig(initial_creatures=0, food_count=0), seed=26)
        first.creatures = [
            Creature(
                id=1,
                genome=MorphologyGenome(shape_type="capsule"),
                controller=constant_controller([0.0, 0.0, 0.0, 0.0]),
                position=np.array([0.2, 0.0, 0.08]),
                energy=30.0,
            )
        ]
        second.creatures = [
            Creature(
                id=1,
                genome=MorphologyGenome(shape_type="capsule"),
                controller=constant_controller([0.0, 0.0, 0.0, 0.0]),
                position=np.array([0.2, 0.0, 0.08]),
                energy=50.0,
            )
        ]

        groups = aggregate_energy_rankings([first, second])
        self.assertEqual(sum(len(values) for values in groups.values()), 2)
        with tempfile.TemporaryDirectory() as directory:
            output_path = os.path.join(directory, "aggregate.png")
            ranked = plot_aggregate_energy_rankings(groups, output_path)
            self.assertTrue(os.path.exists(output_path))
            self.assertEqual(len(ranked), 1)

    def test_repeated_seed_runner_combines_energy_groups(self):
        progress = []
        results, groups = run_repeated_ecosystem_simulations(
            steps=2,
            seeds=[1, 2],
            config=EcosystemConfig(
                initial_creatures=1,
                food_count=0,
                max_population=2,
            ),
            max_workers=1,
            progress_callback=lambda completed, total, result, elapsed: progress.append(
                (completed, total, result["seed"])
            ),
        )

        self.assertEqual([result["seed"] for result in results], [1, 2])
        self.assertGreater(sum(len(values) for values in groups.values()), 0)
        self.assertEqual(progress, [(1, 2, 1), (2, 2, 2)])

    def test_combine_energy_groups(self):
        combined = combine_energy_groups([
            {("sphere", "+x"): [1.0]},
            {("sphere", "+x"): [2.0], ("capsule", "-y"): [3.0]},
        ])

        self.assertEqual(combined[("sphere", "+x")], [1.0, 2.0])
        self.assertEqual(combined[("capsule", "-y")], [3.0])


if __name__ == "__main__":
    unittest.main()
