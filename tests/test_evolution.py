import unittest

import torch

from src.lib.core import ActorCriticModel
from src.lib.evolution import crossover_models, mutate_model, reproduce_model


def parameter_vector(model):
    return torch.cat([parameter.detach().flatten() for parameter in model.parameters()])


class EvolutionTests(unittest.TestCase):
    def test_reproduce_mutates_child_without_changing_parent(self):
        parent = ActorCriticModel(9, 8, 8, 8, 3)
        before_parent = parameter_vector(parent).clone()
        generator = torch.Generator().manual_seed(7)

        child = reproduce_model(
            parent,
            mutation_rate=1.0,
            mutation_strength=0.05,
            generator=generator,
        )

        torch.testing.assert_close(parameter_vector(parent), before_parent)
        self.assertFalse(torch.allclose(parameter_vector(child), before_parent))

    def test_zero_mutation_rate_keeps_weights_unchanged(self):
        model = ActorCriticModel(9, 8, 8, 8, 3)
        before = parameter_vector(model).clone()

        mutate_model(model, mutation_rate=0.0, mutation_strength=1.0)

        torch.testing.assert_close(parameter_vector(model), before)

    def test_crossover_inherits_from_both_parents(self):
        parent_a = ActorCriticModel(9, 8, 8, 8, 3)
        parent_b = ActorCriticModel(9, 8, 8, 8, 3)
        for parameter in parent_a.parameters():
            parameter.data.fill_(1.0)
        for parameter in parent_b.parameters():
            parameter.data.fill_(2.0)

        generator = torch.Generator().manual_seed(3)
        child = crossover_models(parent_a, parent_b, generator=generator)
        values = parameter_vector(child)

        self.assertTrue(torch.any(values == 1.0))
        self.assertTrue(torch.any(values == 2.0))
        self.assertTrue(torch.all((values == 1.0) | (values == 2.0)))


if __name__ == "__main__":
    unittest.main()
