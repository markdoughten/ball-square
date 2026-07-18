"""Evolution helpers for mutating neural controllers."""

from copy import deepcopy

import torch


def clone_model(model):
    """Return an independent copy of a PyTorch model."""
    return deepcopy(model)


def _random_like(parameter, generator=None):
    return torch.rand(
        parameter.shape,
        dtype=parameter.dtype,
        device=parameter.device,
        generator=generator,
    )


def _normal_like(parameter, generator=None):
    return torch.randn(
        parameter.shape,
        dtype=parameter.dtype,
        device=parameter.device,
        generator=generator,
    )


def mutate_model(
    model,
    mutation_rate=0.05,
    mutation_strength=0.02,
    reset_rate=0.0,
    reset_scale=0.1,
    generator=None,
):
    """Mutate model parameters in place and return the same model.

    ``mutation_rate`` controls the chance that each scalar weight receives
    Gaussian noise. ``reset_rate`` controls rare larger jumps by replacing a
    scalar weight with a fresh normal sample.
    """
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be between 0 and 1")
    if not 0.0 <= reset_rate <= 1.0:
        raise ValueError("reset_rate must be between 0 and 1")
    if mutation_strength < 0.0:
        raise ValueError("mutation_strength must be non-negative")
    if reset_scale < 0.0:
        raise ValueError("reset_scale must be non-negative")

    with torch.no_grad():
        for parameter in model.parameters():
            if mutation_rate > 0.0 and mutation_strength > 0.0:
                mutation_mask = _random_like(parameter, generator) < mutation_rate
                parameter.add_(
                    mutation_mask.to(parameter.dtype)
                    * _normal_like(parameter, generator)
                    * mutation_strength
                )
            if reset_rate > 0.0:
                reset_mask = _random_like(parameter, generator) < reset_rate
                reset_values = _normal_like(parameter, generator) * reset_scale
                parameter.copy_(torch.where(reset_mask, reset_values, parameter))
    return model


def reproduce_model(parent, **mutation_kwargs):
    """Clone a parent controller and mutate the offspring copy."""
    child = clone_model(parent)
    return mutate_model(child, **mutation_kwargs)


def crossover_models(parent_a, parent_b, inheritance_rate=0.5, generator=None):
    """Create a child by mixing scalar parameters from two parent models."""
    if not 0.0 <= inheritance_rate <= 1.0:
        raise ValueError("inheritance_rate must be between 0 and 1")

    child = clone_model(parent_a)
    with torch.no_grad():
        for child_parameter, other_parameter in zip(
            child.parameters(), parent_b.parameters()
        ):
            if child_parameter.shape != other_parameter.shape:
                raise ValueError("parent models must have matching parameter shapes")
            inherit_from_b = (
                _random_like(child_parameter, generator) < inheritance_rate
            )
            child_parameter.copy_(
                torch.where(inherit_from_b, other_parameter, child_parameter)
            )
    return child
