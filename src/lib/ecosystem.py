"""Artificial life ecosystem for evolving water creatures."""

from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import numpy as np
import torch
import torch.nn as nn

from .evolution import crossover_models, mutate_model, reproduce_model
from .fluid import FluidRegion, NavierStokesField


SHAPE_TYPES = ("sphere", "ellipsoid", "capsule")


def _clamp(value, low, high):
    return min(max(float(value), float(low)), float(high))


def _unit_or_zero(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-12 else np.zeros_like(vector)


def inside_obstacle(position, radius=0.0):
    return (
        0.4 - radius <= position[0] <= 0.6 + radius
        and -0.15 - radius <= position[1] <= 0.15 + radius
        and position[2] <= 0.10 + radius
    )


def random_orientation(rng):
    """Return a random unit quaternion in MuJoCo wxyz order."""
    vector = rng.normal(size=4)
    vector /= np.linalg.norm(vector) + 1e-12
    if vector[0] < 0:
        vector *= -1
    return tuple(float(value) for value in vector)


class EcosystemController(nn.Module):
    """Small neural controller evolved directly with each creature."""

    def __init__(self, input_dim=12, hidden_dim=16, output_dim=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, observation):
        return torch.tanh(self.network(observation))


@dataclass(frozen=True)
class MorphologyGenome:
    """Mutable body and behavior traits inherited by offspring."""

    shape_type: str = "sphere"
    size: tuple = (0.02, 0.02, 0.02)
    orientation: tuple = (1.0, 0.0, 0.0, 0.0)
    mass: float = 5.0
    motor_force: float = 15.0
    sensor_range: float = 0.25
    aggression: float = 0.5
    defense: float = 0.5
    mutation_scale: float = 0.08

    @property
    def radius(self):
        return float(max(self.size))

    @property
    def cross_section(self):
        return float(np.pi * self.size[1] * self.size[2])

    def rotation_matrix(self):
        w, x, y, z = np.asarray(self.orientation, dtype=np.float64)
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm <= 1e-12:
            return np.eye(3)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

    def drag_properties(self, relative_velocity):
        """Return projected area and drag coefficient for the flow direction."""
        direction = _unit_or_zero(np.asarray(relative_velocity, dtype=np.float64))
        if np.linalg.norm(direction) <= 1e-12:
            return self.cross_section, 0.47
        if self.shape_type == "sphere":
            return np.pi * self.radius**2, 0.47

        rotation = self.rotation_matrix()
        local_direction = rotation.T @ direction
        axes = np.asarray(self.size, dtype=np.float64)
        # Orthographic projected area of an ellipsoid along a direction.
        denominator = np.sqrt(np.sum((axes * local_direction)**2)) + 1e-12
        area = np.pi * np.prod(axes) / denominator
        alignment = abs(float(local_direction[0]))
        if self.shape_type == "capsule":
            side_area = 2.0 * self.size[0] * max(self.size[1], self.size[2])
            nose_area = np.pi * max(self.size[1], self.size[2])**2
            area = alignment * nose_area + (1.0 - alignment) * side_area
            drag_coefficient = 0.28 * alignment + 0.82 * (1.0 - alignment)
        else:
            drag_coefficient = 0.32 * alignment + 0.72 * (1.0 - alignment)
        return float(max(area, 1e-8)), float(drag_coefficient)

    def mutated(self, rng, shape_mutation_rate=0.03):
        shape_type = self.shape_type
        if rng.random() < shape_mutation_rate:
            shape_type = rng.choice([
                candidate for candidate in SHAPE_TYPES
                if candidate != self.shape_type
            ])

        scale = self.mutation_scale
        size = tuple(
            _clamp(value * (1.0 + rng.normal(0.0, scale)), 0.006, 0.08)
            for value in self.size
        )
        if shape_type == "sphere":
            mean_radius = float(np.mean(size))
            size = (mean_radius, mean_radius, mean_radius)

        return MorphologyGenome(
            shape_type=shape_type,
            size=size,
            orientation=random_orientation(rng),
            mass=_clamp(self.mass * (1.0 + rng.normal(0.0, scale)), 0.2, 20.0),
            motor_force=_clamp(
                self.motor_force * (1.0 + rng.normal(0.0, scale)), 1.0, 80.0
            ),
            sensor_range=_clamp(
                self.sensor_range * (1.0 + rng.normal(0.0, scale)), 0.05, 0.7
            ),
            aggression=_clamp(self.aggression + rng.normal(0.0, scale), 0.0, 1.0),
            defense=_clamp(self.defense + rng.normal(0.0, scale), 0.0, 1.0),
            mutation_scale=_clamp(
                self.mutation_scale * (1.0 + rng.normal(0.0, 0.03)), 0.01, 0.3
            ),
        )


@dataclass
class FoodResource:
    position: np.ndarray
    energy: float = 25.0
    radius: float = 0.01


@dataclass
class Creature:
    id: int
    genome: MorphologyGenome
    controller: EcosystemController
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    energy: float = 70.0
    age: int = 0
    alive: bool = True
    food_eaten: int = 0
    prey_eaten: int = 0
    offspring_count: int = 0
    last_attack: float = 0.0


@dataclass(frozen=True)
class EcosystemConfig:
    region: FluidRegion = field(default_factory=FluidRegion)
    initial_creatures: int = 12
    initial_energy_range: tuple = (55.0, 95.0)
    start_nearby_pair: bool = False
    food_count: int = 24
    max_population: int = 80
    dt: float = 0.05
    water_current: tuple = (-0.025, 0.0, 0.0)
    drag_coefficient: float = 0.47
    water_density: float = 997.0
    metabolism_rate: float = 0.012
    movement_cost: float = 0.00035
    reproduction_threshold: float = 135.0
    child_energy: float = 45.0
    predation_efficiency: float = 0.6
    attack_threshold: float = 0.25
    predation_enabled: bool = True
    predator_aggression_threshold: float = 0.75
    predator_hunger_threshold: float = 0.85
    old_age_start: int = 3000
    old_age_metabolism_rate: float = 0.006
    max_age: int = 4000
    controller_mutation_rate: float = 0.04
    controller_mutation_strength: float = 0.03
    controller_reset_rate: float = 0.001
    shape_mutation_rate: float = 0.03
    reward_guided_mutation: bool = False
    mutation_candidates: int = 6
    guided_behavior: bool = False
    guidance_strength: float = 0.85
    wall_avoidance_margin: float = 0.08
    fixed_creature_radius: float | None = None
    food_radius: float = 0.01
    food_energy: float = 25.0
    require_mate_for_reproduction: bool = False
    mating_distance: float = 0.08
    mate_energy_cost: float = 25.0
    use_navier_stokes: bool = False
    navier_stokes_shape: tuple = (24, 14, 8)
    navier_stokes_pressure_iterations: int = 8
    device: str = "cpu"


class Ecosystem:
    """Step-based resource competition and predation simulation."""

    def __init__(self, config=None, seed=None):
        self.config = config or EcosystemConfig()
        self.rng = np.random.default_rng(seed)
        requested_device = torch.device(self.config.device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            requested_device = torch.device("cpu")
        self.device = requested_device
        self.torch_generator = torch.Generator(device=self.device).manual_seed(
            int(self.rng.integers(0, 2**31 - 1))
        )
        self.creatures = []
        self.food = []
        self.next_creature_id = 1
        self.generation = 0
        self.total_births = 0
        self.total_births_blocked_capacity = 0
        self.total_natural_deaths = 0
        self.total_old_age_deaths = 0
        self.total_starvation_deaths = 0
        self.total_predation_deaths = 0
        self.total_food_eaten = 0
        self.total_prey_eaten = 0
        self.fluid_field = None
        if self.config.use_navier_stokes:
            nx, ny, nz = self.config.navier_stokes_shape
            self.fluid_field = NavierStokesField(
                self.config.region,
                self.config.water_current,
                nx=nx,
                ny=ny,
                nz=nz,
                pressure_iterations=self.config.navier_stokes_pressure_iterations,
                seed=seed,
            )
        for index in range(self.config.initial_creatures):
            creature = self.create_random_creature()
            if (
                self.config.start_nearby_pair
                and index == 1
                and self.creatures
            ):
                creature.position = self.creatures[0].position + np.array([
                    min(self.config.mating_distance * 0.5, 0.04),
                    0.0,
                    0.0,
                ])
                self.keep_inside_region(creature)
            self.creatures.append(creature)
        for _ in range(self.config.food_count):
            self.food.append(self.spawn_food())

    def random_position(self):
        region = self.config.region
        for _ in range(1000):
            position = np.array([
                self.rng.uniform(region.x_min, region.x_max),
                self.rng.uniform(region.y_min, region.y_max),
                self.rng.uniform(region.bottom_z + 0.02, region.surface_z - 0.01),
            ], dtype=np.float64)
            if not inside_obstacle(position, radius=0.025):
                return position
        return np.array([region.x_min + 0.05, region.y_min + 0.05, 0.12])

    def random_genome(self):
        if self.config.fixed_creature_radius is not None:
            radius = self.config.fixed_creature_radius
            shape_type = "sphere"
            size = (radius, radius, radius)
        else:
            shape_type = self.rng.choice(SHAPE_TYPES)
            if shape_type == "sphere":
                radius = self.rng.uniform(0.012, 0.035)
                size = (radius, radius, radius)
            else:
                size = tuple(self.rng.uniform(0.01, 0.045, size=3))
        return MorphologyGenome(
            shape_type=shape_type,
            size=size,
            mass=self.rng.uniform(1.0, 8.0),
            motor_force=self.rng.uniform(6.0, 30.0),
            sensor_range=self.rng.uniform(0.15, 0.45),
            aggression=self.rng.uniform(0.0, 1.0),
            defense=self.rng.uniform(0.0, 1.0),
        )

    def create_random_creature(self):
        creature = Creature(
            id=self.next_creature_id,
            genome=self.random_genome(),
            controller=EcosystemController().to(self.device),
            position=self.random_position(),
            energy=self.rng.uniform(*self.config.initial_energy_range),
        )
        self.next_creature_id += 1
        return creature

    def spawn_food(self):
        return FoodResource(
            position=self.random_position(),
            energy=self.config.food_energy,
            radius=self.config.food_radius,
        )

    def normalize_genome(self, genome):
        if self.config.fixed_creature_radius is None:
            return genome
        radius = self.config.fixed_creature_radius
        size = np.asarray(genome.size, dtype=np.float64)
        size *= radius / max(float(np.max(size)), 1e-12)
        return MorphologyGenome(
            shape_type=genome.shape_type,
            size=tuple(float(value) for value in size),
            orientation=genome.orientation,
            mass=genome.mass,
            motor_force=genome.motor_force,
            sensor_range=genome.sensor_range,
            aggression=genome.aggression,
            defense=genome.defense,
            mutation_scale=genome.mutation_scale,
        )

    def morphology_reward(self, genome):
        """Estimate how useful a body is before spending a full lifetime on it."""
        radius = max(genome.radius, 1e-12)
        slenderness = genome.size[0] / max(genome.size[1], genome.size[2], 1e-12)
        drag_area, drag_coefficient = genome.drag_properties([1.0, 0.0, 0.0])
        drag_load = max(drag_area * drag_coefficient, 1e-12)
        thrust_to_mass = genome.motor_force / max(genome.mass, 1e-12)
        efficiency = thrust_to_mass / drag_load
        sensor_bonus = 0.35 * genome.sensor_range
        survival_bonus = 0.25 * genome.defense
        aggression_balance = 0.06 * (1.0 - abs(genome.aggression - 0.65))
        size_penalty = 0.08 * abs(radius - 0.02)
        return (
            0.0002 * efficiency
            + 0.08 * min(slenderness, 3.0)
            + sensor_bonus
            + survival_bonus
            + aggression_balance
            - size_penalty
        )

    def mutate_genome(self, genome):
        """Mutate a genome, optionally selecting the best candidate by reward."""
        if not self.config.reward_guided_mutation:
            return self.normalize_genome(genome.mutated(
                self.rng,
                shape_mutation_rate=self.config.shape_mutation_rate,
            ))
        candidates = [
            self.normalize_genome(genome.mutated(
                self.rng,
                shape_mutation_rate=self.config.shape_mutation_rate,
            ))
            for _ in range(max(1, self.config.mutation_candidates))
        ]
        candidates.extend(self.shape_reward_candidates(genome))
        return max(candidates, key=self.morphology_reward)

    def shape_reward_candidates(self, genome):
        """Create explicit shape alternatives for reward-guided selection."""
        radius = (
            self.config.fixed_creature_radius
            if self.config.fixed_creature_radius is not None
            else genome.radius
        )
        templates = {
            "sphere": (radius, radius, radius),
            "ellipsoid": (radius, 0.65 * radius, 0.5 * radius),
            "capsule": (radius, 0.55 * radius, 0.55 * radius),
        }
        candidates = []
        for shape_type, size in templates.items():
            candidate = MorphologyGenome(
                shape_type=shape_type,
                size=size,
                orientation=random_orientation(self.rng),
                mass=genome.mass,
                motor_force=genome.motor_force,
                sensor_range=genome.sensor_range,
                aggression=genome.aggression,
                defense=genome.defense,
                mutation_scale=genome.mutation_scale,
            )
            candidates.append(self.normalize_genome(candidate))
        return candidates

    def nearest_food(self, creature):
        if not self.food:
            return None, np.zeros(3), np.inf
        distances = [
            np.linalg.norm(food.position - creature.position)
            for food in self.food
        ]
        index = int(np.argmin(distances))
        food = self.food[index]
        return food, food.position - creature.position, distances[index]

    def nearest_creature(self, creature):
        others = [
            other for other in self.creatures
            if other.alive and other.id != creature.id
        ]
        if not others:
            return None, np.zeros(3), np.inf
        distances = [
            np.linalg.norm(other.position - creature.position)
            for other in others
        ]
        index = int(np.argmin(distances))
        other = others[index]
        return other, other.position - creature.position, distances[index]

    def observation_for(self, creature):
        nearest_food, food_vector, food_distance = self.nearest_food(creature)
        nearest_creature, creature_vector, creature_distance = (
            self.nearest_creature(creature)
        )
        sensor_range = max(creature.genome.sensor_range, 1e-6)
        food_visible = food_distance <= sensor_range
        creature_visible = creature_distance <= sensor_range
        food_direction = (
            food_vector / sensor_range if food_visible else np.zeros(3)
        )
        creature_direction = (
            creature_vector / sensor_range if creature_visible else np.zeros(3)
        )
        other_energy = (
            nearest_creature.energy / 150.0
            if nearest_creature is not None and creature_visible else 0.0
        )
        nearest_food_energy = (
            nearest_food.energy / 50.0
            if nearest_food is not None and food_visible else 0.0
        )
        return np.concatenate((
            creature.velocity / 1.5,
            np.clip(food_direction, -1.0, 1.0),
            np.clip(creature_direction, -1.0, 1.0),
            [creature.energy / 150.0, other_energy, nearest_food_energy],
        )).astype(np.float32)

    def controller_action(self, creature):
        observation = torch.as_tensor(
            self.observation_for(creature),
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            action = creature.controller(observation).squeeze(0).cpu().numpy()
        if self.config.guided_behavior:
            guidance = self.guided_action(creature)
            action = (
                (1.0 - self.config.guidance_strength) * action
                + self.config.guidance_strength * guidance
            )
            action = np.clip(action, -1.0, 1.0)
        creature.last_attack = max(0.0, float(action[3]))
        return action

    def guided_action(self, creature):
        """Waypoint-style steering toward food or a nearby eligible mate."""
        target_vector = None
        attack_intent = 0.0
        prey = self.predation_target(creature)
        if prey is not None:
            target_vector = prey.position - creature.position
            attack_intent = 1.0
        if creature.energy >= self.config.reproduction_threshold:
            mate = self.nearest_eligible_mate(creature)
            if mate is not None and target_vector is None:
                target_vector = mate.position - creature.position
        if target_vector is None:
            food, food_vector, _ = self.nearest_food(creature)
            if food is not None:
                target_vector = food_vector
        if target_vector is None:
            center = np.array([0.45, 0.0, 0.08])
            target_vector = center - creature.position
        steering = _unit_or_zero(target_vector) + self.wall_avoidance(creature)
        action = np.zeros(4, dtype=np.float64)
        action[:3] = np.clip(_unit_or_zero(steering), -1.0, 1.0)
        action[3] = attack_intent
        return action

    def nearest_eligible_mate(self, creature):
        candidates = [
            other for other in self.creatures
            if other.alive
            and other.id != creature.id
            and other.energy >= self.config.reproduction_threshold
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda other: np.linalg.norm(other.position - creature.position),
        )

    def predation_target(self, creature):
        if (
            not self.config.predation_enabled
            or creature.genome.aggression < self.config.predator_aggression_threshold
            or creature.energy >= (
                self.config.reproduction_threshold
                * self.config.predator_hunger_threshold
            )
        ):
            return None
        candidates = [
            other for other in self.creatures
            if other.alive
            and other.id != creature.id
            and other.energy < creature.energy
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda other: np.linalg.norm(other.position - creature.position),
        )

    def wall_avoidance(self, creature):
        region = self.config.region
        margin = self.config.wall_avoidance_margin
        position = creature.position
        push = np.zeros(3, dtype=np.float64)
        distances = (
            (position[0] - region.x_min, np.array([1.0, 0.0, 0.0])),
            (region.x_max - position[0], np.array([-1.0, 0.0, 0.0])),
            (position[1] - region.y_min, np.array([0.0, 1.0, 0.0])),
            (region.y_max - position[1], np.array([0.0, -1.0, 0.0])),
            (position[2] - region.bottom_z, np.array([0.0, 0.0, 1.0])),
            (region.surface_z - position[2], np.array([0.0, 0.0, -1.0])),
        )
        for distance, direction in distances:
            if distance < margin:
                push += direction * ((margin - distance) / margin)
        if inside_obstacle(position, creature.genome.radius + margin):
            obstacle_center = np.array([0.5, 0.0, min(position[2], 0.10)])
            push += 2.0 * _unit_or_zero(position - obstacle_center)
        return push

    def advance_fluid(self):
        if self.fluid_field is None:
            return
        live_creatures = [creature for creature in self.creatures if creature.alive]
        if live_creatures:
            self.fluid_field.set_moving_spheres(
                [creature.position for creature in live_creatures],
                [creature.genome.radius for creature in live_creatures],
                [creature.velocity for creature in live_creatures],
            )
        else:
            self.fluid_field.set_moving_spheres([], [], [])
        self.fluid_field.step(self.config.dt, self.config.water_density)

    def flow_velocity_for(self, creature):
        if self.fluid_field is None:
            return np.asarray(self.config.water_current, dtype=np.float64)
        return self.fluid_field.velocity_around_sphere(
            creature.position, creature.genome.radius
        )

    def move_creature(self, creature, action):
        config = self.config
        thrust = np.asarray(action[:3], dtype=np.float64) * creature.genome.motor_force
        current = self.flow_velocity_for(creature)
        relative_velocity = creature.velocity - current
        drag_area, drag_coefficient = creature.genome.drag_properties(
            relative_velocity
        )
        drag = (
            -0.5 * config.water_density * drag_coefficient
            * drag_area * np.linalg.norm(relative_velocity) * relative_velocity
        )
        acceleration = (thrust + drag) / max(creature.genome.mass, 1e-6)
        creature.velocity += acceleration * config.dt
        speed = np.linalg.norm(creature.velocity)
        max_speed = 2.0 + creature.genome.motor_force / 80.0
        if speed > max_speed:
            creature.velocity *= max_speed / speed
        creature.position += creature.velocity * config.dt
        self.keep_inside_region(creature)

        creature.energy -= config.metabolism_rate * creature.genome.mass
        if creature.age > config.old_age_start:
            age_fraction = min(
                (creature.age - config.old_age_start)
                / max(config.max_age - config.old_age_start, 1),
                1.0,
            )
            creature.energy -= (
                config.old_age_metabolism_rate
                * creature.genome.mass
                * age_fraction
            )
        creature.energy -= config.movement_cost * float(np.dot(thrust, thrust))

    def keep_inside_region(self, creature):
        region = self.config.region
        lower = np.array([region.x_min, region.y_min, region.bottom_z])
        upper = np.array([region.x_max, region.y_max, region.surface_z])
        clipped = np.clip(creature.position, lower, upper)
        hit_wall = clipped != creature.position
        creature.position = clipped
        creature.velocity[hit_wall] *= -0.25
        if np.any(hit_wall):
            creature.energy -= 0.5
        self.keep_outside_obstacle(creature)

    def keep_outside_obstacle(self, creature):
        radius = creature.genome.radius
        if not inside_obstacle(creature.position, radius=radius):
            return
        clearance = 1e-6
        x_faces = np.array([0.4 - radius - clearance, 0.6 + radius + clearance])
        y_faces = np.array([-0.15 - radius - clearance, 0.15 + radius + clearance])
        z_face = 0.10 + radius + clearance
        candidates = [
            (
                abs(creature.position[0] - x_faces[0]),
                np.array([x_faces[0], creature.position[1], creature.position[2]]),
                np.array([-0.35, 0.0, 0.0]),
            ),
            (
                abs(creature.position[0] - x_faces[1]),
                np.array([x_faces[1], creature.position[1], creature.position[2]]),
                np.array([0.35, 0.0, 0.0]),
            ),
            (
                abs(creature.position[1] - y_faces[0]),
                np.array([creature.position[0], y_faces[0], creature.position[2]]),
                np.array([0.0, -0.35, 0.0]),
            ),
            (
                abs(creature.position[1] - y_faces[1]),
                np.array([creature.position[0], y_faces[1], creature.position[2]]),
                np.array([0.0, 0.35, 0.0]),
            ),
            (
                abs(creature.position[2] - z_face),
                np.array([creature.position[0], creature.position[1], z_face]),
                np.array([0.0, 0.0, 0.25]),
            ),
        ]
        _, position, bounce = min(candidates, key=lambda candidate: candidate[0])
        creature.position = position
        creature.velocity += bounce
        creature.energy -= 0.75

    def resolve_food(self):
        remaining = []
        for food in self.food:
            eaters = [
                creature for creature in self.creatures
                if creature.alive
                and np.linalg.norm(creature.position - food.position)
                <= creature.genome.radius + food.radius
            ]
            if eaters:
                eater = max(eaters, key=lambda creature: creature.energy)
                eater.energy += food.energy
                eater.food_eaten += 1
                self.total_food_eaten += 1
            else:
                remaining.append(food)
        self.food = remaining
        while len(self.food) < self.config.food_count:
            self.food.append(self.spawn_food())

    def resolve_predation(self):
        config = self.config
        if not config.predation_enabled:
            return
        alive = [creature for creature in self.creatures if creature.alive]
        for predator in alive:
            if not predator.alive or predator.last_attack < config.attack_threshold:
                continue
            prey, prey_vector, distance = self.nearest_creature(predator)
            if prey is None or not prey.alive:
                continue
            bite_radius = predator.genome.radius + prey.genome.radius + 0.02
            if distance > bite_radius:
                continue
            attack_score = (
                predator.genome.aggression * predator.last_attack
                + 0.25 * predator.genome.radius / max(prey.genome.radius, 1e-6)
            )
            defense_score = prey.genome.defense + 0.1 * np.linalg.norm(prey.velocity)
            if attack_score > defense_score:
                predator.energy += prey.energy * config.predation_efficiency
                predator.prey_eaten += 1
                self.total_prey_eaten += 1
                self.total_predation_deaths += 1
                prey.alive = False
                prey.energy = 0.0

    def reproduce(self):
        if self.config.require_mate_for_reproduction:
            return self.reproduce_with_mates()
        return self.reproduce_asexually()

    def reproduce_asexually(self):
        config = self.config
        children = []
        for parent in list(self.creatures):
            if not parent.alive or parent.energy < config.reproduction_threshold:
                continue
            if len(self.creatures) + len(children) >= config.max_population:
                self.total_births_blocked_capacity += 1
                continue
            parent.energy -= config.child_energy
            parent.offspring_count += 1
            child_genome = self.mutate_genome(parent.genome)
            child_controller = reproduce_model(
                parent.controller,
                mutation_rate=config.controller_mutation_rate,
                mutation_strength=config.controller_mutation_strength,
                reset_rate=config.controller_reset_rate,
                generator=self.torch_generator,
            )
            offset = _unit_or_zero(self.rng.normal(size=3)) * (
                parent.genome.radius + child_genome.radius + 0.01
            )
            child = Creature(
                id=self.next_creature_id,
                genome=child_genome,
                controller=child_controller,
                position=parent.position + offset,
                velocity=parent.velocity * 0.25,
                energy=config.child_energy,
            )
            self.next_creature_id += 1
            self.keep_inside_region(child)
            children.append(child)
        self.creatures.extend(children)
        if children:
            self.generation += 1
            self.total_births += len(children)
        return children

    def mated_genome(self, first, second):
        child = MorphologyGenome(
            shape_type=first.genome.shape_type,
            size=tuple(
                (first_value + second_value) / 2
                for first_value, second_value in zip(
                    first.genome.size, second.genome.size
                )
            ),
            orientation=random_orientation(self.rng),
            mass=(first.genome.mass + second.genome.mass) / 2,
            motor_force=(first.genome.motor_force + second.genome.motor_force) / 2,
            sensor_range=(
                first.genome.sensor_range + second.genome.sensor_range
            ) / 2,
            aggression=(first.genome.aggression + second.genome.aggression) / 2,
            defense=(first.genome.defense + second.genome.defense) / 2,
            mutation_scale=(
                first.genome.mutation_scale + second.genome.mutation_scale
            ) / 2,
        )
        return self.mutate_genome(child)

    def mated_controller(self, first, second):
        child = crossover_models(
            first.controller,
            second.controller,
            generator=self.torch_generator,
        )
        return mutate_model(
            child,
            mutation_rate=self.config.controller_mutation_rate,
            mutation_strength=self.config.controller_mutation_strength,
            reset_rate=self.config.controller_reset_rate,
            generator=self.torch_generator,
        )

    def reproduce_with_mates(self):
        config = self.config
        children = []
        used_parent_ids = set()
        eligible = [
            creature for creature in self.creatures
            if creature.alive and creature.energy >= config.reproduction_threshold
        ]
        for first in eligible:
            if first.id in used_parent_ids:
                continue
            if len(self.creatures) + len(children) >= config.max_population:
                self.total_births_blocked_capacity += 1
                continue
            partners = [
                second for second in eligible
                if second.id not in used_parent_ids
                and second.id != first.id
                and np.linalg.norm(second.position - first.position)
                <= config.mating_distance
            ]
            if not partners:
                continue
            second = max(partners, key=lambda creature: creature.energy)
            used_parent_ids.update({first.id, second.id})
            first.energy -= config.mate_energy_cost
            second.energy -= config.mate_energy_cost
            first.offspring_count += 1
            second.offspring_count += 1
            child_genome = self.mated_genome(first, second)
            child_controller = self.mated_controller(first, second)
            child_position = (first.position + second.position) / 2
            offset = _unit_or_zero(self.rng.normal(size=3)) * (
                child_genome.radius + 0.01
            )
            child = Creature(
                id=self.next_creature_id,
                genome=child_genome,
                controller=child_controller,
                position=child_position + offset,
                velocity=(first.velocity + second.velocity) * 0.125,
                energy=config.child_energy,
            )
            self.next_creature_id += 1
            self.keep_inside_region(child)
            children.append(child)
        self.creatures.extend(children)
        if children:
            self.generation += 1
            self.total_births += len(children)
        return children

    def remove_dead(self):
        for creature in self.creatures:
            if not creature.alive:
                continue
            if creature.age > self.config.max_age:
                creature.alive = False
                self.total_old_age_deaths += 1
            elif creature.energy <= 0.0:
                creature.alive = False
                self.total_starvation_deaths += 1
        self.creatures = [creature for creature in self.creatures if creature.alive]
        self.total_natural_deaths = (
            self.total_old_age_deaths + self.total_starvation_deaths
        )

    def step(self):
        self.advance_fluid()
        for creature in list(self.creatures):
            if not creature.alive:
                continue
            creature.age += 1
            action = self.controller_action(creature)
            self.move_creature(creature, action)
        self.resolve_food()
        self.resolve_predation()
        self.remove_dead()
        return self.reproduce()

    def run(self, steps):
        history = []
        for _ in range(int(steps)):
            births = self.step()
            history.append(self.summary(births=len(births)))
        return history

    def summary(self, births=0):
        energies = [creature.energy for creature in self.creatures]
        return {
            "population": len(self.creatures),
            "max_population": self.config.max_population,
            "food": len(self.food),
            "generation": self.generation,
            "births_step": births,
            "births_total": self.total_births,
            "births_blocked_capacity": self.total_births_blocked_capacity,
            "deaths_total": (
                self.total_natural_deaths + self.total_predation_deaths
            ),
            "natural_deaths": self.total_natural_deaths,
            "old_age_deaths": self.total_old_age_deaths,
            "starvation_deaths": self.total_starvation_deaths,
            "predation_deaths": self.total_predation_deaths,
            "mean_energy": float(np.mean(energies)) if energies else 0.0,
            "prey_eaten": self.total_prey_eaten,
            "food_eaten": self.total_food_eaten,
            "at_capacity": len(self.creatures) >= self.config.max_population,
        }


def print_ecosystem_summary(step, summary):
    """Print a compact one-line ecosystem status."""
    print(
        f"Step {step:>6}: population={summary['population']:>3}, "
        f"capacity={summary['max_population']:>3}, "
        f"food={summary['food']:>3}, generation={summary['generation']:>3}, "
        f"births_step={summary['births_step']:>2}, "
        f"births_total={summary['births_total']:>4}, "
        f"births_blocked={summary['births_blocked_capacity']:>4}, "
        f"deaths_total={summary['deaths_total']:>4}, "
        f"old_age_deaths={summary['old_age_deaths']:>4}, "
        f"predation_deaths={summary['predation_deaths']:>4}, "
        f"mean_energy={summary['mean_energy']:>6.1f}, "
        f"food_eaten={summary['food_eaten']:>4}, "
        f"prey_eaten={summary['prey_eaten']:>4}"
    )


def orientation_label(orientation):
    """Classify a body orientation by the world axis its local x-axis faces."""
    w, x, y, z = np.asarray(orientation, dtype=np.float64)
    forward = np.array([
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
    ])
    axis = int(np.argmax(np.abs(forward)))
    sign = "+" if forward[axis] >= 0.0 else "-"
    return f"{sign}{'xyz'[axis]}"


def energy_by_shape_orientation(ecosystem):
    """Return final creature energies grouped by shape and orientation bucket."""
    groups = {}
    for creature in ecosystem.creatures:
        key = (
            str(creature.genome.shape_type),
            orientation_label(creature.genome.orientation),
        )
        groups.setdefault(key, []).append(float(creature.energy))
    return groups


def plot_energy_by_shape_orientation(ecosystem, output_path=None):
    """Plot final energy levels grouped by shape and orientation."""
    return plot_energy_groups(
        energy_by_shape_orientation(ecosystem),
        output_path=output_path,
        title="Final Energy by Mutated Shape and Orientation",
        bar_color="steelblue",
        show_points=True,
    )


def plot_energy_groups(
    groups,
    output_path=None,
    title="Final Energy by Mutated Shape and Orientation",
    bar_color="steelblue",
    show_points=True,
):
    """Plot grouped energy values where keys are ``(shape, orientation)``."""
    if output_path:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        f"{shape}\n{orientation}"
        for shape, orientation in sorted(groups)
    ]
    values = [groups[key] for key in sorted(groups)]
    means = [np.mean(group) for group in values]

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.bar(range(len(labels)), means, color="steelblue", alpha=0.75)
    for index, group in enumerate(values):
        jitter = np.linspace(-0.18, 0.18, len(group)) if len(group) > 1 else [0.0]
        axis.scatter(
            np.asarray(jitter) + index,
            group,
            color="black",
            s=18,
            alpha=0.65,
            zorder=3,
        )
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=0)
    axis.set_ylabel("Energy")
    axis.set_xlabel("Shape and orientation")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    if output_path:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()
    return groups


def aggregate_energy_rankings(ecosystems):
    """Aggregate energy groups across repeated seeded ecosystem runs."""
    groups = {}
    for ecosystem in ecosystems:
        for key, values in energy_by_shape_orientation(ecosystem).items():
            groups.setdefault(key, []).extend(values)
    return groups


def combine_energy_groups(group_sets):
    """Combine already-computed energy groups from repeated runs."""
    combined = {}
    for groups in group_sets:
        for key, values in groups.items():
            combined.setdefault(key, []).extend(values)
    return combined


def plot_aggregate_energy_rankings(groups, output_path=None):
    """Plot mean final energy ranking aggregated across repeated runs."""
    if output_path:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked = sorted(
        groups.items(),
        key=lambda item: np.mean(item[1]) if item[1] else -np.inf,
        reverse=True,
    )
    labels = [f"{shape}\n{orientation}" for (shape, orientation), _ in ranked]
    means = [np.mean(values) for _, values in ranked]
    errors = [
        np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        for _, values in ranked
    ]

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.bar(
        range(len(labels)),
        means,
        yerr=errors,
        color="seagreen",
        alpha=0.78,
        capsize=4,
    )
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels)
    axis.set_ylabel("Mean final energy")
    axis.set_xlabel("Shape and orientation")
    axis.set_title("Aggregate Energy Ranking Across Evolution Seeds")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    if output_path:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()
    return ranked


def run_long_ecosystem_simulation(
    steps=10000,
    seed=None,
    config=None,
    report_interval=500,
):
    """Run a long headless ecosystem simulation and print periodic summaries."""
    ecosystem = Ecosystem(config=config, seed=seed)
    history = []
    for step in range(1, int(steps) + 1):
        births = ecosystem.step()
        summary = ecosystem.summary(births=len(births))
        history.append(summary)
        if step == 1 or step % report_interval == 0 or step == int(steps):
            print_ecosystem_summary(step, summary)
        if not ecosystem.creatures:
            print(f"Step {step:>6}: ecosystem extinct")
            break
    return ecosystem, history


def run_ecosystem_seed_worker(args):
    """Run one seed and return serializable summary data for parallel option 6."""
    seed, steps, config, report_interval = args
    ecosystem = Ecosystem(config=config, seed=seed)
    history = []
    for step in range(1, int(steps) + 1):
        births = ecosystem.step()
        summary = ecosystem.summary(births=len(births))
        history.append(summary)
        if not ecosystem.creatures:
            break
    return {
        "seed": seed,
        "history": history,
        "final_summary": history[-1] if history else ecosystem.summary(),
        "energy_groups": energy_by_shape_orientation(ecosystem),
    }


def run_repeated_ecosystem_simulations(
    steps,
    seeds,
    config,
    report_interval=500,
    max_workers=None,
    progress_callback=None,
):
    """Run repeated seed simulations in parallel and combine final energy groups."""
    seeds = list(seeds)
    if max_workers is None:
        max_workers = min(len(seeds), 6) if seeds else 1
    max_workers = max(1, min(int(max_workers), len(seeds) if seeds else 1))
    tasks = [(seed, steps, config, report_interval) for seed in seeds]
    results = []
    started_at = time.perf_counter()
    if max_workers == 1:
        for task in tasks:
            result = run_ecosystem_seed_worker(task)
            results.append(result)
            if progress_callback is not None:
                progress_callback(
                    len(results), len(tasks), result,
                    time.perf_counter() - started_at,
                )
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_ecosystem_seed_worker, task)
                       for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if progress_callback is not None:
                    progress_callback(
                        len(results), len(tasks), result,
                        time.perf_counter() - started_at,
                    )
    results.sort(key=lambda result: result["seed"])
    aggregate_groups = combine_energy_groups(
        result["energy_groups"] for result in results
    )
    return results, aggregate_groups


def render_ecosystem_snapshot(ecosystem, axis, step):
    """Draw one top-down ecosystem frame on a Matplotlib axis."""
    region = ecosystem.config.region
    axis.clear()
    axis.set_xlim(region.x_min, region.x_max)
    axis.set_ylim(region.y_min, region.y_max)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(f"Ecosystem step {step}")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    if ecosystem.food:
        food_positions = np.asarray([food.position for food in ecosystem.food])
        axis.scatter(
            food_positions[:, 0], food_positions[:, 1],
            s=32, c="limegreen", marker=".", label="food",
        )
    if ecosystem.creatures:
        positions = np.asarray([creature.position for creature in ecosystem.creatures])
        energies = np.asarray([creature.energy for creature in ecosystem.creatures])
        radii = np.asarray([creature.genome.radius for creature in ecosystem.creatures])
        aggression = np.asarray([
            creature.genome.aggression for creature in ecosystem.creatures
        ])
        sizes = np.clip((radii / 0.02) ** 2 * 80.0, 20.0, 260.0)
        scatter = axis.scatter(
            positions[:, 0], positions[:, 1],
            s=sizes,
            c=aggression,
            cmap="plasma",
            vmin=0.0,
            vmax=1.0,
            edgecolors="black",
            linewidths=0.4,
            alpha=np.clip(energies / 140.0, 0.25, 1.0),
            label="creatures",
        )
        if not hasattr(axis, "_ecosystem_colorbar"):
            axis._ecosystem_colorbar = axis.figure.colorbar(
                scatter, ax=axis, fraction=0.046, pad=0.04,
                label="aggression",
            )
    summary = ecosystem.summary()
    axis.text(
        0.01, 0.99,
        (
            f"population {summary['population']}  "
            f"food eaten {summary['food_eaten']}  "
            f"prey eaten {summary['prey_eaten']}"
        ),
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    axis.legend(loc="lower right")


def show_ecosystem_window(
    steps=1000,
    seed=None,
    config=None,
    refresh_interval=5,
    pause_seconds=0.001,
):
    """Show a fixed-step top-down ecosystem visualization window."""
    import matplotlib.pyplot as plt

    ecosystem = Ecosystem(config=config, seed=seed)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for step in range(1, int(steps) + 1):
        ecosystem.step()
        if step == 1 or step % refresh_interval == 0 or step == int(steps):
            render_ecosystem_snapshot(ecosystem, axis, step)
            plt.pause(pause_seconds)
        if not plt.fignum_exists(figure.number) or not ecosystem.creatures:
            break
    render_ecosystem_snapshot(ecosystem, axis, min(step, int(steps)))
    print_ecosystem_summary(min(step, int(steps)), ecosystem.summary())
    plt.show()
    return ecosystem


def ecosystem_mujoco_xml(max_creatures, food_count):
    """Build a MuJoCo tank with preallocated ecosystem creature and food bodies."""
    creature_bodies = []
    for index in range(max_creatures):
        creature_bodies.append(
            f"""
    <body name="creature_{index}" pos="2 2 2">
      <freejoint name="creature_joint_{index}" />
      <geom name="creature_sphere_{index}" type="sphere" size="0.02"
        rgba="0.2 0.4 1.0 0.95" contype="0" conaffinity="0" />
      <geom name="creature_ellipsoid_{index}" type="ellipsoid" size="0.02 0.015 0.012"
        rgba="0.2 0.4 1.0 0" contype="0" conaffinity="0" />
      <geom name="creature_capsule_{index}" type="capsule" size="0.012 0.025"
        rgba="0.2 0.4 1.0 0" contype="0" conaffinity="0" />
    </body>"""
        )
    food_bodies = []
    for index in range(food_count):
        food_bodies.append(
            f"""
    <body name="food_{index}" pos="2 2 2">
      <geom name="food_geom_{index}" type="sphere" size="0.01"
        rgba="0.1 0.9 0.15 0.9" contype="0" conaffinity="0" />
    </body>"""
        )
    return f"""
<mujoco>
  <option timestep="0.02" integrator="RK4" />
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true" />
    <geom name="floor" type="box" pos="0.45 0 -0.005" size="0.67 0.32 0.005"
      material="hazy_floor" friction=".5 0.005 0.001"
      priority="2" condim="4" contype="0" conaffinity="0" />
    <body name="wall_3" pos="0.5 0.0 .05">
      <geom name="wall_3" type="box" size="0.1 0.15 0.05"
        mass=".1" material="static" condim="4" contype="0" conaffinity="0" />
    </body>
    <body name="case" pos="0 0 0.1">
      <geom name="gcase_a" type="box" size="0.05 0.32 0.1" pos="-.2 0 0"
        material="static" contype="0" conaffinity="0" />
      <geom name="gcase_b" type="box" size="0.05 0.32 0.1" pos="1.1 0 0"
        material="static" contype="0" conaffinity="0" />
      <geom name="gcase_c" type="box" size="0.7 0.04 0.1" pos="0.45 .36 0"
        material="static" contype="0" conaffinity="0" />
      <geom name="gcase_d" type="box" size="0.7 0.04 0.1" pos="0.45 -.36 0"
        material="static" contype="0" conaffinity="0" />
    </body>
    <body name="water" pos="0.45 0 0.075">
      <geom name="water_surface" type="box" size="0.67 0.32 0.075"
        rgba="0.58 0.72 0.82 0.22" contype="0" conaffinity="0" />
    </body>
    {"".join(creature_bodies)}
    {"".join(food_bodies)}
  </worldbody>
  <asset>
    <material name="MatPlane" rgba="0.8 0.9 0.8 1" />
    <material name="movable" rgba="0.94 1.0 0.0 1" />
    <material name="static" rgba="0.55 0.55 0.55 1" />
    <material name="hazy_floor" rgba="0.55 0.55 0.55 0.55" />
    <material name="BallMat" rgba="0.2 0.2 0.8 1" />
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7"
      rgb2="0 0 0" width="512" height="3072" />
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300" />
    <material name="groundplane" texture="groundplane" texuniform="true"
      texrepeat="5 5" reflectance="0.2" />
  </asset>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
    <rgba haze="0.15 0.25 0.35 1" />
    <global azimuth="60" elevation="-20" />
  </visual>
</mujoco>
"""


def _set_body_pose(model, data, joint_name, position, orientation=None):
    import mujoco

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_address = model.jnt_qposadr[joint_id]
    qvel_address = model.jnt_dofadr[joint_id]
    data.qpos[qpos_address:qpos_address + 3] = position
    data.qpos[qpos_address + 3:qpos_address + 7] = (
        [1.0, 0.0, 0.0, 0.0] if orientation is None else orientation
    )
    data.qvel[qvel_address:qvel_address + 6] = 0.0


def sync_ecosystem_to_mujoco(ecosystem, model, data):
    """Copy ecosystem creature and food state into a preallocated MuJoCo model."""
    import mujoco

    hidden_position = np.array([2.0, 2.0, 2.0])
    shape_names = ("sphere", "ellipsoid", "capsule")
    for index in range(ecosystem.config.max_population):
        geom_ids = {
            shape_name: mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM,
                f"creature_{shape_name}_{index}",
            )
            for shape_name in shape_names
        }
        if index < len(ecosystem.creatures):
            creature = ecosystem.creatures[index]
            _set_body_pose(
                model, data, f"creature_joint_{index}", creature.position,
                creature.genome.orientation,
            )
            color = [
                0.2 + 0.8 * creature.genome.aggression,
                0.15,
                1.0 - 0.7 * creature.genome.aggression,
                np.clip(creature.energy / 120.0, 0.25, 1.0),
            ]
            for shape_name, geom_id in geom_ids.items():
                model.geom_rgba[geom_id, 3] = 0.0
            shape_type = (
                creature.genome.shape_type
                if creature.genome.shape_type in geom_ids else "sphere"
            )
            geom_id = geom_ids[shape_type]
            size = np.asarray(creature.genome.size, dtype=np.float64)
            if shape_type == "sphere":
                model.geom_size[geom_id, 0] = creature.genome.radius
            elif shape_type == "ellipsoid":
                model.geom_size[geom_id, :3] = size
            elif shape_type == "capsule":
                model.geom_size[geom_id, 0] = max(min(size[1], size[2]), 0.004)
                model.geom_size[geom_id, 1] = max(size[0], 0.006)
            model.geom_rgba[geom_id] = color
        else:
            _set_body_pose(model, data, f"creature_joint_{index}", hidden_position)
            for geom_id in geom_ids.values():
                model.geom_rgba[geom_id, 3] = 0.0
    for index in range(ecosystem.config.food_count):
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"food_{index}"
        )
        geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"food_geom_{index}"
        )
        if index < len(ecosystem.food):
            model.body_pos[body_id] = ecosystem.food[index].position
            model.geom_rgba[geom_id, 3] = 0.9
        else:
            model.body_pos[body_id] = hidden_position
            model.geom_rgba[geom_id, 3] = 0.0
    mujoco.mj_forward(model, data)


def show_ecosystem_mujoco_window(
    steps=1000,
    seed=None,
    config=None,
    render_every=1,
    realtime=True,
):
    """Render the ecosystem in a MuJoCo tank backed by Navier-Stokes flow."""
    import glfw
    import mujoco
    import time

    config = config or EcosystemConfig(use_navier_stokes=True)
    if not config.use_navier_stokes:
        config = EcosystemConfig(**{**config.__dict__, "use_navier_stokes": True})
    ecosystem = Ecosystem(config=config, seed=seed)
    model = mujoco.MjModel.from_xml_string(
        ecosystem_mujoco_xml(config.max_population, config.food_count)
    )
    data = mujoco.MjData(model)

    if not glfw.init():
        raise RuntimeError("Could not initialize GLFW")
    window = glfw.create_window(1000, 700, "MuJoCo Ecosystem Tank", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create GLFW window")
    glfw.make_context_current(window)
    camera = mujoco.MjvCamera()
    camera.azimuth = 0.0
    camera.elevation = -90.0
    camera.distance = 2.0
    camera.lookat = np.array([0.5, 0.0, 0.0])
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    viewport = mujoco.MjrRect(0, 0, 1000, 700)
    option = mujoco.MjvOption()
    mouse = {
        "left": False,
        "middle": False,
        "right": False,
        "last_x": 0.0,
        "last_y": 0.0,
    }

    def mouse_button_callback(window, button, action, mods):
        pressed = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_LEFT:
            mouse["left"] = pressed
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            mouse["middle"] = pressed
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            mouse["right"] = pressed
        mouse["last_x"], mouse["last_y"] = glfw.get_cursor_pos(window)

    def cursor_position_callback(window, x_position, y_position):
        if not (mouse["left"] or mouse["middle"] or mouse["right"]):
            return
        width, height = glfw.get_window_size(window)
        delta_x = x_position - mouse["last_x"]
        delta_y = y_position - mouse["last_y"]
        mouse["last_x"], mouse["last_y"] = x_position, y_position
        shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if mouse["right"]:
            camera_action = mujoco.mjtMouse.mjMOUSE_MOVE_H
        elif mouse["middle"]:
            camera_action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif shift:
            camera_action = mujoco.mjtMouse.mjMOUSE_ROTATE_H
        else:
            camera_action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        mujoco.mjv_moveCamera(
            model,
            camera_action,
            delta_x / max(width, 1),
            delta_y / max(height, 1),
            scene,
            camera,
        )

    def scroll_callback(window, x_offset, y_offset):
        mujoco.mjv_moveCamera(
            model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * y_offset,
            scene,
            camera,
        )

    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    try:
        next_frame_time = time.perf_counter()
        for step in range(1, int(steps) + 1):
            if glfw.window_should_close(window) or not ecosystem.creatures:
                break
            if realtime:
                now = time.perf_counter()
                if now < next_frame_time:
                    time.sleep(next_frame_time - now)
                next_frame_time = max(
                    next_frame_time + ecosystem.config.dt,
                    time.perf_counter(),
                )
            ecosystem.step()
            if step == 1 or step % render_every == 0 or step == int(steps):
                width, height = glfw.get_framebuffer_size(window)
                viewport.width = width
                viewport.height = height
                sync_ecosystem_to_mujoco(ecosystem, model, data)
                mujoco.mjv_updateScene(
                    model, data, option, None, camera,
                    mujoco.mjtCatBit.mjCAT_ALL.value, scene,
                )
                mujoco.mjr_render(viewport, scene, context)
                glfw.swap_buffers(window)
                glfw.poll_events()
        print_ecosystem_summary(min(step, int(steps)), ecosystem.summary())
        while not glfw.window_should_close(window):
            glfw.poll_events()
            width, height = glfw.get_framebuffer_size(window)
            viewport.width = width
            viewport.height = height
            mujoco.mjv_updateScene(
                model, data, option, None, camera,
                mujoco.mjtCatBit.mjCAT_ALL.value, scene,
            )
            mujoco.mjr_render(viewport, scene, context)
            glfw.swap_buffers(window)
    finally:
        glfw.terminate()
    return ecosystem
