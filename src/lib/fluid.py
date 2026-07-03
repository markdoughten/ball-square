"""Three-dimensional incompressible flow, drag, and buoyancy."""

from dataclasses import dataclass, field as dataclass_field

import mujoco
import numpy as np


@dataclass(frozen=True)
class FluidRegion:
    x_min: float = -0.15
    x_max: float = 1.05
    y_min: float = -0.32
    y_max: float = 0.32
    bottom_z: float = 0.0
    surface_z: float = 0.15

    def submerged_fraction(self, position, radius):
        x, y = position[:2]
        x_fraction = np.clip((x+radius-self.x_min)/(2*radius), 0, 1)
        x_fraction *= np.clip((self.x_max-x+radius)/(2*radius), 0, 1)
        y_fraction = np.clip((y+radius-self.y_min)/(2*radius), 0, 1)
        y_fraction *= np.clip((self.y_max-y+radius)/(2*radius), 0, 1)
        return float(x_fraction*y_fraction)


class NavierStokesField:
    """Volumetric incompressible Navier-Stokes solver on a regular grid."""

    def __init__(self, region, inlet_velocity, nx=32, ny=18, nz=10,
                 kinematic_viscosity=8.93e-7, pressure_iterations=16,
                 turbulence_intensity=0.04, vorticity_confinement=0.03):
        self.region = region
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx = (region.x_max-region.x_min)/(nx-1)
        self.dy = (region.y_max-region.y_min)/(ny-1)
        self.dz = (region.surface_z-region.bottom_z)/(nz-1)
        inlet = np.asarray(inlet_velocity, dtype=np.float64)
        self.inlet_velocity = np.pad(inlet, (0, max(0, 3-len(inlet))))[:3]
        self.kinematic_viscosity = kinematic_viscosity
        self.pressure_iterations = pressure_iterations
        self.turbulence_intensity = turbulence_intensity
        self.vorticity_confinement = vorticity_confinement
        self.time = 0.0
        self.velocity = np.zeros((nz, ny, nx, 3), dtype=np.float64)
        self.velocity[:] = self.inlet_velocity
        self.pressure = np.zeros((nz, ny, nx), dtype=np.float64)
        xs = np.linspace(region.x_min, region.x_max, nx)
        ys = np.linspace(region.y_min, region.y_max, ny)
        zs = np.linspace(region.bottom_z, region.surface_z, nz)
        self.z, self.y, self.x = np.meshgrid(zs, ys, xs, indexing="ij")
        self.solid = ((self.x >= 0.4) & (self.x <= 0.6)
                      & (self.y >= -0.15) & (self.y <= 0.15)
                      & (self.z <= 0.10))
        self.moving_solid = np.zeros_like(self.solid)
        self.moving_velocity = np.zeros(3, dtype=np.float64)
        self._apply_boundaries(self.velocity)

    @property
    def all_solid(self):
        return self.solid | self.moving_solid

    def set_moving_sphere(self, position, radius, velocity):
        """Rasterize a moving no-slip sphere into the Eulerian fluid grid."""
        position = np.asarray(position, dtype=np.float64)
        distance_squared = ((self.x-position[0])**2 + (self.y-position[1])**2
                            + (self.z-position[2])**2)
        self.moving_solid = (distance_squared <= radius**2) & ~self.solid
        self.moving_velocity = np.asarray(velocity, dtype=np.float64).copy()

    def _sample(self, values, x, y, z):
        gx = np.clip((x-self.region.x_min)/self.dx, 0, self.nx-1)
        gy = np.clip((y-self.region.y_min)/self.dy, 0, self.ny-1)
        gz = np.clip((z-self.region.bottom_z)/self.dz, 0, self.nz-1)
        x0, y0, z0 = np.floor(gx).astype(int), np.floor(gy).astype(int), np.floor(gz).astype(int)
        x1, y1, z1 = np.minimum(x0+1, self.nx-1), np.minimum(y0+1, self.ny-1), np.minimum(z0+1, self.nz-1)
        wx, wy, wz = gx-x0, gy-y0, gz-z0
        if values.ndim == 4:
            wx, wy, wz = wx[..., None], wy[..., None], wz[..., None]
        c00 = (1-wx)*values[z0, y0, x0] + wx*values[z0, y0, x1]
        c01 = (1-wx)*values[z0, y1, x0] + wx*values[z0, y1, x1]
        c10 = (1-wx)*values[z1, y0, x0] + wx*values[z1, y0, x1]
        c11 = (1-wx)*values[z1, y1, x0] + wx*values[z1, y1, x1]
        return (1-wz)*((1-wy)*c00+wy*c01) + wz*((1-wy)*c10+wy*c11)

    def _apply_boundaries(self, velocity):
        velocity[:, :, 0] = velocity[:, :, -1] = 0.0
        velocity[:, 0] = velocity[:, -1] = 0.0
        velocity[0] = 0.0
        velocity[-1] = velocity[-2]  # free surface: zero normal gradient
        velocity[-1, ..., 2] = 0.0
        u, v = self.inlet_velocity[:2]
        if abs(u) >= abs(v):
            if u < 0:
                velocity[:, :, -1] = self.inlet_velocity
                velocity[:, :, 0] = velocity[:, :, 1]
            else:
                velocity[:, :, 0] = self.inlet_velocity
                velocity[:, :, -1] = velocity[:, :, -2]
        elif v < 0:
            velocity[:, -1] = self.inlet_velocity
            velocity[:, 0] = velocity[:, 1]
        else:
            velocity[:, 0] = self.inlet_velocity
            velocity[:, -1] = velocity[:, -2]
        velocity[0] = 0.0
        velocity[-1, ..., 2] = 0.0
        velocity[self.solid] = 0.0
        velocity[self.moving_solid] = self.moving_velocity

    def vorticity(self, velocity=None):
        velocity = self.velocity if velocity is None else velocity
        curl = np.zeros_like(velocity)
        core = np.s_[1:-1, 1:-1, 1:-1]
        dw_dy = (velocity[1:-1, 2:, 1:-1, 2]-velocity[1:-1, :-2, 1:-1, 2])/(2*self.dy)
        dv_dz = (velocity[2:, 1:-1, 1:-1, 1]-velocity[:-2, 1:-1, 1:-1, 1])/(2*self.dz)
        du_dz = (velocity[2:, 1:-1, 1:-1, 0]-velocity[:-2, 1:-1, 1:-1, 0])/(2*self.dz)
        dw_dx = (velocity[1:-1, 1:-1, 2:, 2]-velocity[1:-1, 1:-1, :-2, 2])/(2*self.dx)
        dv_dx = (velocity[1:-1, 1:-1, 2:, 1]-velocity[1:-1, 1:-1, :-2, 1])/(2*self.dx)
        du_dy = (velocity[1:-1, 2:, 1:-1, 0]-velocity[1:-1, :-2, 1:-1, 0])/(2*self.dy)
        curl[core] = np.stack((dw_dy-dv_dz, du_dz-dw_dx, dv_dx-du_dy), axis=-1)
        return curl

    def step(self, dt, density=997.0):
        """Advance du/dt+(u.grad)u=-grad(p)/rho+nu*laplacian(u), div(u)=0."""
        dt = min(float(dt), 0.05)
        old = self.velocity
        velocity = self._sample(old, self.x-dt*old[..., 0],
                                self.y-dt*old[..., 1], self.z-dt*old[..., 2])
        advected = velocity.copy()
        ax, ay, az = (self.kinematic_viscosity*dt/self.dx**2,
                      self.kinematic_viscosity*dt/self.dy**2,
                      self.kinematic_viscosity*dt/self.dz**2)
        for _ in range(6):
            velocity[1:-1, 1:-1, 1:-1] = (advected[1:-1, 1:-1, 1:-1]
                + ax*(velocity[1:-1, 1:-1, 2:]+velocity[1:-1, 1:-1, :-2])
                + ay*(velocity[1:-1, 2:, 1:-1]+velocity[1:-1, :-2, 1:-1])
                + az*(velocity[2:, 1:-1, 1:-1]+velocity[:-2, 1:-1, 1:-1]))/(1+2*ax+2*ay+2*az)
            self._apply_boundaries(velocity)

        self.time += dt
        lx, ly, lz = (self.region.x_max-self.region.x_min,
                      self.region.y_max-self.region.y_min,
                      self.region.surface_z-self.region.bottom_z)
        eddies = np.empty_like(velocity)
        # Each component is independent of its own axis, so this forcing is divergence-free.
        eddies[..., 0] = np.sin(2*np.pi*(self.y-self.region.y_min)/ly+0.7*self.time)*np.cos(np.pi*self.z/lz)
        eddies[..., 1] = np.sin(2*np.pi*self.z/lz+0.8*self.time)*np.cos(2*np.pi*(self.x-self.region.x_min)/lx)
        eddies[..., 2] = np.sin(2*np.pi*(self.x-self.region.x_min)/lx-0.9*self.time)*np.cos(2*np.pi*(self.y-self.region.y_min)/ly)
        velocity += dt*self.turbulence_intensity*eddies

        omega = self.vorticity(velocity)
        magnitude = np.linalg.norm(omega, axis=-1)
        gz, gy, gx = np.gradient(magnitude, self.dz, self.dy, self.dx)
        normal = np.stack((gx, gy, gz), axis=-1)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)+1e-12
        velocity += dt*self.vorticity_confinement*np.cross(normal, omega)
        self._apply_boundaries(velocity)

        divergence = np.zeros_like(self.pressure)
        divergence[1:-1, 1:-1, 1:-1] = (
            (velocity[1:-1, 1:-1, 2:, 0]-velocity[1:-1, 1:-1, :-2, 0])/(2*self.dx)
            + (velocity[1:-1, 2:, 1:-1, 1]-velocity[1:-1, :-2, 1:-1, 1])/(2*self.dy)
            + (velocity[2:, 1:-1, 1:-1, 2]-velocity[:-2, 1:-1, 1:-1, 2])/(2*self.dz))
        divergence[self.all_solid] = 0.0
        pressure = self.pressure.copy()
        denominator = 2/self.dx**2+2/self.dy**2+2/self.dz**2
        for _ in range(self.pressure_iterations):
            pressure[1:-1, 1:-1, 1:-1] = (
                (pressure[1:-1, 1:-1, 2:]+pressure[1:-1, 1:-1, :-2])/self.dx**2
                + (pressure[1:-1, 2:, 1:-1]+pressure[1:-1, :-2, 1:-1])/self.dy**2
                + (pressure[2:, 1:-1, 1:-1]+pressure[:-2, 1:-1, 1:-1])/self.dz**2
                - density*divergence[1:-1, 1:-1, 1:-1]/dt)/denominator
            pressure[:, :, 0], pressure[:, :, -1] = pressure[:, :, 1], pressure[:, :, -2]
            pressure[:, 0], pressure[:, -1] = pressure[:, 1], pressure[:, -2]
            pressure[0], pressure[-1] = pressure[1], pressure[-2]
            pressure[self.all_solid] = 0.0
        velocity[1:-1, 1:-1, 1:-1, 0] -= dt/density*(pressure[1:-1, 1:-1, 2:]-pressure[1:-1, 1:-1, :-2])/(2*self.dx)
        velocity[1:-1, 1:-1, 1:-1, 1] -= dt/density*(pressure[1:-1, 2:, 1:-1]-pressure[1:-1, :-2, 1:-1])/(2*self.dy)
        velocity[1:-1, 1:-1, 1:-1, 2] -= dt/density*(pressure[2:, 1:-1, 1:-1]-pressure[:-2, 1:-1, 1:-1])/(2*self.dz)
        self._apply_boundaries(velocity)
        self.velocity, self.pressure = velocity, pressure

    def velocity_at(self, position):
        return np.asarray(self._sample(self.velocity, *np.asarray(position)[:3]))

    def velocity_around_sphere(self, position, radius):
        """Sample ambient flow outside the immersed boundary."""
        position = np.asarray(position, dtype=np.float64)
        offset = max(1.75*radius, 1.1*min(self.dx, self.dy, self.dz))
        directions = np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0],
            [0, -1, 0], [0, 0, 1], [0, 0, -1],
        ], dtype=np.float64)
        samples = [self.velocity_at(position+offset*direction)
                   for direction in directions]
        return np.mean(samples, axis=0)

    def add_reaction_force(self, position, radius, reaction_force, dt, density):
        """Return the body's hydrodynamic impulse to nearby fluid cells."""
        position = np.asarray(position, dtype=np.float64)
        distance = np.sqrt((self.x-position[0])**2 + (self.y-position[1])**2
                           + (self.z-position[2])**2)
        shell = ((distance > radius) & (distance <= 4*radius)
                 & ~self.all_solid)
        if not np.any(shell):
            return
        weights = np.exp(-((distance[shell]-radius)/(1.5*radius))**2)
        weights /= np.sum(weights)
        cell_mass = density*self.dx*self.dy*self.dz
        impulse_velocity = np.asarray(reaction_force)*dt/cell_mass
        self.velocity[shell] += weights[:, None]*impulse_velocity
        self._apply_boundaries(self.velocity)


@dataclass
class FluidDynamics:
    region: FluidRegion = dataclass_field(default_factory=FluidRegion)
    density: float = 997.0
    dynamic_viscosity: float = 0.00089
    drag_coefficient: float = 0.47
    added_mass_coefficient: float = 0.5
    neutral_buoyancy: bool = True
    surface_equilibrium_fraction: float = 0.5
    turn_rate: float = np.deg2rad(60.0)
    current_velocity: np.ndarray = dataclass_field(default_factory=lambda: np.array([-0.025, 0.0, 0.0]))
    _previous_velocity: np.ndarray | None = dataclass_field(default=None, init=False)
    field: NavierStokesField = dataclass_field(init=False)
    target_current_velocity: np.ndarray = dataclass_field(init=False)

    def __post_init__(self):
        current = np.asarray(self.current_velocity, dtype=np.float64)
        self.current_velocity = np.pad(current, (0, max(0, 3-len(current))))[:3]
        self.target_current_velocity = self.current_velocity.copy()
        self.field = NavierStokesField(self.region, self.current_velocity)

    def reset(self):
        self._previous_velocity = None
        self.field = NavierStokesField(self.region, self.current_velocity)

    def set_current_direction(self, direction):
        direction = np.asarray(direction, dtype=np.float64)
        direction = np.pad(direction, (0, max(0, 3-len(direction))))[:3]
        if np.linalg.norm(direction) == 0:
            raise ValueError("Current direction cannot be zero")
        self.target_current_velocity = direction/np.linalg.norm(direction)*np.linalg.norm(self.current_velocity)

    def advance_current_direction(self, dt):
        current_angle = np.arctan2(self.current_velocity[1], self.current_velocity[0])
        target_angle = np.arctan2(self.target_current_velocity[1], self.target_current_velocity[0])
        error = (target_angle-current_angle+np.pi)%(2*np.pi)-np.pi
        angle = current_angle+np.clip(error, -self.turn_rate*dt, self.turn_rate*dt)
        speed = np.linalg.norm(self.current_velocity)
        self.current_velocity = speed*np.array([np.cos(angle), np.sin(angle), 0.0])
        self.field.inlet_velocity = self.current_velocity.copy()

    def force(self, position, velocity, radius, dt, flow_velocity=None):
        fraction = self.region.submerged_fraction(position, radius)
        velocity = np.asarray(velocity, dtype=np.float64)
        if fraction == 0:
            self._previous_velocity = velocity.copy()
            return np.zeros_like(velocity)
        flow = self.current_velocity[:len(velocity)] if flow_velocity is None else np.asarray(flow_velocity)[:len(velocity)]
        relative = velocity-flow
        drag = (-0.5*self.density*self.drag_coefficient*np.pi*radius**2
                * np.linalg.norm(relative)*relative
                - 6*np.pi*self.dynamic_viscosity*radius*relative)
        added = np.zeros_like(velocity)
        if self._previous_velocity is not None and dt > 0:
            mass = self.added_mass_coefficient*self.density*4/3*np.pi*radius**3
            added = -mass*(velocity-self._previous_velocity)/dt
        self._previous_velocity = velocity.copy()
        return fraction*(drag+added)

    def submerged_volume(self, center_z, radius):
        height = np.clip(self.region.surface_z-(center_z-radius), 0, 2*radius)
        return float(np.pi*height**2*(radius-height/3))

    def buoyancy_force(self, body_mass, center_z, radius, fraction, gravity):
        volume = fraction*self.submerged_volume(center_z, radius)
        if self.neutral_buoyancy:
            ratio = np.clip(volume/(4/3*np.pi*radius**3), 0, 1)
            return float(body_mass*gravity*ratio/self.surface_equilibrium_fraction)
        return float(self.density*gravity*volume)

    def apply(self, model, data, body_id, radius):
        position = data.xpos[body_id, :3]
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                  for name in ("joint_x", "joint_y", "joint_z")]
        velocity = np.array([data.qvel[model.jnt_dofadr[joint]] for joint in joints])
        self.advance_current_direction(model.opt.timestep)
        self.field.set_moving_sphere(position, radius, velocity)
        self.field.step(model.opt.timestep, self.density)
        submerged_ratio = self.submerged_volume(position[2], radius)/(4/3*np.pi*radius**3)
        hydrodynamic_force = self.force(
            position, velocity, radius, model.opt.timestep,
            self.field.velocity_around_sphere(position, radius))*submerged_ratio
        data.xfrc_applied[body_id, :3] += hydrodynamic_force
        self.field.add_reaction_force(
            position, radius, -hydrodynamic_force, model.opt.timestep,
            self.density,
        )
        fraction = self.region.submerged_fraction(position, radius)
        data.xfrc_applied[body_id, 2] += self.buoyancy_force(
            model.body_mass[body_id], position[2], radius, fraction,
            abs(float(model.opt.gravity[2])))
