import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import mujoco
import glfw
import os

PATH_CLEARANCE = 0.06
POSITION_DIMENSIONS = 3
MOTOR_FORCE_SCALE = np.array([15.0, 15.0, 80.0])
CONTROL_TIMESTEP = 0.02
DEFAULT_CAMERA_VIEW = {
    "azimuth": 0.0,
    "elevation": -90.0,
    "distance": 2.0,
    "lookat": np.array([0.5, 0.0, 0.0], dtype=np.float64),
}


def format_position(position, decimals=3):
    return "(" + ", ".join(
        f"{float(value):.{decimals}f}" for value in position
    ) + ")"


def simulation_state(model, data, body_id):
    """Return world position and XYZ joint velocity."""
    return np.hstack((data.xpos[body_id, :3], data.qvel[:3])).copy()


def set_simulation_state(model, data, state, body_id):
    """Set a world-space `[x,y,z,vx,vy,vz]` state in MuJoCo."""
    state = np.asarray(state)
    data.qpos[:2] = state[:2]
    data.qpos[2] = state[2] - model.body_pos[body_id, 2]
    data.qvel[:3] = state[3:6]
    mujoco.mj_forward(model, data)


def apply_training_fluid_forces(model, data, body_id, radius=0.02):
    """Low-cost hydrodynamics used by parallel PPO environments."""
    center_z = float(data.xpos[body_id, 2])
    surface_z = 0.15
    cap_height = np.clip(surface_z-(center_z-radius), 0.0, 2.0*radius)
    submerged_volume = np.pi*cap_height**2*(radius-cap_height/3.0)
    sphere_volume = 4.0/3.0*np.pi*radius**3
    submerged_ratio = submerged_volume/sphere_volume
    gravity = abs(float(model.opt.gravity[2]))
    # Same half-submerged equilibrium used by FluidDynamics.
    data.xfrc_applied[body_id, 2] += (
        model.body_mass[body_id]*gravity*submerged_ratio/0.5
    )
    relative_velocity = data.qvel[:3]-np.array([-0.025, 0.0, 0.0])
    drag = (
        -0.5*997.0*0.47*np.pi*radius**2
        * np.linalg.norm(relative_velocity)*relative_velocity
        - 6.0*np.pi*0.00089*radius*relative_velocity
    )
    data.xfrc_applied[body_id, :3] += submerged_ratio*drag


def step_training_simulation(model, data, body_id, action, fluid=None):
    """Apply one control action with stable hydrodynamic physics substeps."""
    control_timestep = float(model.opt.timestep)
    substeps = max(1, int(round(control_timestep/0.01)))
    model.opt.timestep = control_timestep/substeps
    data.ctrl[:3] = action
    for _ in range(substeps):
        data.xfrc_applied[body_id] = 0.0
        if fluid is None:
            apply_training_fluid_forces(model, data, body_id)
        else:
            fluid.apply(model, data, body_id, radius=0.02)
        mujoco.mj_step(model, data)
    model.opt.timestep = control_timestep


class ActorNetwork(nn.Module):
    
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc_mu = nn.Linear(hidden_dim3, output_dim)
        self.fc_sigma = nn.Linear(hidden_dim3, output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc_mu, self.fc_sigma]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        mu = self.fc_mu(x)
        sigma = torch.clamp(
            F.softplus(self.fc_sigma(x)), min=0.05, max=0.5
        )
        return mu, sigma

class CriticNetwork(nn.Module):
    
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc_value = nn.Linear(hidden_dim3, 1)
        self.initialize_weights()
    
    def initialize_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc_value]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        value = self.fc_value(x)
        return value

class ActorCriticModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, action_dim):
        super(ActorCriticModel, self).__init__()
        self.actor = ActorNetwork(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, action_dim)
        self.critic = CriticNetwork(input_dim, hidden_dim1, hidden_dim2, hidden_dim3)

    def forward(self, state):
        mu, sigma = self.actor(state)
        value = self.critic(state)
        return mu, sigma, value


def apply_camera_view(camera, view):
    camera.azimuth = float(view["azimuth"])
    camera.elevation = float(view["elevation"])
    camera.distance = float(view["distance"])
    camera.lookat = np.asarray(view["lookat"], dtype=np.float64).copy()


def remember_camera_view(camera):
    DEFAULT_CAMERA_VIEW["azimuth"] = float(camera.azimuth)
    DEFAULT_CAMERA_VIEW["elevation"] = float(camera.elevation)
    DEFAULT_CAMERA_VIEW["distance"] = float(camera.distance)
    DEFAULT_CAMERA_VIEW["lookat"] = np.asarray(camera.lookat).copy()


def init_mujoco_render(model):
    if not glfw.init():
        raise Exception("Could not initialize GLFW")

    window = glfw.create_window(800, 600, "MuJoCo Simulation", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Could not create GLFW window")

    glfw.make_context_current(window)
    camera = mujoco.MjvCamera()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    viewport = mujoco.MjrRect(0, 0, 800, 600)
    option = mujoco.MjvOption()

    apply_camera_view(camera, DEFAULT_CAMERA_VIEW)

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
        remember_camera_view(camera)

    def scroll_callback(window, x_offset, y_offset):
        mujoco.mjv_moveCamera(
            model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * y_offset,
            scene,
            camera,
        )
        remember_camera_view(camera)

    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    return window, camera, scene, context, viewport, option

def render_mujoco_scene(model, data, scene, context, viewport, camera, window, option):
    mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    mujoco.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()
    return

def render_trained_policy(
    model,
    goal_position,
    walls,
    outside_walls,
    mujoco_model,
    mujoco_data,
    epsilon=0.1,
    max_steps=1800,
    spawn_margin=0.025,
    fixed_start=False,
    initial_state=None,
    interactive=False,
    destination_positions=None,
    device=torch.device("cpu"),
):
    """Render the deterministic policy without changing model weights."""
    control_timestep = float(mujoco_model.opt.timestep)
    physics_substeps = max(1, int(round(control_timestep / 0.01)))
    mujoco_model.opt.timestep = control_timestep / physics_substeps
    navigation_clearance = spawn_margin
    if initial_state is not None:
        state = np.asarray(initial_state, dtype=np.float64).copy()
    elif fixed_start:
        state = np.array([0.0, 0.0, outside_walls["z_max"], 0.0, 0.0, 0.0])
    else:
        state = random_position(
            walls, outside_walls, goal_position,
            minimum_goal_distance=epsilon * 2,
            safety_margin=max(spawn_margin, PATH_CLEARANCE),
        )
    ball_body_id = mujoco.mj_name2id(
        mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "ball"
    )
    set_simulation_state(mujoco_model, mujoco_data, state, ball_body_id)
    navigation_goal = np.asarray(goal_position, dtype=np.float64)
    if len(navigation_goal) == 2:
        navigation_goal = np.append(navigation_goal, outside_walls["z_max"])
    start_position = mujoco_data.xpos[ball_body_id, :3].copy()
    path = plan_navigation_path(
        start_position, navigation_goal, walls, outside_walls,
        clearance=navigation_clearance,
    )
    paths = [path if path is not None else [start_position, navigation_goal]]
    waypoint_indices = np.ones(1, dtype=np.int32)
    print(f"Following RRT path with {len(paths[0])} waypoints.")
    window, camera, scene, context, viewport, option = init_mujoco_render(
        mujoco_model
    )
    reached_goal = False
    active_goal = navigation_goal.copy()
    destination_positions = destination_positions or {
        "start": np.array([0.0, 0.0, outside_walls["z_max"]]),
        "goal": np.array([0.9, -0.22, 0.06]),
        "underwater": np.array([0.9, -0.22, 0.06]),
    }
    from .fluid import FluidDynamics
    fluid = FluidDynamics()
    target_site_id = mujoco.mj_name2id(
        mujoco_model, mujoco.mjtObj.mjOBJ_SITE, "underwater_target"
    )
    if target_site_id >= 0:
        mujoco_model.site_pos[target_site_id] = active_goal
        mujoco.mj_forward(mujoco_model, mujoco_data)

    last_goal_move_report = 0.0

    def set_active_goal(goal, label="goal", announce=True):
        nonlocal active_goal, paths, reached_goal, last_goal_move_report
        candidate = np.asarray(goal, dtype=np.float64).copy()
        candidate[0] = np.clip(
            candidate[0],
            outside_walls["x_min"] + navigation_clearance,
            outside_walls["x_max"] - navigation_clearance,
        )
        candidate[1] = np.clip(
            candidate[1],
            outside_walls["y_min"] + navigation_clearance,
            outside_walls["y_max"] - navigation_clearance,
        )
        candidate[2] = np.clip(
            candidate[2],
            outside_walls["z_min"],
            outside_walls["z_max"],
        )
        if not path_point_is_free(
            candidate, walls, outside_walls, clearance=navigation_clearance
        ):
            if announce:
                print(f"\nGoal move blocked at {format_position(candidate)}.")
            return False
        physical_position = mujoco_data.xpos[ball_body_id, :3].copy()
        new_path = plan_navigation_path(
            physical_position, candidate, walls, outside_walls,
            clearance=navigation_clearance,
        )
        if new_path is None:
            if announce:
                print(f"\nNo route to {format_position(candidate)}.")
            return False
        active_goal = candidate
        paths = [new_path]
        waypoint_indices[:] = 1
        reached_goal = False
        if target_site_id >= 0:
            mujoco_model.site_pos[target_site_id] = active_goal
            mujoco.mj_forward(mujoco_model, mujoco_data)
        if announce:
            print(
                f"\n{label} set to {format_position(active_goal)} "
                f"({len(paths[0])} waypoints)."
            )
            last_goal_move_report = step if "step" in locals() else 0.0
        return True

    goal_key_directions = {
        glfw.KEY_W: np.array([1.0, 0.0, 0.0]),
        glfw.KEY_S: np.array([-1.0, 0.0, 0.0]),
        glfw.KEY_A: np.array([0.0, 1.0, 0.0]),
        glfw.KEY_D: np.array([0.0, -1.0, 0.0]),
        glfw.KEY_UP: np.array([0.0, 0.0, 1.0]),
        glfw.KEY_DOWN: np.array([0.0, 0.0, -1.0]),
    }
    ball_geom_id = mujoco.mj_name2id(
        mujoco_model, mujoco.mjtObj.mjOBJ_GEOM, "gball_0"
    )
    ball_radius = float(mujoco_model.geom_size[ball_geom_id, 0])
    if interactive:
        from .commands import parse_destination, poll_console_line
        simulation_options = {
            "1": "start",
            "2": "underwater",
            "5": "quit",
        }
        print("Simulation controls: 1=start, 2=underwater goal, 5=quit.")
        print("Hold W/S along the long side, A/D side to side; arrow Up/Down change depth.")
        print("Command: ", end="", flush=True)
    model.eval()
    with torch.no_grad():
        step = 0
        while interactive or step < max_steps:
            if glfw.window_should_close(window):
                break
            step += 1
            if interactive:
                goal_direction = np.zeros(3, dtype=np.float64)
                for key, direction in goal_key_directions.items():
                    if glfw.get_key(window, key) == glfw.PRESS:
                        goal_direction += direction
                if np.any(goal_direction):
                    goal_direction /= np.linalg.norm(goal_direction)
                    moved = set_active_goal(
                        active_goal + goal_direction * 0.18 * control_timestep,
                        "Goal",
                        announce=False,
                    )
                    if moved and step - last_goal_move_report >= 30:
                        print(f"\nGoal at {format_position(active_goal)}.")
                        print("Command: ", end="", flush=True)
                        last_goal_move_report = step
            if interactive:
                command = poll_console_line()
                if command is not None:
                    normalized_command = command.lower()
                    selected_command = simulation_options.get(command)
                    if selected_command == "quit" or normalized_command in {"quit", "exit"}:
                        break
                    destination = (
                        selected_command
                        if selected_command in destination_positions
                        else parse_destination(command)
                    )
                    if destination is None:
                        print("Unknown command; use 1=start, 2=underwater goal, or 5=quit.")
                    else:
                        new_goal = np.asarray(
                            destination_positions[destination], dtype=np.float64
                        )
                        if len(new_goal) == 2:
                            new_goal = np.append(new_goal, outside_walls["z_max"])
                        set_active_goal(new_goal, f"Redirecting to {destination}")
                    print("Command: ", end="", flush=True)
            physical_state = simulation_state(
                mujoco_model, mujoco_data, ball_body_id
            )
            targets_3d = current_path_targets(
                physical_state[None, :], paths, waypoint_indices
            )
            targets = targets_3d
            state_tensor = torch.as_tensor(
                observations_for_targets(state, targets)[0],
                dtype=torch.float32, device=device
            )
            mu, _, _ = model(state_tensor)
            action = safety_shield_actions(
                state, targets, torch.tanh(mu).cpu().numpy(),
                walls, outside_walls,
            )[0]
            for _ in range(physics_substeps):
                # Applied forces persist in MuJoCo until explicitly reset.
                mujoco_data.xfrc_applied[ball_body_id] = 0.0
                mujoco_data.ctrl[:3] = action
                fluid.apply(mujoco_model, mujoco_data, ball_body_id, ball_radius)
                mujoco.mj_step(mujoco_model, mujoco_data)
            state = simulation_state(mujoco_model, mujoco_data, ball_body_id)
            render_mujoco_scene(
                mujoco_model, mujoco_data, scene, context, viewport,
                camera, window, option,
            )
            goal_tolerance = (
                min(epsilon, 0.01)
                if active_goal[2] < outside_walls["z_max"]-0.005
                else epsilon
            )
            if (
                not reached_goal
                and np.linalg.norm(
                    mujoco_data.xpos[ball_body_id, :3] - active_goal
                ) <= goal_tolerance
            ):
                reached_goal = True
                goal_distance = np.linalg.norm(
                    mujoco_data.xpos[ball_body_id, :3] - active_goal
                )
                print(
                    f"Goal reached in {step} steps at "
                    f"{format_position(mujoco_data.xpos[ball_body_id, :3], 4)}; "
                    f"distance {goal_distance:.4f}."
                )
                if interactive:
                    break
                else:
                    break
    model.train()
    mujoco_model.opt.timestep = control_timestep
    glfw.terminate()
    if not reached_goal:
        print("Goal was not reached during this evaluation episode.")
    return reached_goal, state

def sample_action(mu, sigma):
    dist = torch.distributions.Normal(mu, sigma)
    action = dist.sample()
    return action.clamp(-1, 1), dist

def environment_geometry(model, data, clearance=0.005):
    """Return collision-safe spawn geometry derived from the MuJoCo model."""
    mujoco.mj_forward(model, data)

    def geom_bounds(name):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        center = data.geom_xpos[geom_id]
        half_size = model.geom_size[geom_id]
        return center - half_size, center + half_size

    ball_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "gball_0"
    )
    ball_radius = float(model.geom_size[ball_geom_id, 0])
    spawn_margin = ball_radius + clearance

    wall_min, wall_max = geom_bounds("wall_3")
    left_min, left_max = geom_bounds("gcase_a")
    right_min, right_max = geom_bounds("gcase_b")
    top_min, top_max = geom_bounds("gcase_c")
    bottom_min, bottom_max = geom_bounds("gcase_d")
    water_min, water_max = geom_bounds("water_surface")

    walls = {
        "x_min": float(wall_min[0]), "x_max": float(wall_max[0]),
        "y_min": float(wall_min[1]), "y_max": float(wall_max[1]),
        "z_min": float(wall_min[2]), "z_max": float(wall_max[2]),
    }
    outside_walls = {
        "x_min": float(left_max[0]), "x_max": float(right_min[0]),
        "y_min": float(bottom_max[1]), "y_max": float(top_min[1])
    }
    outside_walls["z_floor"] = float(water_min[2])
    outside_walls["z_min"] = max(outside_walls["z_floor"]+ball_radius, ball_radius)
    outside_walls["z_max"] = float(water_max[2])
    return walls, outside_walls, spawn_margin

def wall_collision(model, data):
    """Return True when the ball touches an internal or boundary wall."""
    ball_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "gball_0"
    )
    wall_geom_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("wall_3", "gcase_a", "gcase_b", "gcase_c", "gcase_d")
    }
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if (
            contact.geom1 == ball_geom_id and contact.geom2 in wall_geom_ids
        ) or (
            contact.geom2 == ball_geom_id and contact.geom1 in wall_geom_ids
        ):
            return True
    return False

def navigation_targets(positions, goal_position, walls, clearance=0.05):
    """Choose the next collision-free waypoint around the fixed obstacle."""
    positions = np.atleast_2d(positions)
    targets = np.repeat(np.asarray(goal_position)[None, :], len(positions), axis=0)
    x = positions[:, 0]
    y = positions[:, 1]
    left = walls["x_min"] - clearance
    right = walls["x_max"] + clearance
    vertical = max(abs(walls["y_min"]), abs(walls["y_max"])) + clearance
    side = np.where(y >= 0.0, 1.0, -1.0)

    before_wall = x < left
    in_blocked_band = np.abs(y) < vertical
    approach_corner = before_wall & in_blocked_band
    cross_wall = (x < right) & ~approach_corner
    targets[approach_corner, 0] = left
    targets[approach_corner, 1] = side[approach_corner] * vertical
    targets[cross_wall, 0] = right
    targets[cross_wall, 1] = side[cross_wall] * vertical
    return targets

def policy_observations(states, goal_position, walls):
    """Add the vector to the next navigation waypoint to each physical state."""
    states = np.atleast_2d(states)
    targets = navigation_targets(states[:, :3], goal_position, walls)
    observations = np.concatenate((states, targets - states[:, :3]), axis=1)
    return observations

def navigation_progress(previous_states, next_states, goal_position, walls):
    """Progress toward the waypoint selected before taking the action."""
    previous_states = np.atleast_2d(previous_states)
    next_states = np.atleast_2d(next_states)
    targets = navigation_targets(previous_states[:, :3], goal_position, walls)
    previous_distance = np.linalg.norm(targets - previous_states[:, :3], axis=1)
    next_distance = np.linalg.norm(targets - next_states[:, :3], axis=1)
    return previous_distance - next_distance

class PathNode:
    def __init__(self, position, parent=None):
        self.position = np.asarray(position, dtype=np.float64)
        self.parent = parent

def path_point_is_free(point, walls, outside_walls, clearance=PATH_CLEARANCE):
    point = np.asarray(point)
    x, y = point[:2]
    if not (
        outside_walls["x_min"] + clearance <= x
        <= outside_walls["x_max"] - clearance
        and outside_walls["y_min"] + clearance <= y
        <= outside_walls["y_max"] - clearance
    ):
        return False
    overlaps_obstacle_xy = (
        walls["x_min"] - clearance <= x <= walls["x_max"] + clearance
        and walls["y_min"] - clearance <= y <= walls["y_max"] + clearance
    )
    if len(point) == 2:
        return not overlaps_obstacle_xy
    z = point[2]
    if not (
        outside_walls["z_min"] - 0.005
        <= z <= outside_walls["z_max"] + 0.025
    ):
        return False
    # Vertical clearance represents the ball radius plus a small safety gap;
    # the larger planar RRT clearance would incorrectly forbid surface paths.
    overlaps_obstacle_z = z <= walls["z_max"] + 0.025
    return not (overlaps_obstacle_xy and overlaps_obstacle_z)

def path_segment_is_free(
    start, end, walls, outside_walls, clearance=PATH_CLEARANCE, samples=40
):
    for amount in np.linspace(0.0, 1.0, samples):
        point = (1.0 - amount) * np.asarray(start) + amount * np.asarray(end)
        if not path_point_is_free(point, walls, outside_walls, clearance):
            return False
    return True

def reconstruct_path(node):
    path = []
    while node is not None:
        path.append(node.position)
        node = node.parent
    return path[::-1]

def smooth_rrt_path(path, walls, outside_walls, clearance=PATH_CLEARANCE):
    """Remove unnecessary RRT nodes using collision-free shortcuts."""
    if len(path) <= 2:
        return path
    smoothed = [np.asarray(path[0])]
    index = 0
    while index < len(path) - 1:
        next_index = len(path) - 1
        while next_index > index + 1 and not path_segment_is_free(
            path[index], path[next_index], walls, outside_walls, clearance
        ):
            next_index -= 1
        smoothed.append(np.asarray(path[next_index]))
        index = next_index
    return smoothed

def densify_path(path, maximum_interval=0.10):
    """Resample path segments so the controller always receives nearby targets."""
    dense_path = [np.asarray(path[0], dtype=np.float64)]
    for start, end in zip(path, path[1:]):
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        distance = np.linalg.norm(end - start)
        interval_count = max(1, int(np.ceil(distance / maximum_interval)))
        for interval in range(1, interval_count + 1):
            dense_path.append(
                start + (end - start) * (interval / interval_count)
            )
    return dense_path

def plan_rrt_path(
    start,
    goal,
    walls,
    outside_walls,
    clearance=PATH_CLEARANCE,
    max_iterations=2500,
    step_size=0.075,
    goal_bias=0.15,
):
    """Plan and smooth a collision-free RRT path for the ball center."""
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    if path_segment_is_free(start, goal, walls, outside_walls, clearance):
        return densify_path([start, goal])

    nodes = [PathNode(start)]
    dimensions = len(start)
    x_min = outside_walls["x_min"] + clearance
    x_max = outside_walls["x_max"] - clearance
    y_min = outside_walls["y_min"] + clearance
    y_max = outside_walls["y_max"] - clearance
    for _ in range(max_iterations):
        if random.random() < goal_bias:
            sample = goal
        else:
            coordinates = [
                random.uniform(x_min, x_max),
                random.uniform(y_min, y_max),
            ]
            if dimensions == 3:
                coordinates.append(random.uniform(
                    outside_walls["z_min"], outside_walls["z_max"]
                ))
            sample = np.array(coordinates)
        nearest = min(nodes, key=lambda node: np.linalg.norm(node.position - sample))
        direction = sample - nearest.position
        distance = np.linalg.norm(direction)
        if distance == 0.0:
            continue
        candidate = nearest.position + direction / distance * min(step_size, distance)
        if not path_point_is_free(candidate, walls, outside_walls, clearance):
            continue
        if not path_segment_is_free(
            nearest.position, candidate, walls, outside_walls, clearance
        ):
            continue
        new_node = PathNode(candidate, nearest)
        nodes.append(new_node)
        if path_segment_is_free(
            candidate, goal, walls, outside_walls, clearance
        ):
            path = reconstruct_path(PathNode(goal, new_node))
            return densify_path(
                smooth_rrt_path(path, walls, outside_walls, clearance)
            )
    return None


def plan_navigation_path(
    start, goal, walls, outside_walls, clearance=PATH_CLEARANCE
):
    """Plan an unrestricted volumetric route through the water."""
    return plan_rrt_path(
        start, goal, walls, outside_walls, clearance=clearance
    )

def create_episode_paths(
    states, goal_position, walls, outside_walls,
    planning_retries=3,
):
    """Plan every episode without changing its sampled starting state."""
    paths = []
    for index in range(len(states)):
        path = None
        for _ in range(planning_retries):
            path = plan_rrt_path(
                states[index, :3], goal_position, walls, outside_walls
            )
            if path is not None:
                break
        if path is None:
            raise RuntimeError(
                "Unable to plan a collision-free RRT path after "
                f"{planning_retries} attempts for start {states[index, :3]}"
            )
        paths.append(path)
    return paths

def current_path_targets(states, paths, waypoint_indices, tolerance=0.02):
    dimensions = len(paths[0][0])
    targets = np.empty((len(states), dimensions), dtype=np.float64)
    for index, (state, path) in enumerate(zip(states, paths)):
        while (
            waypoint_indices[index] < len(path) - 1
            and np.linalg.norm(
                state[:dimensions] - path[waypoint_indices[index]]
            ) <= tolerance
        ):
            waypoint_indices[index] += 1
        targets[index] = path[waypoint_indices[index]]
    return targets

def observations_for_targets(states, targets):
    states = np.atleast_2d(states)
    targets = np.atleast_2d(targets)
    return np.concatenate((states, targets - states[:, :3]), axis=1)

def pid_expert_actions(states, targets, kp=6.0, kd=6.0):
    """Bounded PD controls used as collision-free path-following demonstrations."""
    states = np.atleast_2d(states)
    targets = np.atleast_2d(targets)
    controls = np.empty((len(states), 3), dtype=np.float64)
    controls[:, :2] = np.tanh(
        kp*(targets[:, :2]-states[:, :2])-kd*states[:, 3:5]
    )
    ball_radius = 0.02
    cap_height = np.clip(
        0.15-(targets[:, 2]-ball_radius), 0.0, 2.0*ball_radius
    )
    target_volume = np.pi*cap_height**2*(ball_radius-cap_height/3.0)
    sphere_volume = 4.0/3.0*np.pi*ball_radius**3
    target_submersion = target_volume/sphere_volume
    weight = 5.0*9.81
    dive_feedforward = weight-weight*target_submersion/0.5
    vertical_force = (
        80.0*(targets[:, 2]-states[:, 2])
        - 50.0*states[:, 5]
        + dive_feedforward
    )
    controls[:, 2] = np.clip(vertical_force/MOTOR_FORCE_SCALE[2], -1.0, 1.0)
    return controls

def segment_cross_track_distances(points, starts, ends):
    points = np.atleast_2d(points)
    starts = np.atleast_2d(starts)
    ends = np.atleast_2d(ends)
    segments = ends - starts
    lengths_squared = np.sum(segments * segments, axis=1)
    amounts = np.sum((points - starts) * segments, axis=1) / np.maximum(
        lengths_squared, 1e-8
    )
    amounts = np.clip(amounts, 0.0, 1.0)
    projections = starts + amounts[:, None] * segments
    return np.linalg.norm(points - projections, axis=1)

def current_path_starts(paths, waypoint_indices):
    starts = np.empty((len(paths), 3), dtype=np.float64)
    for index, path in enumerate(paths):
        starts[index] = path[max(0, waypoint_indices[index] - 1)]
    return starts

def safety_shield_actions(
    states, targets, actions, walls, outside_walls, z_positions=None
):
    """Blend toward PID control and reject predicted geometry violations."""
    states = np.atleast_2d(states)
    actions = np.atleast_2d(actions).copy()
    expert = pid_expert_actions(states, targets)
    brake = np.empty_like(actions)
    brake[:, :2] = np.tanh(-8.0*states[:, 3:5])
    brake[:, 2] = np.clip(
        -50.0*states[:, 5]/MOTOR_FORCE_SCALE[2], -1.0, 1.0
    )
    alignment = np.sum(actions*expert, axis=1)
    blend = np.where(alignment > 0.0, 0.05, 0.0)[:, None]
    actions = blend*actions + (1.0-blend)*expert
    predicted_positions = (
        states[:, :3] + states[:, 3:6] * CONTROL_TIMESTEP
        + actions * MOTOR_FORCE_SCALE * (CONTROL_TIMESTEP ** 2 / 5.0)
    )
    for index, predicted in enumerate(predicted_positions):
        collision_point = predicted
        if not path_point_is_free(
            collision_point, walls, outside_walls, clearance=0.025
        ):
            actions[index] = brake[index]
    return np.clip(actions, -1.0, 1.0)

def pretrain_actor_from_rrt(
    model,
    goal_position,
    walls,
    outside_walls,
    device,
    sample_count=4096,
    epochs=30,
):
    """Imitate a PID expert on collision-free RRT path segments."""
    states = []
    targets = []
    while len(states) < sample_count:
        start = random_position(
            walls, outside_walls, goal_position,
            minimum_goal_distance=0.2,
            safety_margin=PATH_CLEARANCE,
        )
        path = plan_rrt_path(start[:3], goal_position, walls, outside_walls)
        if path is None:
            continue
        for path_start, path_end in zip(path, path[1:]):
            for amount in np.linspace(0.0, 0.9, 5):
                position = (1.0 - amount) * path_start + amount * path_end
                velocity = np.random.uniform(-0.15, 0.15, size=3)
                states.append(np.hstack((position, velocity)))
                targets.append(path_end)
                if len(states) >= sample_count:
                    break
            if len(states) >= sample_count:
                break

    states = np.asarray(states)
    targets = np.asarray(targets)
    observations = torch.as_tensor(
        observations_for_targets(states, targets),
        dtype=torch.float32, device=device,
    )
    expert_actions = torch.as_tensor(
        pid_expert_actions(states, targets),
        dtype=torch.float32, device=device,
    )
    optimizer = optim.Adam(model.actor.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(sample_count, device=device)
        for offset in range(0, sample_count, 512):
            indices = permutation[offset:offset + 512]
            mu, _ = model.actor(observations[indices])
            loss = F.mse_loss(torch.tanh(mu), expert_actions[indices])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
            optimizer.step()
    with torch.no_grad():
        mu, _ = model.actor(observations)
        imitation_error = F.mse_loss(torch.tanh(mu), expert_actions).item()
    print(f"PID imitation pretraining MSE: {imitation_error:.6f}")
    return imitation_error

def transition(state, action, dt=0.01):
    state = np.asarray(state)
    action = np.asarray(action)
    velocity = state[3:6] + (action - np.random.normal(0, 0.1, 3)) * dt
    position = state[:3] + velocity * dt
    return np.hstack((position, velocity))

def actor_critic_training(
    model,
    goal_position,
    walls,
    outside_walls,
    actor_optimizer,
    critic_optimizer,
    num_episodes=1000,
    max_steps=100,
    gamma=0.99,
    log_interval=1,
    render=False,
    mujoco_model=None,
    mujoco_data=None,
    plot=False,
    epsilon=0.2,
    fixed=False,
    log=True,
    spawn_margin=0.025,
    collision_penalty=5.0,
    terminate_on_collision=False,
    device=torch.device("cpu")  # <-- Device parameter
):
    """
    Train the Actor-Critic Model on the selected device (CPU or GPU).
    """
    starting_positions = []
    episode_rewards = []
    actor_losses = []
    critic_losses = []

    if render and mujoco_model is not None:
        window, camera, scene, context, viewport, option = init_mujoco_render(mujoco_model)

    goal_state = np.array([goal_position[0], goal_position[1], 0.0, 0.0])
    
    for episode in range(num_episodes):
        # Pick random or fixed initial position
        if fixed:
            state = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            state = random_position(
                walls, outside_walls, goal_position,
                minimum_goal_distance=epsilon * 2,
                safety_margin=spawn_margin
            )

        starting_positions.append(state[:3])
        
        # If MuJoCo is used, place the ball in MuJoCo as well
        if mujoco_model is not None and mujoco_data is not None:
            mujoco_data.qpos[0:2] = state[0:2]
            mujoco_data.qvel[0:2] = state[2:4]
            mujoco.mj_forward(mujoco_model, mujoco_data)

        total_reward = 0
        
        for t in range(max_steps):
            critic_optimizer.zero_grad()
            actor_optimizer.zero_grad()

            # Move state to GPU/CPU device
            state_tensor = torch.tensor(
                policy_observations(state, goal_position, walls)[0],
                dtype=torch.float32, device=device
            )

            # Forward pass
            mu, sigma, value = model(state_tensor)
            action, dist = sample_action(mu, sigma)
            log_prob = dist.log_prob(action).sum()

            # Action on CPU for environment
            action_cpu = action.detach().cpu().numpy()

            # Apply action in MuJoCo or custom transition
            if mujoco_model is not None and mujoco_data is not None:
                ball_body_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "ball")
                mujoco_data.xfrc_applied[ball_body_id, :2] = action_cpu
                mujoco.mj_step(mujoco_model, mujoco_data)
                next_state = np.hstack((mujoco_data.qpos[0:2], mujoco_data.qvel[0:2]))
            else:
                next_state = transition(state, action_cpu)

            # Calculate reward
            reward = calculate_reward(next_state, goal_position, epsilon, t)
            reward += 10.0 * navigation_progress(
                state, next_state, goal_position, walls
            )[0]
            collided = (
                mujoco_data is not None
                and wall_collision(mujoco_model, mujoco_data)
            )
            if collided:
                reward -= collision_penalty
            total_reward += reward
            reached_goal = np.linalg.norm(next_state[:2] - goal_position) <= epsilon
            done = reached_goal or (terminate_on_collision and collided)

            # Next state
            next_state_tensor = torch.tensor(
                policy_observations(next_state, goal_position, walls)[0],
                dtype=torch.float32, device=device
            )
            
            # Compute next state's value (no gradient needed)
            with torch.no_grad():
                next_value = model.critic(next_state_tensor)

            td_target = reward + gamma * next_value
            advantage = td_target - value

            critic_loss = F.smooth_l1_loss(value, td_target.detach())
            actor_loss = -(advantage.detach() * log_prob).mean()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

            # Backpropagation
            critic_loss.backward()
            actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)

            # Optimizer step
            critic_optimizer.step()
            actor_optimizer.step()

            if log:
                print(f"Episode {episode}, Step {t}")
                print(f"  - TD Error: {advantage.item()}")
                print(f"  - Critic Loss: {critic_loss.item()}, Actor Loss: {actor_loss.item()}")
                print(f"  - Action Mean: {mu.detach().cpu().numpy()}, Action Std: {sigma.detach().cpu().numpy()}")
                print(f"  - Action Taken: {action_cpu}")

            # Render if requested
            if render and mujoco_model is not None and mujoco_data is not None:
                render_mujoco_scene(mujoco_model, mujoco_data, scene, context, viewport, camera, window, option)

            if done:
                break
            
            state = next_state

        # Store the total reward (averaged per step if you wish)
        episode_rewards.append(total_reward / (t + 1) if t > 0 else 0)

        if episode % log_interval == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    if render and mujoco_model is not None:
        glfw.terminate()

    if plot:
        plot_starting_points(starting_positions, goal_position, outside_walls, walls, epsilon)
        plot_learning_curve(episode_rewards)
        plot_loss_curve(critic_losses, actor_losses)

    return model

def actor_critic_batch_training(
    model,
    goal_position,
    walls,
    outside_walls,
    actor_optimizer,
    critic_optimizer,
    num_episodes=360,
    batch_size=64,
    max_steps=1800,
    gamma=0.99,
    mujoco_model=None,
    epsilon=0.2,
    plot=False,
    spawn_margin=0.025,
    collision_penalty=5.0,
    terminate_on_collision=False,
    min_update_batch_size=8,
    device=torch.device("cpu")
):
    """Train independent environments in parallel neural-network batches."""
    starting_positions = []
    episode_rewards = []
    actor_losses = []
    critic_losses = []
    all_successes = []
    all_collisions = []

    for batch_start in range(0, num_episodes, batch_size):
        current_batch_size = min(batch_size, num_episodes - batch_start)
        states = np.stack([
            random_position(
                walls, outside_walls, goal_position,
                minimum_goal_distance=epsilon * 2,
                safety_margin=spawn_margin
            )
            for _ in range(current_batch_size)
        ])
        starting_positions.extend(states[:, :3].copy().tolist())

        mujoco_batch = None
        if mujoco_model is not None:
            mujoco_batch = [mujoco.MjData(mujoco_model) for _ in range(current_batch_size)]
            for state, data in zip(states, mujoco_batch):
                data.qpos[0:2] = state[0:2]
                data.qvel[0:2] = state[2:4]
                mujoco.mj_forward(mujoco_model, data)
            ball_body_id = mujoco.mj_name2id(
                mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "ball"
            )

        active = np.ones(current_batch_size, dtype=bool)
        total_rewards = np.zeros(current_batch_size, dtype=np.float64)
        step_counts = np.zeros(current_batch_size, dtype=np.int32)
        episode_successes = np.zeros(current_batch_size, dtype=bool)
        episode_collisions = np.zeros(current_batch_size, dtype=bool)

        for t in range(max_steps):
            active_indices = np.flatnonzero(active)
            if active_indices.size == 0:
                break

            state_tensor = torch.as_tensor(
                policy_observations(states[active_indices], goal_position, walls),
                dtype=torch.float32, device=device
            )
            mu, sigma, values = model(state_tensor)
            actions, dist = sample_action(mu, sigma)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            actions_cpu = actions.detach().cpu().numpy()

            next_states = np.empty_like(states[active_indices])
            collisions = np.zeros(active_indices.size, dtype=bool)
            if mujoco_batch is not None:
                for local_index, (env_index, action) in enumerate(
                    zip(active_indices, actions_cpu)
                ):
                    data = mujoco_batch[env_index]
                    data.xfrc_applied[ball_body_id, :2] = action
                    mujoco.mj_step(mujoco_model, data)
                    next_states[local_index] = np.hstack(
                        (data.qpos[0:2], data.qvel[0:2])
                    )
                    collisions[local_index] = wall_collision(mujoco_model, data)
            else:
                next_states = np.stack([
                    transition(state, action)
                    for state, action in zip(states[active_indices], actions_cpu)
                ])

            distances = np.linalg.norm(
                next_states[:, :2] - goal_position, axis=1
            )
            progress = navigation_progress(
                states[active_indices], next_states, goal_position, walls
            )
            rewards = -distances * 2 - 1e-2 + 10.0 * progress
            rewards += (distances <= epsilon) * 100
            rewards -= collisions * collision_penalty
            reached_goal = np.linalg.norm(
                next_states[:, :2] - goal_position, axis=1
            ) <= epsilon
            done = reached_goal | (collisions & terminate_on_collision)

            reward_tensor = torch.as_tensor(
                rewards, dtype=torch.float32, device=device
            )
            done_tensor = torch.as_tensor(done, dtype=torch.bool, device=device)
            with torch.no_grad():
                next_values = model.critic(
                    torch.as_tensor(
                        policy_observations(next_states, goal_position, walls),
                        dtype=torch.float32, device=device
                    )
                ).squeeze(-1)
                td_targets = reward_tensor + gamma * next_values * (~done_tensor)

            values = values.squeeze(-1)
            advantages = td_targets - values
            critic_loss = F.smooth_l1_loss(values, td_targets)
            actor_advantages = advantages.detach()
            if actor_advantages.numel() > 1:
                actor_advantages = (
                    actor_advantages - actor_advantages.mean()
                ) / (actor_advantages.std(unbiased=False) + 1e-8)
            actor_loss = -(actor_advantages * log_probs).mean()

            if active_indices.size >= min_update_batch_size:
                critic_optimizer.zero_grad()
                actor_optimizer.zero_grad()
                critic_loss.backward()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
                critic_optimizer.step()
                actor_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
            total_rewards[active_indices] += rewards
            step_counts[active_indices] += 1
            states[active_indices] = next_states
            episode_successes[active_indices] |= reached_goal
            episode_collisions[active_indices] |= collisions
            active[active_indices[done]] = False

        averaged_rewards = total_rewards / np.maximum(step_counts, 1)
        episode_rewards.extend(averaged_rewards.tolist())
        all_successes.extend(episode_successes.tolist())
        all_collisions.extend(episode_collisions.tolist())
        completed = batch_start + current_batch_size
        print(
            f"Completed {completed}/{num_episodes} episodes; "
            f"batch mean reward: {averaged_rewards.mean():.3f}; "
            f"success rate: {episode_successes.mean():.1%}; "
            f"collision rate: {episode_collisions.mean():.1%}"
        )

    model.training_metrics = {
        "starting_positions": np.asarray(starting_positions),
        "episode_rewards": np.asarray(episode_rewards),
        "actor_losses": np.asarray(actor_losses),
        "critic_losses": np.asarray(critic_losses),
        "successes": np.asarray(all_successes, dtype=bool),
        "collisions": np.asarray(all_collisions, dtype=bool),
    }

    if plot:
        plot_starting_points(
            starting_positions, goal_position, outside_walls, walls, epsilon
        )
        plot_learning_curve(episode_rewards)
        plot_loss_curve(critic_losses, actor_losses)

    return model

def ppo_log_prob(mu, sigma, raw_actions):
    """Log probability for tanh-squashed Normal actions."""
    distribution = torch.distributions.Normal(mu, sigma)
    actions = torch.tanh(raw_actions)
    correction = torch.log(1.0 - actions.pow(2) + 1e-6)
    return (distribution.log_prob(raw_actions) - correction).sum(dim=-1)

def deterministic_evaluation(
    model,
    goal_position,
    walls,
    outside_walls,
    mujoco_model,
    episodes=32,
    max_steps=4800,
    epsilon=0.1,
    collision_penalty=5.0,
    expert_only=False,
    device=torch.device("cpu"),
):
    """Evaluate the mean policy without exploration or weight updates."""
    spawn_margin = max(PATH_CLEARANCE, float(mujoco_model.geom_size[
        mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_GEOM, "gball_0"), 0
    ]) + 0.005)
    states = np.stack([
        random_position(
            walls, outside_walls, goal_position,
            minimum_goal_distance=epsilon * 2,
            safety_margin=spawn_margin,
        )
        for _ in range(episodes)
    ])
    ball_body_id = mujoco.mj_name2id(
        mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "ball"
    )
    paths = create_episode_paths(
        states, goal_position, walls, outside_walls,
    )
    data_batch = [mujoco.MjData(mujoco_model) for _ in range(episodes)]
    for state, data in zip(states, data_batch):
        set_simulation_state(mujoco_model, data, state, ball_body_id)
    waypoint_indices = np.ones(episodes, dtype=np.int32)
    active = np.ones(episodes, dtype=bool)
    successes = np.zeros(episodes, dtype=bool)
    collisions = np.zeros(episodes, dtype=bool)
    total_rewards = np.zeros(episodes)
    step_counts = np.zeros(episodes, dtype=np.int32)
    best_distances = np.linalg.norm(states[:, :3]-goal_position, axis=1)
    no_progress_steps = np.zeros(episodes, dtype=np.int32)

    model.eval()
    with torch.no_grad():
        for _ in range(max_steps):
            indices = np.flatnonzero(active)
            if indices.size == 0:
                break
            replanned = np.zeros(episodes, dtype=bool)
            for env_index in indices[no_progress_steps[indices] >= 250]:
                new_path = plan_navigation_path(
                    states[env_index, :3], goal_position, walls, outside_walls
                )
                if new_path is not None:
                    paths[env_index] = new_path
                    waypoint_indices[env_index] = 1
                    no_progress_steps[env_index] = 0
                    replanned[env_index] = True
            targets = current_path_targets(states, paths, waypoint_indices)
            state_tensor = torch.as_tensor(
                observations_for_targets(states[indices], targets[indices]),
                dtype=torch.float32, device=device
            )
            mu, _, _ = model(state_tensor)
            if expert_only:
                expert_actions = pid_expert_actions(
                    states[indices], targets[indices]
                )
                actions = safety_shield_actions(
                    states[indices], targets[indices], expert_actions,
                    walls, outside_walls,
                )
            else:
                actions = safety_shield_actions(
                    states[indices], targets[indices],
                    torch.tanh(mu).cpu().numpy(), walls, outside_walls,
                )
            recovering = (
                (no_progress_steps[indices] >= 250) | replanned[indices]
            )
            if recovering.any():
                actions[recovering] = safety_shield_actions(
                    states[indices][recovering], targets[indices][recovering],
                    pid_expert_actions(
                        states[indices][recovering], targets[indices][recovering]
                    ), walls, outside_walls,
                )
            next_states = np.empty_like(states[indices])
            step_collisions = np.zeros(indices.size, dtype=bool)
            for local_index, (env_index, action) in enumerate(zip(indices, actions)):
                data = data_batch[env_index]
                step_training_simulation(
                    mujoco_model, data, ball_body_id, action
                )
                next_states[local_index] = simulation_state(
                    mujoco_model, data, ball_body_id
                )
                step_collisions[local_index] = wall_collision(mujoco_model, data)

            new_distances = np.linalg.norm(
                next_states[:, :3] - goal_position, axis=1
            )
            old_goal_distances = np.linalg.norm(
                states[indices, :3] - goal_position, axis=1
            )
            rewards = (
                25.0 * (old_goal_distances - new_distances)
                - 0.02 * new_distances - 0.002
            )
            old_waypoint_distances = np.linalg.norm(
                targets[indices] - states[indices, :3], axis=1
            )
            new_waypoint_distances = np.linalg.norm(
                targets[indices] - next_states[:, :3], axis=1
            )
            rewards += 10.0 * (
                old_waypoint_distances - new_waypoint_distances
            )
            rewards -= 0.002 * np.sum(actions * actions, axis=1)
            rewards -= 0.05 * (
                (new_waypoint_distances > epsilon)
                & (np.linalg.norm(next_states[:, 3:6], axis=1) < 0.01)
            )
            reached_goal = new_distances <= epsilon
            improved = new_distances < best_distances[indices]-1e-4
            best_distances[indices] = np.minimum(
                best_distances[indices], new_distances
            )
            no_progress_steps[indices] = np.where(
                improved, 0, no_progress_steps[indices]+1
            )
            rewards += reached_goal * 25.0
            rewards -= step_collisions * collision_penalty

            states[indices] = next_states
            total_rewards[indices] += rewards
            step_counts[indices] += 1
            successes[indices] |= reached_goal
            collisions[indices] |= step_collisions
            finished = reached_goal | step_collisions
            active[indices[finished]] = False
    model.train()
    final_distances = np.linalg.norm(states[:, :3]-goal_position, axis=1)
    return {
        "success_rate": float(successes.mean()),
        "collision_rate": float(collisions.mean()),
        "timeout_rate": float(active.mean()),
        "timeout_mean_distance": (
            float(final_distances[active].mean()) if active.any() else 0.0
        ),
        "mean_reward": float((total_rewards / np.maximum(step_counts, 1)).mean()),
    }

def ppo_training(
    model,
    goal_position,
    walls,
    outside_walls,
    num_episodes=360,
    batch_size=64,
    max_steps=4800,
    gamma=0.99,
    gae_lambda=0.95,
    clip_ratio=0.2,
    ppo_epochs=6,
    minibatch_size=1024,
    learning_rate=3e-4,
    value_coefficient=0.5,
    entropy_coefficient=0.01,
    mujoco_model=None,
    epsilon=0.1,
    collision_penalty=25.0,
    evaluation_episodes=32,
    imitation_epochs=30,
    checkpoint_path="best_ppo_model.pt",
    plot=False,
    plot_directory=None,
    device=torch.device("cpu"),
):
    """Train with clipped PPO, GAE, deterministic evaluation, and checkpointing."""
    if imitation_epochs > 0:
        pretrain_actor_from_rrt(
            model, goal_position, walls, outside_walls, device,
            epochs=imitation_epochs,
        )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    starting_positions = []
    episode_rewards = []
    actor_losses = []
    critic_losses = []
    all_successes = []
    all_collisions = []
    evaluations = []
    best_score = (-float("inf"), -1.0, -float("inf"))
    spawn_margin = max(PATH_CLEARANCE, float(mujoco_model.geom_size[
        mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_GEOM, "gball_0"), 0
    ]) + 0.005)
    ball_body_id = mujoco.mj_name2id(
        mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "ball"
    )

    for batch_start in range(0, num_episodes, batch_size):
        current_size = min(batch_size, num_episodes - batch_start)
        states = np.stack([
            random_position(
                walls, outside_walls, goal_position,
                minimum_goal_distance=epsilon * 2,
                safety_margin=spawn_margin,
            )
            for _ in range(current_size)
        ])
        paths = create_episode_paths(
            states, goal_position, walls, outside_walls,
        )
        starting_positions.extend(states[:, :3].copy().tolist())
        data_batch = [mujoco.MjData(mujoco_model) for _ in range(current_size)]
        for state, data in zip(states, data_batch):
            set_simulation_state(mujoco_model, data, state, ball_body_id)
        waypoint_indices = np.ones(current_size, dtype=np.int32)

        active = np.ones(current_size, dtype=bool)
        successes = np.zeros(current_size, dtype=bool)
        collisions = np.zeros(current_size, dtype=bool)
        total_rewards = np.zeros(current_size)
        step_counts = np.zeros(current_size, dtype=np.int32)
        best_distances = np.linalg.norm(states[:, :3]-goal_position, axis=1)
        no_progress_steps = np.zeros(current_size, dtype=np.int32)
        observations_buffer = []
        raw_actions_buffer = []
        old_log_probs_buffer = []
        values_buffer = []
        rewards_buffer = []
        dones_buffer = []
        valid_buffer = []

        model.eval()
        for step in range(max_steps):
            valid = active.copy()
            if not valid.any():
                break
            replanned = np.zeros(current_size, dtype=bool)
            for env_index in np.flatnonzero(
                valid & (no_progress_steps >= 250)
            ):
                new_path = plan_navigation_path(
                    states[env_index, :3], goal_position, walls, outside_walls
                )
                if new_path is not None:
                    paths[env_index] = new_path
                    waypoint_indices[env_index] = 1
                    no_progress_steps[env_index] = 0
                    replanned[env_index] = True
            targets = current_path_targets(states, paths, waypoint_indices)
            segment_starts = current_path_starts(paths, waypoint_indices)
            observations = observations_for_targets(states, targets)
            state_tensor = torch.as_tensor(
                observations, dtype=torch.float32, device=device
            )
            with torch.no_grad():
                mu, sigma, values = model(state_tensor)
                distribution = torch.distributions.Normal(mu, sigma)
                raw_actions = distribution.sample()
                actions = torch.tanh(raw_actions)
                log_probs = ppo_log_prob(mu, sigma, raw_actions)

            actions_cpu = safety_shield_actions(
                states, targets, actions.cpu().numpy(), walls, outside_walls
            )
            recovering = valid & (
                (no_progress_steps >= 250) | replanned
            )
            if recovering.any():
                actions_cpu[recovering] = safety_shield_actions(
                    states[recovering], targets[recovering],
                    pid_expert_actions(states[recovering], targets[recovering]),
                    walls, outside_walls,
                )
            raw_actions = torch.as_tensor(
                np.arctanh(np.clip(actions_cpu, -0.999999, 0.999999)),
                dtype=torch.float32, device=device,
            )
            with torch.no_grad():
                log_probs = ppo_log_prob(mu, sigma, raw_actions)
            next_states = states.copy()
            step_collisions = np.zeros(current_size, dtype=bool)
            for env_index in np.flatnonzero(valid):
                data = data_batch[env_index]
                step_training_simulation(
                    mujoco_model, data, ball_body_id, actions_cpu[env_index]
                )
                next_states[env_index] = simulation_state(
                    mujoco_model, data, ball_body_id
                )
                step_collisions[env_index] = wall_collision(mujoco_model, data)

            new_distances = np.linalg.norm(next_states[:, :3] - goal_position, axis=1)
            old_goal_distances = np.linalg.norm(
                states[:, :3] - goal_position, axis=1
            )
            rewards = (
                25.0 * (old_goal_distances - new_distances)
                - 0.02 * new_distances - 0.002
            )
            old_waypoint_distances = np.linalg.norm(
                targets - states[:, :3], axis=1
            )
            new_waypoint_distances = np.linalg.norm(
                targets - next_states[:, :3], axis=1
            )
            rewards += 10.0 * (
                old_waypoint_distances - new_waypoint_distances
            )
            rewards -= 0.002 * np.sum(actions_cpu * actions_cpu, axis=1)
            rewards -= 0.05 * (
                (new_waypoint_distances > epsilon)
                & (np.linalg.norm(next_states[:, 3:6], axis=1) < 0.01)
            )
            rewards -= 0.5 * segment_cross_track_distances(
                next_states[:, :3], segment_starts, targets
            )
            reached_goal = (new_distances <= epsilon) & valid
            improved = new_distances < best_distances-1e-4
            best_distances[valid] = np.minimum(
                best_distances[valid], new_distances[valid]
            )
            no_progress_steps[valid] = np.where(
                improved[valid], 0, no_progress_steps[valid]+1
            )
            rewards += reached_goal * 25.0
            rewards -= step_collisions * collision_penalty
            timed_out = valid & (step == max_steps - 1)
            dones = reached_goal | step_collisions | timed_out
            rewards[~valid] = 0.0

            observations_buffer.append(observations.copy())
            raw_actions_buffer.append(raw_actions.cpu().numpy())
            old_log_probs_buffer.append(log_probs.cpu().numpy())
            values_buffer.append(values.squeeze(-1).cpu().numpy())
            rewards_buffer.append(rewards.copy())
            dones_buffer.append(dones.copy())
            valid_buffer.append(valid.copy())

            states = next_states
            total_rewards[valid] += rewards[valid]
            step_counts[valid] += 1
            successes |= reached_goal
            collisions |= step_collisions
            active[dones] = False

        model.train()
        observations = np.asarray(observations_buffer)
        raw_actions = np.asarray(raw_actions_buffer)
        old_log_probs = np.asarray(old_log_probs_buffer)
        values = np.asarray(values_buffer)
        rewards = np.asarray(rewards_buffer)
        dones = np.asarray(dones_buffer)
        valid = np.asarray(valid_buffer)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = np.zeros(current_size, dtype=np.float32)
        next_values = np.zeros(current_size, dtype=np.float32)
        for time_index in reversed(range(len(rewards))):
            not_done = 1.0 - dones[time_index].astype(np.float32)
            delta = rewards[time_index] + gamma * next_values * not_done - values[time_index]
            gae = delta + gamma * gae_lambda * not_done * gae
            gae *= valid[time_index]
            advantages[time_index] = gae
            next_values = values[time_index]
        returns = advantages + values

        flat_valid = valid.reshape(-1)
        flat_observations = torch.as_tensor(
            observations.reshape(-1, observations.shape[-1])[flat_valid],
            dtype=torch.float32, device=device,
        )
        flat_raw_actions = torch.as_tensor(
            raw_actions.reshape(-1, raw_actions.shape[-1])[flat_valid],
            dtype=torch.float32, device=device,
        )
        flat_old_log_probs = torch.as_tensor(
            old_log_probs.reshape(-1)[flat_valid], dtype=torch.float32, device=device,
        )
        flat_advantages = torch.as_tensor(
            advantages.reshape(-1)[flat_valid], dtype=torch.float32, device=device,
        )
        flat_returns = torch.as_tensor(
            returns.reshape(-1)[flat_valid], dtype=torch.float32, device=device,
        )
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (
            flat_advantages.std(unbiased=False) + 1e-8
        )

        sample_count = flat_observations.shape[0]
        for _ in range(ppo_epochs):
            permutation = torch.randperm(sample_count, device=device)
            for offset in range(0, sample_count, minibatch_size):
                indices = permutation[offset:offset + minibatch_size]
                mu, sigma, predicted_values = model(flat_observations[indices])
                new_log_probs = ppo_log_prob(mu, sigma, flat_raw_actions[indices])
                ratio = torch.exp(new_log_probs - flat_old_log_probs[indices])
                surrogate = ratio * flat_advantages[indices]
                clipped_surrogate = torch.clamp(
                    ratio, 1.0 - clip_ratio, 1.0 + clip_ratio
                ) * flat_advantages[indices]
                actor_loss = -torch.min(surrogate, clipped_surrogate).mean()
                critic_loss = F.smooth_l1_loss(
                    predicted_values.squeeze(-1), flat_returns[indices]
                )
                entropy = torch.distributions.Normal(mu, sigma).entropy().sum(-1).mean()
                loss = actor_loss + value_coefficient * critic_loss - entropy_coefficient * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        episode_rewards.extend(total_rewards.tolist())
        all_successes.extend(successes.tolist())
        all_collisions.extend(collisions.tolist())
        evaluation = deterministic_evaluation(
            model, goal_position, walls, outside_walls, mujoco_model,
            episodes=evaluation_episodes, max_steps=max_steps,
            epsilon=epsilon, collision_penalty=collision_penalty, device=device,
        )
        evaluation["episodes_completed"] = batch_start + current_size
        evaluations.append(evaluation)
        score = (
            -evaluation["collision_rate"],
            evaluation["success_rate"],
            evaluation["mean_reward"],
        )
        if score > best_score:
            best_score = score
            torch.save({
                "model_state_dict": model.state_dict(),
                "evaluation": evaluation,
                "episodes_completed": batch_start + current_size,
            }, checkpoint_path)
        print(
            f"PPO {batch_start + current_size}/{num_episodes}; "
            f"train success: {successes.mean():.1%}; "
            f"eval success: {evaluation['success_rate']:.1%}; "
            f"eval collisions: {evaluation['collision_rate']:.1%}; "
            f"eval reward: {evaluation['mean_reward']:.3f}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.training_metrics = {
        "starting_positions": np.asarray(starting_positions),
        "episode_rewards": np.asarray(episode_rewards),
        "actor_losses": np.asarray(actor_losses),
        "critic_losses": np.asarray(critic_losses),
        "successes": np.asarray(all_successes, dtype=bool),
        "collisions": np.asarray(all_collisions, dtype=bool),
        "evaluations": evaluations,
        "best_evaluation": checkpoint["evaluation"],
    }
    if plot or plot_directory:
        if plot_directory:
            os.makedirs(plot_directory, exist_ok=True)
        plot_starting_points(
            starting_positions, goal_position, outside_walls, walls, epsilon,
            output_path=(os.path.join(plot_directory, "starting_positions.png")
                         if plot_directory else None),
        )
        plot_learning_curve(
            episode_rewards,
            output_path=(os.path.join(plot_directory, "learning_curve.png")
                         if plot_directory else None),
        )
        plot_loss_curve(
            critic_losses, actor_losses,
            output_path=(os.path.join(plot_directory, "loss_curves.png")
                         if plot_directory else None),
        )
        if plot_directory:
            plot_turbulence(
                walls,
                os.path.join(plot_directory, "turbulence.png"),
            )
    return model

def calculate_reward(state, goal_position, epsilon, t):
    distance_to_goal = np.linalg.norm(state[:3] - goal_position)
    reward = -distance_to_goal * 2
    if distance_to_goal <= epsilon:
        reward += 100
    reward -= 1e-2
    return reward

def random_position(
    walls, outside_walls, goal_position, minimum_goal_distance=0.0,
    safety_margin=0.025
):
    x_min = outside_walls['x_min'] 
    x_max = outside_walls['x_max']
    y_min = outside_walls['y_min']
    y_max = outside_walls['y_max']

    while True:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        z = random.uniform(outside_walls["z_min"], outside_walls["z_max"])
        far_enough_from_goal = (
            np.linalg.norm(np.array([x, y, z]) - goal_position)
            >= minimum_goal_distance
        )
        if (
            not is_inside_wall(
                x, y, walls, outside_walls, safety_margin, z=z
            )
            and far_enough_from_goal
        ):
            return np.array([x, y, z, 0.0, 0.0, 0.0])

def is_inside_wall(
    x, y, walls, outside_walls, safety_margin=0.025, z=None
):
    # Outside boundary
    if not (outside_walls['x_min'] + safety_margin <= x <= outside_walls['x_max'] - safety_margin and
            outside_walls['y_min'] + safety_margin <= y <= outside_walls['y_max'] - safety_margin):
        return True
    # Inside the internal wall
    overlaps_xy = (
        walls['x_min'] - safety_margin <= x <= walls['x_max'] + safety_margin
        and walls['y_min'] - safety_margin <= y <= walls['y_max'] + safety_margin
    )
    # Use ball-radius clearance vertically rather than the wider planar RRT
    # margin, allowing safe spawns above the obstacle.
    overlaps_z = z is None or z <= walls["z_max"]+0.025
    if overlaps_xy and overlaps_z:
        return True
    return False

def plot_loss_curve(critic_losses, actor_losses, output_path=None):
    figure, (critic_axis, actor_axis) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    critic_axis.plot(critic_losses, color="tab:blue", alpha=0.7)
    critic_axis.set_ylabel("Critic Loss")
    critic_axis.set_title("Critic Loss Over Training")
    critic_axis.grid(alpha=0.2)

    actor_axis.plot(actor_losses, color="tab:orange", alpha=0.7)
    actor_axis.set_xlabel("Training Steps")
    actor_axis.set_ylabel("Actor Loss")
    actor_axis.set_title("Actor Loss Over Training")
    actor_axis.grid(alpha=0.2)

    figure.tight_layout()
    if output_path:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()
    return

def plot_learning_curve(episode_rewards, output_path=None):
    window_size = 10  # Adjust for smoothing
    moving_avg_rewards = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(episode_rewards, label="Episode return", alpha=0.5)
    axis.plot(range(window_size - 1, len(episode_rewards)), moving_avg_rewards,
             label=f"Moving Average (window={window_size})", color='orange')
    axis.set_xlabel("Episode")
    axis.set_ylabel("Episode Return")
    axis.set_title("Episode Return with Moving Average")
    axis.legend()
    if output_path:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()
    return 

def plot_starting_points(
    starting_positions, goal_position, outside_walls, wall, epsilon,
    output_path=None,
):
    figure = plt.figure(figsize=(13, 8))
    figure.subplots_adjust(left=0.05, right=0.88, bottom=0.14, top=0.90)
    axis = figure.add_subplot(111, projection="3d")
    starting_positions = np.asarray(starting_positions)
    axis.scatter(
        starting_positions[:, 0], starting_positions[:, 1],
        starting_positions[:, 2], color="tab:blue", s=16, alpha=0.65,
        label="Starting positions",
    )
    axis.scatter(
        *goal_position[:3], color="black", marker="x", s=28,
        depthshade=False, label="Goal center",
    )
    longitude, latitude = np.meshgrid(
        np.linspace(0, 2*np.pi, 30), np.linspace(0, np.pi, 16)
    )
    axis.plot_surface(
        goal_position[0]+epsilon*np.cos(longitude)*np.sin(latitude),
        goal_position[1]+epsilon*np.sin(longitude)*np.sin(latitude),
        goal_position[2]+epsilon*np.cos(latitude),
        color="cyan", alpha=0.25, linewidth=0,
    )
    axis.bar3d(
        wall["x_min"], wall["y_min"], wall["z_min"],
        wall["x_max"]-wall["x_min"],
        wall["y_max"]-wall["y_min"],
        wall["z_max"]-wall["z_min"],
        color="grey", alpha=0.55, shade=True, label="Obstacle",
    )
    x_values = [outside_walls["x_min"], outside_walls["x_max"]]
    y_values = [outside_walls["y_min"], outside_walls["y_max"]]
    z_values = [outside_walls.get("z_floor", 0.0), outside_walls["z_max"]]
    corners = [(x, y, z) for x in x_values for y in y_values for z in z_values]
    for index, first in enumerate(corners):
        for second in corners[index+1:]:
            if sum(a != b for a, b in zip(first, second)) == 1:
                axis.plot(*zip(first, second), color="black", alpha=0.5)
    axis.set_xlabel("X Position")
    axis.set_ylabel("Y Position")
    axis.set_zlabel("Z Position")
    axis.xaxis.labelpad = 18
    axis.yaxis.labelpad = 18
    axis.zaxis.labelpad = 12
    axis.tick_params(axis="x", pad=2, labelsize=9)
    axis.tick_params(axis="y", pad=2, labelsize=9)
    axis.tick_params(axis="z", pad=2, labelsize=9)
    axis.set_title("3D Starting Positions and Underwater Goal")
    axis.legend()
    axis.set_xlim(*x_values)
    axis.set_ylim(*y_values)
    axis.set_zlim(0.0, max(0.2, outside_walls["z_max"]+0.02))
    axis.set_box_aspect((
        x_values[1]-x_values[0],
        y_values[1]-y_values[0],
        0.2,
    ))
    axis.view_init(elev=27, azim=-58)
    if output_path:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()
    return


def plot_rrt_paths_3d(paths, goal_position, outside_walls, wall, output_path=None):
    """Plot several collision-free volumetric RRT paths."""
    figure = plt.figure(figsize=(13, 8))
    figure.subplots_adjust(left=0.05, right=0.88, bottom=0.14, top=0.90)
    axis = figure.add_subplot(111, projection="3d")
    colors = mpl.colormaps["tab10"](np.linspace(0, 1, len(paths)))
    for index, (path, color) in enumerate(zip(paths, colors), start=1):
        points = np.asarray(path)
        axis.plot(
            points[:, 0], points[:, 1], points[:, 2], color=color,
            linewidth=2.2, label=f"Path {index}",
        )
        axis.scatter(*points[0], color=color, marker="o", s=28, depthshade=False)
    axis.scatter(
        *goal_position[:3], color="black", marker="x", s=38,
        depthshade=False, label="Underwater goal",
    )
    axis.bar3d(
        wall["x_min"], wall["y_min"], wall["z_min"],
        wall["x_max"]-wall["x_min"], wall["y_max"]-wall["y_min"],
        wall["z_max"]-wall["z_min"], color="grey", alpha=0.5,
        shade=True, label="Obstacle",
    )
    x_values = [outside_walls["x_min"], outside_walls["x_max"]]
    y_values = [outside_walls["y_min"], outside_walls["y_max"]]
    z_values = [outside_walls.get("z_floor", 0.0), outside_walls["z_max"]]
    corners = [(x, y, z) for x in x_values for y in y_values for z in z_values]
    for corner_index, first in enumerate(corners):
        for second in corners[corner_index+1:]:
            if sum(a != b for a, b in zip(first, second)) == 1:
                axis.plot(*zip(first, second), color="black", alpha=0.45)
    axis.set_xlabel("X Position", labelpad=18)
    axis.set_ylabel("Y Position", labelpad=18)
    axis.set_zlabel("Z Position", labelpad=12)
    axis.tick_params(axis="x", pad=2, labelsize=9)
    axis.tick_params(axis="y", pad=2, labelsize=9)
    axis.tick_params(axis="z", pad=2, labelsize=9)
    axis.set_title("Example 3D RRT Paths", pad=18)
    axis.set_xlim(*x_values)
    axis.set_ylim(*y_values)
    axis.set_zlim(0.0, 0.2)
    axis.set_box_aspect((x_values[1]-x_values[0], y_values[1]-y_values[0], 0.2))
    axis.view_init(elev=27, azim=-58)
    axis.legend(loc="upper right", fontsize=9)
    if output_path:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()


def plot_turbulence(wall, output_path=None, simulation_steps=400):
    """Plot 3D velocity arrows colored by local vorticity magnitude."""
    from .fluid import FluidDynamics

    fluid = FluidDynamics()
    for _ in range(simulation_steps):
        fluid.field.step(0.01, fluid.density)
    vorticity = np.linalg.norm(fluid.field.vorticity(), axis=-1)
    sample = np.zeros_like(vorticity, dtype=bool)
    sample[::2, ::3, ::3] = True
    sample &= ~fluid.field.solid
    velocity = fluid.field.velocity[sample]
    speed = np.linalg.norm(velocity, axis=1)
    sample_indices = np.flatnonzero(sample)
    keep = speed > np.percentile(speed, 15)
    selected = sample_indices[keep]
    mask = np.zeros_like(sample)
    mask.flat[selected] = True
    selected_vorticity = vorticity[mask]
    normalization = mpl.colors.Normalize(
        vmin=float(selected_vorticity.min()),
        vmax=float(selected_vorticity.max()),
    )
    colors = mpl.colormaps["inferno"](normalization(selected_vorticity))

    figure = plt.figure(figsize=(13, 8))
    figure.subplots_adjust(left=0.04, right=0.72, bottom=0.12, top=0.90)
    axis = figure.add_subplot(111, projection="3d")
    axis.quiver(
        fluid.field.x[mask], fluid.field.y[mask], fluid.field.z[mask],
        fluid.field.velocity[..., 0][mask],
        fluid.field.velocity[..., 1][mask],
        fluid.field.velocity[..., 2][mask],
        length=0.55, normalize=False, colors=colors,
        linewidth=0.8, arrow_length_ratio=0.3,
    )
    axis.bar3d(
        wall["x_min"], wall["y_min"], wall["z_min"],
        wall["x_max"]-wall["x_min"], wall["y_max"]-wall["y_min"],
        wall["z_max"]-wall["z_min"], color="grey", alpha=0.4,
    )
    color_map = mpl.cm.ScalarMappable(norm=normalization, cmap="inferno")
    color_map.set_array([])
    colorbar = figure.colorbar(
        color_map, ax=axis, shrink=0.68, pad=0.18,
        label="Vorticity magnitude (1/s)",
    )
    colorbar.ax.yaxis.labelpad = 12
    axis.set_xlabel("X Position")
    axis.set_ylabel("Y Position")
    axis.set_zlabel("Z Position")
    axis.xaxis.labelpad = 16
    axis.yaxis.labelpad = 16
    axis.zaxis.labelpad = 10
    axis.tick_params(axis="x", pad=2, labelsize=9)
    axis.tick_params(axis="y", pad=2, labelsize=9)
    axis.tick_params(axis="z", pad=2, labelsize=9)
    axis.set_title("3D Turbulent Velocity Field", pad=18)
    axis.set_box_aspect((1.2, 0.64, 0.15))
    axis.view_init(elev=24, azim=-58)
    if output_path:
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()

def main():
    from .commands import parse_destination
    
    # Select device: GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    goal_position = np.array([0.9, -0.22, 0.06])
    destination_positions = {
        "start": np.array([0.0, 0.0, 0.15]),
        "goal": goal_position,
        "underwater": goal_position,
    }
    epsilon = 0.015

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    environment_path = os.path.join(project_root, "env", "ball_square.xml")
    models_directory = os.path.join(project_root, "models")
    os.makedirs(models_directory, exist_ok=True)

    # Load MuJoCo model
    mujoco_model = mujoco.MjModel.from_xml_path(environment_path)
    mujoco_model.opt.timestep = CONTROL_TIMESTEP
    mujoco_data = mujoco.MjData(mujoco_model)
    walls, outside_walls, spawn_margin = environment_geometry(
        mujoco_model, mujoco_data
    )

    # Create and move Actor-Critic Model to the chosen device
    model = ActorCriticModel(input_dim=9, hidden_dim1=32, hidden_dim2=64, hidden_dim3=32, action_dim=3).to(device)
    checkpoint_path = os.path.join(models_directory, "best_ppo_model.pt")

    current_state = np.array([0.0, 0.0, 0.15, 0.0, 0.0, 0.0])
    checkpoint_loaded = False

    menu_options = {
        "1": ("start", "Navigate to surface start"),
        "2": ("underwater", "Navigate to underwater goal"),
        "3": ("random", "Random start to underwater goal"),
        "4": ("train", "Train PPO model"),
        "5": ("quit", "Quit"),
    }

    while True:
        print("\nMenu:")
        for number, (_, label) in menu_options.items():
            print(f"  {number}. {label}")

        command = input("Select option: ").strip()
        selected_action = menu_options.get(command, (None, None))[0]
        choice = (
            selected_action
            if selected_action in destination_positions
            else parse_destination(command)
        )

        if choice is not None or selected_action == "random":
            if checkpoint_loaded:
                pass
            elif os.path.exists(checkpoint_path):
                checkpoint = torch.load(
                    checkpoint_path, map_location=device, weights_only=False
                )
                try:
                    model.load_state_dict(checkpoint["model_state_dict"])
                except RuntimeError:
                    print(
                        "The saved checkpoint uses the old observation layout. "
                        "Run option 4 to train the wall-aware PPO model first."
                    )
                    continue
                checkpoint_loaded = True
                evaluation = checkpoint.get("evaluation", {})
                print(
                    "Loaded best PPO model"
                    f" (evaluation success: {evaluation.get('success_rate', 0):.1%})."
                )
            else:
                print(
                    "No PPO checkpoint found; rendering the current in-memory model."
                )
            if selected_action == "random":
                destination = goal_position
                initial_state = None
                print("Starting randomly and navigating to the goal...")
            else:
                destination = destination_positions[choice]
                initial_state = current_state
                print(
                    f"Navigating to {choice} at "
                    f"{format_position(destination)}..."
                )
            _, current_state = render_trained_policy(
                model=model,
                goal_position=destination,
                walls=walls,
                outside_walls=outside_walls,
                mujoco_model=mujoco_model,
                mujoco_data=mujoco_data,
                epsilon=epsilon,
                spawn_margin=spawn_margin,
                initial_state=initial_state,
                interactive=True,
                destination_positions=destination_positions,
                device=device,
            )

        elif selected_action == "train":
            print("Training the network...")
            num_training_episodes = 1200
            model = ppo_training(
                model=model,
                goal_position=goal_position,
                walls=walls,
                outside_walls=outside_walls,
                num_episodes=num_training_episodes,
                batch_size=64,
                mujoco_model=mujoco_model,
                max_steps=4800,
                gamma=0.99,
                plot_directory=os.path.join(
                    project_root, "report", "images", "fluid"
                ),
                epsilon=epsilon,
                checkpoint_path=checkpoint_path,
                device=device  # pass device here
            )

        elif selected_action == "quit" or command.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        else:
            print("Choose 1, 2, 3, 4, or 5.")
