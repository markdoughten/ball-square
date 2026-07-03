# Collision-Free PPO Navigation

A MuJoCo ball-navigation environment using PPO, RRT path planning, PID imitation
pretraining, and a safety controller for collision-free obstacle avoidance.
Interactive navigation uses 3D RRT waypoints. The planner includes obstacle
height, ball clearance, and water depth, allowing a surface-floating robot to
travel over an obstacle when the waterline provides sufficient clearance.
Routes are sampled freely throughout the water volume rather than being forced
to cross at the surface and dive vertically. The single underwater goal uses a
strict 1 cm tolerance.
The PPO actor controls X, Y, and Z force. Its observation contains world-space
position `(x, y, z)`, velocity `(vx, vy, vz)`, and 3D waypoint error, giving a
9-input/3-output network. Existing 2D checkpoints are incompatible; run the
`train` command to create `models/best_ppo_model.pt`.
The normalized policy outputs drive control-limited motors with 15 N horizontal
authority and 80 N vertical authority for diving against buoyancy. Training
uses 0.01 s physics substeps, progress/stagnation
rewards, a small motor-energy penalty, and an emergency velocity-braking shield.

## Run

```powershell
cd src
..\.venv\Scripts\python.exe app.py
```

At the command prompt, type a natural-language instruction containing `goal` or
`start` (for example, `go to the goal` or `return home`). The robot preserves its
position between commands and plans a path to the single underwater goal
`(0.9, -0.22, 0.06)` or start `(0, 0, 0.15)`. There is no surface goal. While
the simulation window is running, type `start` or `goal` in the
terminal and press Enter to redirect the robot immediately from its current
position. The `underwater` command sends it to `(0.9, -0.22, 0.06)` inside the
translucent blue tank, where density, viscosity, current, pressure drag, partial
submersion, and added-mass forces are applied. Type `quit` to close the simulation. Other main-menu commands are
`train`, `random`, and `quit`.
The tank is filled from the floor to `z = 0.15`, above the central obstacle
(`z = 0.10`) and below the outer wall tops (`z = 0.20`).
The water velocity and pressure evolve through a 3D incompressible
Navier–Stokes solver with semi-Lagrangian advection, viscous diffusion,
pressure projection, inlet/outlet flow, a free-surface boundary, and no-slip
floor and obstacle boundaries. Each fluid cell stores `(u, v, w)` velocity and
pressure. The
robot has a vertical joint and receives Archimedes buoyancy
`F_b = density * gravity * submerged_volume` plus vertical drag.
Two-way immersed-boundary coupling makes the ball enforce a moving no-slip
boundary and returns the equal-and-opposite drag/added-mass impulse to nearby
fluid cells, producing a wake. Fixed walls and the center obstacle exchange
momentum through their no-slip constraints.
With the simulation window focused, use the arrow keys to change the current
direction. The inlet turns gradually at 60 degrees per second while preserving
flow speed, and the Navier–Stokes field evolves continuously.
Time-varying divergence-free eddies and vorticity confinement add turbulence.
Buoyancy is calibrated so the robot rises when fully submerged and settles at
the water surface with half its spherical volume submerged; vertical drag
damps oscillation around the waterline.

Simulation camera controls: use the mouse wheel to zoom, left-drag to orbit,
Shift+left-drag for horizontal orbit, right-drag to pan horizontally, and
middle-drag to pan vertically.

Model checkpoints are generated artifacts and are intentionally ignored by Git.
