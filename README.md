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
.\.venv\Scripts\python.exe src/app.py
```

At the command prompt, choose a numbered menu option: navigate to the surface
start, navigate to the underwater goal, start a random trial, train the PPO
model, or quit. The robot preserves its position between navigation commands
and plans a path to the single underwater goal `(0.9, -0.22, 0.06)` or start
`(0, 0, 0.15)`. There is no surface goal. While the simulation window is
running, type `1` to redirect to start, `2` to redirect to the underwater goal,
or `5` to quit. With the simulation window focused, hold W/S to move the
active goal along the long side, A/D to move side to side, and arrow Up/Down to
change its depth. Completed runs
return to the main menu. The underwater goal is inside the translucent blue
tank, where density, viscosity, current, pressure drag, partial submersion, and
added-mass forces are applied.
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
The inlet current remains active while the fluid field evolves continuously.
Time-varying divergence-free eddies and vorticity confinement add turbulence.
Buoyancy is calibrated so the robot rises when fully submerged and settles at
the water surface with half its spherical volume submerged; vertical drag
damps oscillation around the waterline.

Simulation camera controls: use the mouse wheel to zoom, left-drag to orbit,
Shift+left-drag for horizontal orbit, right-drag to pan horizontally, and
middle-drag to pan vertically.

Model checkpoints are generated artifacts and are intentionally ignored by Git.
