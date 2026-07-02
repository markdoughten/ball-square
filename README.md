# Collision-Free PPO Navigation

A MuJoCo ball-navigation environment using PPO, RRT path planning, PID imitation
pretraining, and a safety controller for collision-free obstacle avoidance.

## Run

```powershell
cd src
..\.venv\Scripts\python.exe movement.py
```

- Option 1 renders the best local checkpoint from a random start.
- Option 2 trains PPO and writes `src/best_ppo_model.pt`.
- Option 3 renders the best local checkpoint from the fixed `(0, 0)` start.

Model checkpoints are generated artifacts and are intentionally ignored by Git.
