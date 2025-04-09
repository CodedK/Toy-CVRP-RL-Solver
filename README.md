# Toy CVRP RL Solver

A minimal reinforcement learning environment and Q-learning agent to solve Capacitated Vehicle Routing Problem (CVRP) instances using `.vrp` files (e.g., A-n32-k5.vrp format).

## Features
- Custom parser for `.vrp` files
- CVRP environment (CVRPEnv) with capacity constraints
- Q-learning agent with tabular updates
- Route visualization with comparison to the optimal solution

## Requirements
```bash
pip install matplotlib numpy
```

## How to Run
```bash
python main.py
```

Make sure you place your `.vrp` file (e.g., `A-n32-k5.vrp`) inside the same directory or update the filename in `main.py`.

## Files
- `main.py`: Main entry point to run training and visualization
- `env.py`: CVRPEnv environment definition
- `q_learning.py`: Tabular Q-learning implementation
- `utils.py`: Utilities like the `.vrp` parser and plotting tools
