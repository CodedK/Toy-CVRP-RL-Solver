# Toy CVRP RL Solver

A Reinforcement Learning-based solver for the Capacitated Vehicle Routing Problem (CVRP) using Q-Learning.

## Project Structure

```
Toy-CVRP-RL-Solver/
├── data/
│   └── vrp/                    # VRP instance files
├── src/
│   ├── __init__.py
│   ├── main.py                # Main script to run the RL solver
│   ├── utils.py               # Utility functions (parsing, plotting, etc.)
│   └── rl/                    # Reinforcement Learning components
│       ├── __init__.py
│       ├── env.py             # CVRP environment implementation
│       └── q_learning.py      # Q-Learning agent implementation
└── tests/                     # Test files (to be implemented)
```

## Features

- Q-Learning based solution for CVRP
- Visualization of learned routes with distance metrics
- Support for standard VRP file format
- Modular design for easy extension

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Toy-CVRP-RL-Solver.git
cd Toy-CVRP-RL-Solver
```

2. Install the package in development mode:
```bash
pip install -e .
```

## Usage

1. Place your VRP instance file in the `data/vrp/` directory
2. Run the solver:
```bash
python -m src.main
```

The script will:
- Load and parse the VRP instance
- Train the Q-Learning agent
- Generate and visualize the solution
- Save the visualization as `cvrp_rl_comparison.png`

## Visualization

The visualization includes:
- Learned route plot
- Total distance
- Individual route distances
- Comparison with optimal solution (when available)

## Example

For the A-n32-k5 instance:
- Optimal distance: 784.0
- The solver will attempt to find a solution close to this value

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
