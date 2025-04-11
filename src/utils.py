import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from colored import fg, attr
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from rl.env import CVRPEnv
from src.rl.q_learning import QLearningAgent
from src.utils import parse_vrp_file, parse_solution_file, plot_route


def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def parse_vrp_file(filepath):
    """Parse a VRP instance file and return problem data."""
    coords = {}
    demands = {}
    capacity = None
    depot_id = 1  # Depot is always node 1 in these instances

    with open(filepath, "r") as file:
        lines = file.readlines()

    # Parse file line by line
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("CAPACITY"):
            capacity = int(line.split()[-1])

        elif line.startswith("NODE_COORD_SECTION"):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("DEMAND_SECTION"):
                parts = lines[i].strip().split()
                if len(parts) == 3:  # Valid coordinate line
                    node_id = int(parts[0])
                    coords[node_id] = (float(parts[1]), float(parts[2]))
                i += 1
            i -= 1  # Back up one line since we'll increment at end of loop

        elif line.startswith("DEMAND_SECTION"):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("DEPOT_SECTION"):
                parts = lines[i].strip().split()
                if len(parts) == 2:  # Valid demand line
                    node_id = int(parts[0])
                    demands[node_id] = float(parts[1])
                i += 1
            i -= 1  # Back up one line since we'll increment at end of loop

        i += 1

    if not coords or not demands or capacity is None:
        raise ValueError("Failed to parse all required data from VRP file")

    return coords, demands, capacity, depot_id


def parse_solution_file(filepath):
    """Parse a solution file and return the optimal route and distance."""
    routes = []
    total_distance = None

    with open(filepath, "r") as file:
        content = file.read()

    # Split content into lines and process each line
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        if line.startswith("Route"):
            # Extract numbers after the colon
            numbers = line.split(":")[1].strip()
            # Add depot at start and end of each route
            route = [int(x) for x in numbers.split()]
            routes.append(route)
        elif line.startswith("cost"):  # Get cost line
            cost_str = line.split()[1]  # Get the number after "cost"
            total_distance = float(cost_str)

    return routes, total_distance


def plot_route(
    coords,
    route,
    title,
    ax,
    color="blue",
    total_distance=None,
    optimal_distance=None,
    depot_id=None,
    animated=False,
    current_step=None,
):
    """Plot a route with optional animation support."""
    # Define a list of distinct colors for different routes
    route_colors = ["blue", "red", "orange", "green", "purple"]

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    if not animated:
        # Split route into individual routes (separated by depot)
        routes = []
        current_route = []
        for node in route:
            if node == depot_id:
                if current_route:  # End of a route
                    current_route.append(node)  # Add closing depot
                    routes.append(current_route)
                    current_route = [node]  # Start new route with depot
                else:  # Start of first route
                    current_route = [node]
            else:
                current_route.append(node)
        # Add the last route if it exists
        if current_route and len(current_route) > 1:
            if current_route[-1] != depot_id:
                current_route.append(depot_id)
            routes.append(current_route)

        # Plot each route with a different color and style
        for i, r in enumerate(routes):
            # Extract x and y coordinates for this route
            x = [coords[node][0] for node in r]
            y = [coords[node][1] for node in r]

            # Use dashed lines for depot connections
            linestyle = "--" if depot_id in r else "-"

            # Plot route with lines and markers
            ax.plot(
                x,
                y,
                color=route_colors[i % len(route_colors)],
                marker="o",
                linestyle=linestyle,
                label=f"Route {i+1}",
                zorder=2,
            )

            # Add arrows to show direction
            for j in range(len(x) - 1):
                ax.annotate(
                    "",
                    xy=(x[j + 1], y[j + 1]),
                    xytext=(x[j], y[j]),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=route_colors[i % len(route_colors)],
                        lw=1.5,
                    ),
                )

    else:
        # For animation, plot up to current_step
        if current_step is not None and current_step < len(route):
            x = [coords[node][0] for node in route[: current_step + 1]]
            y = [coords[node][1] for node in route[: current_step + 1]]
            ax.plot(x, y, marker="o", color=color)
            # Add a different marker for the current node
            ax.plot(x[-1], y[-1], marker="*", color="red", markersize=12)

    # Plot depot with a different marker
    if depot_id is not None:
        ax.plot(
            coords[depot_id][0],
            coords[depot_id][1],
            marker="s",
            color="black",
            markersize=10,
            zorder=3,
        )

    # Add node labels
    for node in route:
        ax.text(
            coords[node][0],
            coords[node][1],
            str(node),
            fontsize=9,
            ha="right",
            va="bottom",
        )

    ax.set_title(title)

    # Add legend
    if not animated:
        ax.legend(loc="upper right")

    # Create text box content
    text_content = []
    if total_distance is not None:
        text_content.append(f"Total Distance: {total_distance:.2f}")
    if optimal_distance is not None:
        text_content.append(f"Optimal Distance: {optimal_distance:.2f}")

    # Add text box
    if text_content:
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.05,
            0.95,
            "\n".join(text_content),
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )


def create_route_animation(coords, route_history, depot_id):
    """Create an animation of route construction."""
    fig, ax = plt.subplots(figsize=(8, 6))

    def init():
        ax.clear()
        return (ax,)

    def update(frame):
        ax.clear()
        route = route_history[frame]
        plot_route(
            coords,
            route,
            "Route Construction",
            ax,
            depot_id=depot_id,
            animated=True,
            current_step=len(route) - 1,
        )
        return (ax,)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(route_history),
        init_func=init,
        blit=True,
        repeat=False,
        interval=500,
    )

    return anim
