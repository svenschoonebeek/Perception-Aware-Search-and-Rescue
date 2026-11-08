import logging
import argparse

from .simulator import Simulator, MapConfig, visualize_grid_gen
from .agents.mpc import MpcAgent
from .agents.tmpc import TmpcAgent
from .agents.random import RandAgent
from .agent import Agent
from .visualization import Visualizer

import matplotlib.pyplot as plt

WIDTH = 35
HEIGHT = 35

logger = logging.getLogger(__name__)

def setup_logging(is_debug: bool):
    """Configures the global logging level and format."""
    log_level = logging.DEBUG if is_debug else logging.INFO

    logging.basicConfig(
        level=log_level,
        format='%(levelname)s - %(message)s',
    )

def main():
    parser = argparse.ArgumentParser(description="Run the search and rescue simulation.")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging")
    args = parser.parse_args()
    setup_logging(args.debug)

    sim = Simulator(WIDTH, HEIGHT)
    config = MapConfig(num_rooms=10, num_victims=15, min_room_length=5, min_room_width=5, max_room_length=10, max_room_width=10, max_tunnel_thickness=2, fire_spread_rate=0.05, fire_duration=10, initial_fire_points=2)
    sim.generate_ground_truth(config)

    # Saved seeds: 182840517, 210577037, 335492940, 95007438

    # agent1 = RandAgent("randy", 0, 0, WIDTH, HEIGHT, 0.01, 0.9, 7, False)
    # agent1 = MpcAgent("mpc", 0, 0, WIDTH, HEIGHT, 0.01, 0.9, 4, True)
    agent1 = TmpcAgent("tmpc", 1, 1, WIDTH, HEIGHT, 0.01, 0.9, 4, False)
    sim.add_agent(agent1)

    visualize_grid_gen(sim.ground_truth.traversability, sim.ground_truth.agents, 
                       sim.ground_truth.victims, sim.ground_truth.vulnerability,
                       sim.ground_truth.fire)

    print("Running Simulation steps...")
    history = sim.run(500)

    # plt.matshow(agent1._closest_unexplored())
    # plt.show()

    logger.info("Launching Visualizer...")
    viz = Visualizer(history, 800, 800)
    viz.run()

if __name__ == "__main__":
    main()