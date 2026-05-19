import logging
import argparse
import pandas as pd

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

    config = MapConfig(
        num_rooms=7,
        min_tunnel_thickness=2,
        max_tunnel_thickness=2,
        num_victims=10,
        fire_spread_rate=0.02,
        fire_duration=20,
        initial_fire_points=2
    )
    # Set up pd dataframe with list of columns, rows to be appended 
    metrics = [
        'outcome',
        'victims_found', 
        'duration', 
        'avg_time_per_victim', 
        'wall_collisions', 
        'victims_collisions',
        'wait_actions',
        'area_explored',
        'infeasible_states'
        ]
    df = pd.DataFrame(columns=metrics)

    number_of_sims = 100
    max_steps = 300
    agent = MpcAgent("mpc", 1, 1, WIDTH, HEIGHT, 0.05, 0.9, 4, True)
    # agent = TmpcAgent("tmpc", 1, 1, WIDTH, HEIGHT, 0.05, 0.9, 4, True)

    for i in range(number_of_sims):

        sim = Simulator(WIDTH, HEIGHT)
        agent_instance = agent.copy()
        sim.generate_ground_truth(config)
        sim.add_agent(agent_instance)

        if i == 0:
            visualize_grid_gen(sim.ground_truth.traversability, sim.ground_truth.agents, sim.ground_truth.victims, sim.ground_truth.vulnerability, sim.ground_truth.fire)
        sim.run(max_steps)

        outcome = sim.metrics.outcome.name
        victims_found = sim.metrics.victims_found
        duration = sim.metrics.steps_taken
        avg_time_per_victim = duration / victims_found
        wall_collisions = sim.metrics.terrain_collisions[agent_instance]
        victim_collisions = sim.metrics.victim_collisions[agent_instance]
        wait_actions = sim.metrics.wait_actions[agent_instance]
        area_explored = sim.metrics.area_explored[agent_instance]
        infeasible_states = sim.metrics.infeasible_states[agent_instance]

        df.loc[i] = [outcome, victims_found, duration, avg_time_per_victim, wall_collisions, victim_collisions, wait_actions, area_explored, infeasible_states]
        logger.info("Run %d complete in %d steps: %s", i + 1, duration, outcome)
    
    df.to_csv("results.csv")


if __name__ == "__main__":
    main()