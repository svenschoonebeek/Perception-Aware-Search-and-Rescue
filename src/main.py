import logging
import argparse
import numpy as np

from .simulator import Simulator, MapConfig, visualize_grid_gen
from .agents.mpc import MpcAgent
from .agents.task_mpc import TaskMpcAgent
from .agents.random import RandAgent
from .agent import Agent
from .visualization import Visualizer
from .supervisor import Supervisor
from .util import *

import matplotlib.pyplot as plt

WIDTH = 60
HEIGHT = 60

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run the search and rescue simulation.")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging")
    args = parser.parse_args()
    setup_logging(args.debug)
    warmup_numba_compilers()
    
    sim = Simulator(WIDTH, HEIGHT)
    config = MapConfig(num_rooms=15, num_victims=10, min_room_length=4, min_room_width=4, max_room_length=8, max_room_width=8, max_tunnel_thickness=3, min_tunnel_thickness=2
                       , initial_fire_points=0, room_vulnerability_probability=0.1, tunnel_vulnerability_probability=0.1,)
    # config = MapConfig(num_rooms=12, num_victims=10, room_vulnerability_probability=0.1, room_vulnerability_severity=0.2, min_room_length=5, min_room_width=5, max_room_length=10, max_room_width=10, min_tunnel_thickness=2,max_tunnel_thickness=3, fire_spread_rate=0.02, fire_duration=100, initial_fire_points=0)

    sim.generate_ground_truth(config,178470943) # 281871615 224833043 48617497

    # Saved seeds: 182840517, 210577037, 335492940, 95007438

    agent1 = TaskMpcAgent("tasker1", 0, 0, WIDTH, HEIGHT, 0.01, 0.9, 4, False)
    # agent1 = MpcAgent("mpc1", 0, 0, WIDTH, HEIGHT, 0.05, 0.9, 4, False)
    # agent1 = RandAgent("randy", 0, 0, WIDTH, HEIGHT, 0.01, 0.9, 7, False)
    sim.add_agent(agent1)

    agent2 = TaskMpcAgent("tasker2", 0, 1, WIDTH, HEIGHT, 0.01, 0.9, 4, False)
    # agent2 = MpcAgent("mpc2", 0, 1, WIDTH, HEIGHT, 0.01, 0.9, 3, False)
    # # agent2 = RandAgent("rand2", 0, 1, WIDTH, HEIGHT, 0.01, 0.9, 7, False)
    sim.add_agent(agent2)

    agent3 = TaskMpcAgent("tasker3", 0, 2, WIDTH, HEIGHT, 0.01, 0.9, 4, False)
    # agent3 = MpcAgent("mpc3", 0, 2, WIDTH, HEIGHT, 0.01, 0.9, 4, False)
    sim.add_agent(agent3)

    supervisor = Supervisor(WIDTH, HEIGHT, 0.01, 8, False)
    sim.set_supervisor(supervisor)

    # visualize_grid_gen(sim.ground_truth.traversability, sim.ground_truth.agents, sim.ground_truth.victims, sim.ground_truth.vulnerability, sim.ground_truth.fire)

    logger.info("Running Simulation steps...")
    history = sim.run(300)

    average_overlap = 0
    for i in range(0, len(sim.agents)):
        for j in range(i+1, len(sim.agents)):
            agent_i = sim.agents[i]
            agent_j = sim.agents[j]
            agent_i_explored = agent_i.explored
            agent_j_explored = agent_j.explored
            overlap = np.sum(np.logical_and(agent_i_explored, agent_j_explored))
            agent_i_percentage_overlap = overlap / agent_i_explored.sum()
            agent_j_percentage_overlap = overlap / agent_j_explored.sum()
            logger.info(f"Overlap in explored area between {agent_i.name} and {agent_j.name}: {overlap} cells")
            logger.info(f"Percentage overlap - {agent_i.name}: {agent_i_percentage_overlap:.2%}, {agent_j.name}: {agent_j_percentage_overlap:.2%}")
            average_overlap += (agent_i_percentage_overlap + agent_j_percentage_overlap) / 2 / (len(sim.agents))

    logger.info(f"Average overlap: {average_overlap}")
    logger.info(f"Total explored cells: {np.sum(supervisor.global_explored.sum())}")

    logger.info("Launching Visualizer...")
    viz = Visualizer(history, 800, 800)
    # viz = Visualizer.from_file('saved_runs\\private\\stuck_at_vulnerable.pkl', 800, 800)
    viz.run()

    # plot_total_exploration(sim.metrics.history)


if __name__ == "__main__":
    main()