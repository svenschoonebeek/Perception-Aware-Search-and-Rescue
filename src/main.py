import logging
import argparse
import numpy as np

from .simulator import Simulator, MapConfig, visualize_grid_gen
from .agents.mpc import MpcAgent
from .agents.task_mpc import TaskMpcAgent
from .agents.random import RandAgent
from .agents.rescue import RescueAgent
from .agent import Agent
from .visualization import Visualizer
from .supervisor import Supervisor
from .util import *
from .constants import Capabilities, StoppingCondition

WIDTH = 25
HEIGHT = 25

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run the search and rescue simulation.")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging")
    args = parser.parse_args()
    setup_logging(args.debug)
    warmup_numba_compilers()
    
    sim = Simulator(WIDTH, HEIGHT, StoppingCondition.ALL_VICTIMS_RESCUED)
    # config = MapConfig(num_rooms=12, num_victims=10, min_room_length=4, min_room_width=4, max_room_length=8, max_room_width=8, max_tunnel_thickness=3, min_tunnel_thickness=2
    #                    , initial_fire_points=1, room_vulnerability_probability=0.1, tunnel_vulnerability_probability=0.1, fire_spread_rate=0.01, fire_duration=100)
    config = MapConfig(num_rooms=5, num_victims=2, room_vulnerability_probability=0.0, room_vulnerability_severity=0.2, tunnel_vulnerability_probability=0.0, min_room_length=5, min_room_width=5, max_room_length=10, max_room_width=10, min_tunnel_thickness=2,max_tunnel_thickness=3, fire_spread_rate=0.02, fire_duration=100, initial_fire_points=0)
    # config = MapConfig(num_rooms=3, num_victims=1, room_vulnerability_probability=0.0, room_vulnerability_severity=0.0, initial_fire_points=0, min_tunnel_thickness=2, max_tunnel_thickness=3, tunnel_vulnerability_probability=0.0, tunnel_vulnerability_severity=0.0)
    sim.generate_ground_truth(config,) # 281871615 224833043 48617497
    
    # Saved seeds: 182840517, 210577037, 335492940, 95007438

    # agent1 = TaskMpcAgent("tasker1", 0, 0, WIDTH, HEIGHT, 0.01, 0.9, 4, False, Capabilities(exploration=1.0, rescuing=0.0))
    agent1 = RescueAgent("rescuer1", 0, 0, WIDTH, HEIGHT, 0.01, 0.9, 4, False, None, Capabilities(exploration=1.0, rescuing=1.0))
    # agent1 = MpcAgent("mpc1", 0, 0, WIDTH, HEIGHT, 0.01, 0.9, 4, False)
    # agent1 = RandAgent("randy", 0, 0, WIDTH, HEIGHT, 0.01, 0.9, 7, False)
    sim.add_agent(agent1,)

    # agent2 = TaskMpcAgent("tasker2", 0, 1, WIDTH, HEIGHT, 0.01, 0.9, 4, False, Capabilities(exploration=1.0, rescuing=1.0))
    agent2 = RescueAgent("rescuer2", 0, 1, WIDTH, HEIGHT, 0.01, 0.9, 4, False, None, Capabilities(exploration=1.0, rescuing=1.0))
    # agent2 = MpcAgent("mpc2", 0, 1, WIDTH, HEIGHT, 0.01, 0.9, 4, False)
    # agent2 = RandAgent("rand2", 0, 1, WIDTH, HEIGHT, 0.01, 0.9, 7, False)
    sim.add_agent(agent2)

    # agent3 = TaskMpcAgent("tasker3", 0, 2, WIDTH, HEIGHT, 0.01, 0.9, 4, False, Capabilities(exploration=1.0, rescuing=0.0))
    agent3 = RescueAgent("rescuer3", 0, 2, WIDTH, HEIGHT, 0.01, 0.9, 4, False, None, Capabilities(exploration=1.0, rescuing=1.0))
    # agent3 = MpcAgent("mpc3", 0, 2, WIDTH, HEIGHT, 0.01, 0.9, 4, False)
    sim.add_agent(agent3)

    # agent4 = TaskMpcAgent("tasker4", 1, 0, WIDTH, HEIGHT, 0.01, 0.9, 4, False, Capabilities(exploration=1.0, rescuing=0.0))
    agent4 = RescueAgent("rescuer4", 1, 0, WIDTH, HEIGHT, 0.01, 0.9, 4, False, None, Capabilities(exploration=1.0, rescuing=1.0))
    # agent4 = MpcAgent("mpc4", 1, 0, WIDTH, HEIGHT, 0.01, 0.9, 4, False)
    sim.add_agent(agent4)

    supervisor = Supervisor(WIDTH, HEIGHT, 0.01, 8, True)
    sim.set_supervisor(supervisor)

    # visualize_grid_gen(sim.ground_truth.traversability, sim.ground_truth.agents, sim.ground_truth.victims, sim.ground_truth.vulnerability, sim.ground_truth.fire)

    logger.info("Running Simulation steps...")
    history = sim.run(200)

    logger.info("Launching Visualizer...")
    viz = Visualizer(history, 800, 800)
    # viz = Visualizer.from_file('saved_runs\\private\\3_agents_nice_distribution.pkl', 800, 800)
    # viz = Visualizer.from_file('saved_runs\\private\\sim_20260512_151550_300steps.pkl', 800, 800)
    viz.run()

    logger.info(f"final metrics: {sim.metrics.history[-1]}")

    # plot_all(sim.metrics.history, sim.supervisor is not None)


if __name__ == "__main__":
    main()