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

WIDTH = 45
HEIGHT = 45

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run the search and rescue simulation.")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging")
    args = parser.parse_args()
    setup_logging(args.debug)
    warmup_numba_compilers()
    
    sim = Simulator(WIDTH, HEIGHT, StoppingCondition.ALL_VICTIMS_RESCUED)
    config = MapConfig(num_rooms=10, num_victims=5, min_room_length=4, min_room_width=4, max_room_length=8, max_room_width=8, max_tunnel_thickness=2, min_tunnel_thickness=2
                       , initial_fire_points=0, room_vulnerability_probability=0.0, room_vulnerability_severity=0.2, tunnel_vulnerability_probability=0.0, fire_spread_rate=0.01, fire_duration=100)
    # config = MapConfig(num_rooms=15, num_victims=10, room_vulnerability_probability=0.0, room_vulnerability_severity=0.2, tunnel_vulnerability_probability=0.0, min_room_length=5, min_room_width=5, max_room_length=10, max_room_width=10, min_tunnel_thickness=2,max_tunnel_thickness=3, fire_spread_rate=0.02, fire_duration=100, initial_fire_points=0)
    # config = MapConfig(num_rooms=3, num_victims=1, room_vulnerability_probability=0.0, room_vulnerability_severity=0.0, initial_fire_points=0, min_tunnel_thickness=2, max_tunnel_thickness=3, tunnel_vulnerability_probability=0.0, tunnel_vulnerability_severity=0.0)
    sim.generate_ground_truth(config,) # 281871615 224833043 48617497
    # 509072484
    # Saved seeds: 182840517, 210577037, 335492940, 95007438

    agent1 = RescueAgent("rescuer1", 0, 0, WIDTH, HEIGHT, 0.01, 0.9, 4, False, None, Capabilities(exploration=1.0, rescuing=1.0))
    sim.add_agent(agent1,)

    agent2 = RescueAgent("rescuer2", 0, 1, WIDTH, HEIGHT, 0.01, 0.9, 4, False, None, Capabilities(exploration=1.0, rescuing=1.0))
    sim.add_agent(agent2)

    # agent3 = TaskMpcAgent("tasker1", 0, 2, WIDTH, HEIGHT, 0.01, 0.9, 4, False, Capabilities(exploration=1.0, rescuing=0.0))
    agent3 = RescueAgent("rescuer3", 0, 2, WIDTH, HEIGHT, 0.01, 0.9, 4, False, None, Capabilities(exploration=1.0, rescuing=1.0))
    sim.add_agent(agent3)

    agent4 = TaskMpcAgent("tasker2", 1, 0, WIDTH, HEIGHT, 0.01, 0.9, 4, False, Capabilities(exploration=1.0, rescuing=0.0))
    # agent4 = RescueAgent("rescuer4", 1, 0, WIDTH, HEIGHT, 0.01, 0.9, 4, False, None, Capabilities(exploration=1.0, rescuing=1.0))
    sim.add_agent(agent4)

    agent5 = TaskMpcAgent("tasker1", 1, 1, WIDTH, HEIGHT, 0.01, 0.9, 4, False, Capabilities(exploration=1.0, rescuing=0.0))
    sim.add_agent(agent5)

    # agent6 = TaskMpcAgent("tasker2", 1, 2, WIDTH, HEIGHT, 0.01, 0.9, 4, False, Capabilities(exploration=1.0, rescuing=0.0))
    # sim.add_agent(agent6)

    supervisor = Supervisor(WIDTH, HEIGHT, 0.01, 8, True)
    sim.set_supervisor(supervisor)


    logger.info("Running Simulation steps...")
    history = sim.run(300)

    logger.info("Launching Visualizer...")
    viz = Visualizer(history, 800, 800)
    viz.run()

    logger.info(f"final metrics: {sim.metrics.history[-1]}")


if __name__ == "__main__":
    main()