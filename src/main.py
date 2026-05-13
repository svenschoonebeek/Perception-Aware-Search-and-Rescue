import logging
import argparse

from .simulator import Simulator, MapConfig, visualize_grid_gen
from .agents.mpc import MpcAgent
from .agents.task_mpc import TaskMpcAgent
from .agents.random import RandAgent
from .agent import Agent
from .visualization import Visualizer
from .supervisor import Supervisor

WIDTH = 40
HEIGHT = 40

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
    config = MapConfig(num_rooms=7, num_victims=5, min_room_length=4, min_room_width=4, max_room_length=10, max_room_width=10, max_tunnel_thickness=2, min_tunnel_thickness=2
                       , initial_fire_points=0)
    sim.generate_ground_truth(config,450215754) # 809315671

    # Saved seeds: 182840517 (181 steps); 

    agent1 = TaskMpcAgent("tasker1", 0, 0, WIDTH, HEIGHT, 0.01, 0.9, 3, False)
    # agent1 = MpcAgent("mpc", 0, 0, WIDTH, HEIGHT, 0.05, 0.9, 4, False)
    # agent1 = RandAgent("randy", 0, 0, WIDTH, HEIGHT, 0.01, 0.9, 7, False)
    sim.add_agent(agent1)

    agent2 = TaskMpcAgent("tasker2", 0, 1, WIDTH, HEIGHT, 0.01, 0.9, 3, False)
    # # agent2 = MpcAgent("tasker2", 0, 1, WIDTH, HEIGHT, 0.01, 0.9, 3, False)
    # # agent2 = RandAgent("rand2", 0, 1, WIDTH, HEIGHT, 0.01, 0.9, 7, False)
    sim.add_agent(agent2)

    # agent3 = TaskMpcAgent("tasker3", 0, 2, WIDTH, HEIGHT, 0.01, 0.9, 3, False)
    # sim.add_agent(agent3)

    supervisor = Supervisor(WIDTH, HEIGHT, 0.05)
    sim.set_supervisor(supervisor)

    # visualize_grid_gen(sim.ground_truth.traversability, sim.ground_truth.agents, 
    #                    sim.ground_truth.victims, sim.ground_truth.vulnerability,
    #                    sim.ground_truth.fire)

    logger.info("Running Simulation steps...")
    history = sim.run(100) 

    logger.info("Launching Visualizer...")
    viz = Visualizer(history, 800, 800)
    viz.run()

if __name__ == "__main__":
    main()