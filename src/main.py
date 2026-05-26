import logging
import argparse

from .simulator import Simulator, MapConfig, visualize_grid_gen
from .agents.mpc import MpcAgent
from .agents.tmpc import TmpcAgent
from .agents.sdd_tmpc import SddTmpcAgent
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
        num_rooms=10, 
        num_victims=15, 
        min_room_length=5, 
        min_room_width=5, 
        max_room_length=10, 
        max_room_width=10, 
        min_tunnel_thickness=2,
        max_tunnel_thickness=2, 
        fire_spread_rate=0.05, 
        fire_duration=15, 
        initial_fire_points=2
        )

    randy   = RandAgent("randy", 1, 1, WIDTH, HEIGHT, 0.01, 0.35, 4, True)
    mpc     = MpcAgent("mpc", 1, 1, WIDTH, HEIGHT, 0.01, 0.35, 4, True)
    tmpc    = TmpcAgent("tmpc", 1, 1, WIDTH, HEIGHT, 0.01, 0.35, 4, True)
    sddtmpc = SddTmpcAgent("sdd-tmpc", 1, 1, WIDTH, HEIGHT, 0.01, 0.35, 4, True)

    # sim = Simulator(WIDTH, HEIGHT)
    # sim.generate_ground_truth(config)
    # visualize_grid_gen(sim.ground_truth.traversability, sim.ground_truth.agents, 
    #                    sim.ground_truth.victims, sim.ground_truth.vulnerability,
    #                    sim.ground_truth.fire)
    # return

    seed = 435132393    # Saved seeds: 182840517, 210577037, 335492940, 95007438

    sim_rand = Simulator(WIDTH, HEIGHT)
    sim_mpc = Simulator(WIDTH, HEIGHT)
    sim_tmpc = Simulator(WIDTH, HEIGHT)
    sim_sddtmpc = Simulator(WIDTH, HEIGHT)
    
    sim_rand.generate_ground_truth(config, seed)
    sim_rand.add_agent(randy)
    history = sim_rand.run(500)
    outcome = sim_rand.metrics.outcome.name
    victims = sim_rand.metrics.victims_found
    duration = sim_rand.metrics.steps_taken
    wall_collisions = sim_rand.metrics.terrain_collisions[randy]
    victim_collisions = sim_rand.metrics.victim_collisions[randy]
    fire_collisions = sim_rand.metrics.fire_collisions[randy]
    logger.info("Randy took %d steps, found %d victims, and hit %d walls, %d victims, and %d fires: %s", duration, victims, wall_collisions, victim_collisions, fire_collisions, outcome)
    Visualizer(history, 1200, 1200).run()

    sim_mpc.generate_ground_truth(config, seed)
    sim_mpc.add_agent(mpc)
    history = sim_mpc.run(500)
    outcome = sim_mpc.metrics.outcome.name
    victims = sim_mpc.metrics.victims_found
    duration = sim_mpc.metrics.steps_taken
    wall_collisions = sim_mpc.metrics.terrain_collisions[mpc]
    victim_collisions = sim_mpc.metrics.victim_collisions[mpc]
    fire_collisions = sim_mpc.metrics.fire_collisions[mpc]
    logger.info("MPC took %d steps, found %d victims, and hit %d walls, %d victims, and %d fires: %s", duration, victims, wall_collisions, victim_collisions, fire_collisions, outcome)
    Visualizer(history, 1200, 1200).run()

    sim_tmpc.generate_ground_truth(config, seed)
    sim_tmpc.add_agent(tmpc)
    history = sim_tmpc.run(500)
    outcome = sim_tmpc.metrics.outcome.name
    victims = sim_tmpc.metrics.victims_found
    duration = sim_tmpc.metrics.steps_taken
    wall_collisions = sim_tmpc.metrics.terrain_collisions[tmpc]
    victim_collisions = sim_tmpc.metrics.victim_collisions[tmpc]
    fire_collisions = sim_tmpc.metrics.fire_collisions[tmpc]
    logger.info("TMPC took %d steps, found %d victims, and hit %d walls, %d victims, and %d fires: %s", duration, victims, wall_collisions, victim_collisions, fire_collisions, outcome)
    Visualizer(history, 1200, 1200).run()
    
    sim_sddtmpc.generate_ground_truth(config, seed)
    sim_sddtmpc.add_agent(sddtmpc)
    history = sim_sddtmpc.run(500)
    outcome = sim_sddtmpc.metrics.outcome.name
    victims = sim_sddtmpc.metrics.victims_found
    duration = sim_sddtmpc.metrics.steps_taken
    wall_collisions = sim_sddtmpc.metrics.terrain_collisions[sddtmpc]
    victim_collisions = sim_sddtmpc.metrics.victim_collisions[sddtmpc]
    fire_collisions = sim_sddtmpc.metrics.fire_collisions[sddtmpc]
    logger.info("SDDTMPC took %d steps, found %d victims, and hit %d walls, %d victims, and %d fires: %s", duration, victims, wall_collisions, victim_collisions, fire_collisions, outcome)
    Visualizer(history, 1200, 1200).run()

    # NORMAL
    # print("Running Simulation steps...")
    # history = sim.run(500)

    # plt.matshow(agent1._closest_unexplored())
    # plt.show()

    # logger.info("Launching Visualizer...")
    # viz = Visualizer(history, 1800, 1800)
    # viz.run()

if __name__ == "__main__":
    main()