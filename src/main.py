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

<<<<<<< Updated upstream
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
=======
    number_of_sims = 250
    max_steps = 500
    mpc = MpcAgent("mpc", 1, 1, WIDTH, HEIGHT, 0.05, 0.4, 4, True)
    tmpc = TmpcAgent("tmpc", 1, 1, WIDTH, HEIGHT, 0.05, 0.4, 4, True)
    # agent = TmpcAgent("tmpc", 1, 1, WIDTH, HEIGHT, 0.05, 0.4, 4, True)

    # * RUN SIMULATIONS
    for i in range(number_of_sims):

        seed = int(np.random.random() * 1_000_000_000)
        sim_mpc = Simulator(WIDTH, HEIGHT)
        sim_tmpc = Simulator(WIDTH, HEIGHT)
        mpc_instance = mpc.copy()
        tmpc_instance = tmpc.copy()
        
        sim_mpc.generate_ground_truth(config, seed)
        sim_mpc.add_agent(mpc_instance)

        if i == 0:
            visualize_grid_gen(sim_mpc.ground_truth.traversability, sim_mpc.ground_truth.agents, sim_mpc.ground_truth.victims, sim_mpc.ground_truth.vulnerability, sim_mpc.ground_truth.fire)
        
        sim_mpc.run(max_steps)
        outcome = sim_mpc.metrics.outcome.name
        victims_found = sim_mpc.metrics.victims_found
        duration = sim_mpc.metrics.steps_taken
        wall_collisions = sim_mpc.metrics.terrain_collisions[mpc_instance]
        victim_collisions = sim_mpc.metrics.victim_collisions[mpc_instance]
        fire_collisions = sim_mpc.metrics.victim_collisions[mpc_instance]
        wait_actions = sim_mpc.metrics.wait_actions[mpc_instance]
        area_explored = sim_mpc.metrics.area_explored[mpc_instance]
        infeasible_states = sim_mpc.metrics.infeasible_states[mpc_instance]
        df_mpc.loc[i] = [outcome, victims_found, duration, wall_collisions, victim_collisions, fire_collisions, wait_actions, area_explored, infeasible_states, seed]
        logger.info("MPC run %d complete in %d steps: %s", i + 1, duration, outcome)

        sim_tmpc.generate_ground_truth(config, seed)
        sim_tmpc.add_agent(tmpc_instance)

        sim_tmpc.run(max_steps)
        outcome = sim_tmpc.metrics.outcome.name
        victims_found = sim_tmpc.metrics.victims_found
        duration = sim_tmpc.metrics.steps_taken
        wall_collisions = sim_tmpc.metrics.terrain_collisions[tmpc_instance]
        victim_collisions = sim_tmpc.metrics.victim_collisions[tmpc_instance]
        fire_collisions = sim_tmpc.metrics.victim_collisions[tmpc_instance]
        wait_actions = sim_tmpc.metrics.wait_actions[tmpc_instance]
        area_explored = sim_tmpc.metrics.area_explored[tmpc_instance]
        infeasible_states = sim_tmpc.metrics.infeasible_states[tmpc_instance]
        df_tmpc.loc[i] = [outcome, victims_found, duration, wall_collisions, victim_collisions, fire_collisions, wait_actions, area_explored, infeasible_states, seed]
        logger.info("TMPC run %d complete in %d steps: %s\n", i + 1, duration, outcome)
    
    df_mpc.to_csv("saved_runs/data/mpc_results.csv")
    df_tmpc.to_csv("saved_runs/data/tmpc_results.csv")

    # * TEST SEED
    # sim_tmpc = Simulator(WIDTH, HEIGHT)

    # sim_tmpc.generate_ground_truth(config, 179742288)
    # sim_tmpc.add_agent(tmpc)
    # history = sim_tmpc.run(max_steps)

    # viz = Visualizer(history, 1200, 1200)
    # viz.run()
>>>>>>> Stashed changes

if __name__ == "__main__":
    main()