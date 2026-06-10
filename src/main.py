import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .simulator import Simulator, MapConfig, visualize_grid_gen
from .agents.mpc import MpcAgent
from .agents.tmpc import TmpcAgent
from .agents.sdd_tmpc import SddTmpcAgent
from .visualization import Visualizer


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
        min_tunnel_thickness=1,
        max_tunnel_thickness=3,
        num_victims=12,
        fire_spread_rate=0.04,
        fire_duration=15,
        initial_fire_points=2
    )
    # Set up pd dataframe with list of columns, rows to be appended 
    metrics = [
        'outcome',
        'victims_found', 
        'duration', 
        'wall_collisions', 
        'victims_collisions',
        'fire_collisions',
        'wait_actions',
        'area_explored',
        'infeasible_states',
        'seed'
        ]
    df_mpc = pd.DataFrame(columns=metrics)
    df_tmpc = pd.DataFrame(columns=metrics)
    df_sdd = pd.DataFrame(columns=metrics)

    number_of_sims = 250
    max_steps = 300
    mpc = MpcAgent("mpc", 1, 1, WIDTH, HEIGHT, 0.05, 0.35, 4, True)
    tmpc = TmpcAgent("tmpc", 1, 1, WIDTH, HEIGHT, 0.05, 0.35, 4, True)
    sdd = SddTmpcAgent("sdd", 1, 1, WIDTH, HEIGHT, 0.05, 0.35, 4, True)

    # * RUN SIMULATIONS
    # for i in range(number_of_sims):

    #     seed = int(np.random.random() * 1_000_000_000)
    #     sim_mpc = Simulator(WIDTH, HEIGHT)
    #     sim_tmpc = Simulator(WIDTH, HEIGHT)
    #     sim_sdd = Simulator(WIDTH, HEIGHT)
    #     mpc_instance = mpc.copy()
    #     tmpc_instance = tmpc.copy()
    #     sdd_instance = sdd.copy()
        
    #     sim_mpc.generate_ground_truth(config, seed)
    #     sim_mpc.add_agent(mpc_instance)

    #     sim_mpc.run(max_steps)
    #     outcome = sim_mpc.metrics.outcome.name
    #     victims_found = sim_mpc.metrics.victims_found
    #     duration = sim_mpc.metrics.steps_taken
    #     wall_collisions = sim_mpc.metrics.terrain_collisions[mpc_instance]
    #     victim_collisions = sim_mpc.metrics.victim_collisions[mpc_instance]
    #     fire_collisions = sim_mpc.metrics.victim_collisions[mpc_instance]
    #     wait_actions = sim_mpc.metrics.wait_actions[mpc_instance]
    #     area_explored = sim_mpc.metrics.area_explored[mpc_instance]
    #     infeasible_states = sim_mpc.metrics.infeasible_states[mpc_instance]
    #     df_mpc.loc[i] = [outcome, victims_found, duration, wall_collisions, victim_collisions, fire_collisions, wait_actions, area_explored, infeasible_states, seed]
    #     logger.info("MPC run %d complete in %d steps: %s", i + 1, duration, outcome)

    #     sim_tmpc.generate_ground_truth(config, seed)
    #     sim_tmpc.add_agent(tmpc_instance)

    #     sim_tmpc.run(max_steps)
    #     outcome = sim_tmpc.metrics.outcome.name
    #     victims_found = sim_tmpc.metrics.victims_found
    #     duration = sim_tmpc.metrics.steps_taken
    #     wall_collisions = sim_tmpc.metrics.terrain_collisions[tmpc_instance]
    #     victim_collisions = sim_tmpc.metrics.victim_collisions[tmpc_instance]
    #     fire_collisions = sim_tmpc.metrics.victim_collisions[tmpc_instance]
    #     wait_actions = sim_tmpc.metrics.wait_actions[tmpc_instance]
    #     area_explored = sim_tmpc.metrics.area_explored[tmpc_instance]
    #     infeasible_states = sim_tmpc.metrics.infeasible_states[tmpc_instance]
    #     df_tmpc.loc[i] = [outcome, victims_found, duration, wall_collisions, victim_collisions, fire_collisions, wait_actions, area_explored, infeasible_states, seed]
    #     logger.info("TMPC run %d complete in %d steps: %s", i + 1, duration, outcome)
        
    #     sim_sdd.generate_ground_truth(config, seed)
    #     sim_sdd.add_agent(sdd_instance)

    #     sim_sdd.run(max_steps)
    #     outcome = sim_sdd.metrics.outcome.name
    #     victims_found = sim_sdd.metrics.victims_found
    #     duration = sim_sdd.metrics.steps_taken
    #     wall_collisions = sim_sdd.metrics.terrain_collisions[sdd_instance]
    #     victim_collisions = sim_sdd.metrics.victim_collisions[sdd_instance]
    #     fire_collisions = sim_sdd.metrics.victim_collisions[sdd_instance]
    #     wait_actions = sim_sdd.metrics.wait_actions[sdd_instance]
    #     area_explored = sim_sdd.metrics.area_explored[sdd_instance]
    #     infeasible_states = sim_sdd.metrics.infeasible_states[sdd_instance]
    #     df_sdd.loc[i] = [outcome, victims_found, duration, wall_collisions, victim_collisions, fire_collisions, wait_actions, area_explored, infeasible_states, seed]
    #     logger.info("SDD-TMPC run %d complete in %d steps: %s\n", i + 1, duration, outcome)

    # df_mpc.to_csv("saved_runs/data/mpc_results.csv")
    # df_tmpc.to_csv("saved_runs/data/tmpc_results.csv")
    # df_sdd.to_csv("saved_runs/data/sdd_results.csv")

    # * TEST SEED
    # sim = Simulator(WIDTH, HEIGHT)

    # agent = tmpc

    # sim.generate_ground_truth(config, 698461091)
    # sim.add_agent(agent)

    # visualize_grid_gen(sim.ground_truth.traversability, sim.ground_truth.agents, sim.ground_truth.victims, sim.ground_truth.vulnerability, sim.ground_truth.fire)

    # history = sim.run(max_steps)
    # print(sim.metrics.terrain_collisions[agent], sim.metrics.victim_collisions[agent], sim.metrics.fire_collisions[agent])

    # viz = Visualizer(history, 1200, 1200)
    # viz.run()

    # * HISTORY 
    viz = Visualizer.from_file("saved_runs/private/sim_20260602_003720_122steps_mpc.pkl")
    viz.run()

    viz = Visualizer.from_file("saved_runs/private/sim_20260602_003814_300steps_tmpc.pkl")
    viz.run()
    
    viz = Visualizer.from_file("saved_runs/private/sim_20260602_003614_122steps_sdd.pkl")
    viz.run()

    # * Personal
    # seeds = pd.read_csv("saved_runs/data/sdd_results.csv").seed
    # data = []
    # for i, seed in enumerate(seeds):
    #     logger.info(f"Run {i + 1}:")
    #     sim = Simulator(WIDTH, HEIGHT)
    #     agent = sdd.copy()
    #     sim.generate_ground_truth(config, seed)
    #     sim.add_agent(agent)
    #     history = sim.run(max_steps)

    #     total_steps = sim.metrics.steps_taken
    #     logger.info(f"Total steps: {total_steps}\n")
    #     for step, size in enumerate(agent.tube_sizes):
    #         progress = step / total_steps
    #         data.append([seed, progress, size])

    # size_df = pd.DataFrame(data, columns=["seed", "progress", "tube_size"])
    # size_df.to_csv("saved_runs/data/size_time.csv")

if __name__ == "__main__":
    main()