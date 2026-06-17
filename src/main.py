import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .simulator import Simulator, MapConfig, visualize_grid_gen
from .agent import Agent
from .agents.mpc import MpcAgent
from .agents.tmpc import TmpcAgent
from .agents.sdd_tmpc import SddTmpcAgent
from .visualization import Visualizer


WIDTH = 35
HEIGHT = 35

logger = logging.getLogger(__name__)

N_WORKERS = 14

def setup_logging(is_debug: bool):
    """Configures the global logging level and format."""
    log_level = logging.DEBUG if is_debug else logging.INFO

    logging.basicConfig(
        level=log_level,
        format='%(levelname)s - %(message)s',
    )

def run_single(seed: int, delta: float, config: MapConfig, agent_template: Agent, width: int, height: int, max_steps: int):
    """Run one simulation for a given seed/delta and return its metrics row.
 
    Each call gets its own Simulator and its own copy of the agent, so there
    is no shared mutable state between threads.
    """
    sim = Simulator(width, height)
    agent = agent_template.copy()
    sim.generate_ground_truth(config, seed)
    sim.add_agent(agent)
 
    history = sim.run(max_steps)
 
    outcome = sim.metrics.outcome.name
    victims_found = sim.metrics.victims_found
    duration = sim.metrics.steps_taken
    wall_collisions = sim.metrics.terrain_collisions[agent]
    victim_collisions = sim.metrics.victim_collisions[agent]
    fire_collisions = sim.metrics.fire_collisions[agent]
    wait_actions = sim.metrics.wait_actions[agent]
    area_explored = sim.metrics.area_explored[agent]
    infeasible_states = sim.metrics.infeasible_states[agent]
    avg_tube_size = agent.tube_size / duration
 
    row = [
        outcome,
        victims_found,
        duration,
        wall_collisions,
        victim_collisions,
        fire_collisions,
        wait_actions,
        area_explored,
        infeasible_states,
        avg_tube_size,
        seed,
        delta
    ]

    return seed, delta, duration, outcome, row

def gather_results(seeds, deltas, config, agent_template, width, height, max_steps, metrics, n_workers=N_WORKERS):
    """Run all (seed, delta) combinations in a thread pool and collect results.
 
    Submits every (seed, delta) pair up front and lets the pool's n_workers
    threads pull from the queue as they free up, which keeps ~n_workers sims
    running at all times regardless of how unevenly run durations vary.
    """
    jobs = [(seed, delta) for delta in deltas for seed in seeds]
    total = len(jobs)
    data = [None] * total
    completed = 0
 
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                run_single, seed, delta, config, agent_template, width, height, max_steps
            ): idx
            for idx, (seed, delta) in enumerate(jobs)
        }
 
        for future in as_completed(futures):
            idx = futures[future]
            try:
                seed, delta, duration, outcome, row = future.result()
            except Exception:
                logger.exception("Run with seed=%s failed", jobs[idx][0])
                raise
 
            data[idx] = row
            completed += 1
            logger.info(
                "Run %d/%d complete (seed=%s, delta=%s) in %d steps: %s",
                completed, total, seed, delta, duration, outcome,
            )
 
    return pd.DataFrame(data, columns=metrics)


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
        'avg_tube_size',
        'seed',
        'delta'
        ]
    # df_mpc = pd.DataFrame(columns=metrics)
    # df_tmpc = pd.DataFrame(columns=metrics)
    # df_sdd = pd.DataFrame(columns=metrics)

    max_steps = 300
    mpc = MpcAgent("mpc", 1, 1, WIDTH, HEIGHT, 0.05, 0.35, 4, True)
    tmpc = TmpcAgent("tmpc", 1, 1, WIDTH, HEIGHT, 0.05, 0.35, 4, True)
    sdd = SddTmpcAgent("sdd", 1, 1, WIDTH, HEIGHT, 0.05, 0.35, 4, True)

    seeds = pd.read_csv('saved_runs/data/sdd_results.csv').seed
    deltas = [0.0, 0.05, 0.10, 0.20]

    # Get first 5 seeds
    # seeds = seeds[:5]
    deltas = [0.0]

    agent_template = tmpc

    noise_df = gather_results(
        seeds,
        deltas,
        config,
        agent_template,
        WIDTH,
        HEIGHT,
        max_steps,
        metrics,
        n_workers=N_WORKERS
    )

    noise_df.to_csv('saved_runs/data/tmpc_results.csv', index=False)

if __name__ == "__main__":
    main()