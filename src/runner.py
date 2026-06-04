import concurrent.futures
import os
from time import time
import numpy as np
import pandas as pd
import logging

from .constants import MapConfig, StoppingCondition, Capabilities
from .simulator import Simulator
from .supervisor import Supervisor
from .agents.task_mpc import TaskMpcAgent
from .util import extract_useful_run_metrics, setup_logging, warmup_numba_compilers

logger = logging.getLogger(__name__)

def run_single_seed(seed: int, 
                    width: int, height: int, 
                    stoppingCondition, 
                    config: MapConfig):
    """
    Runs a single simulation with, and a single simulation without a supervisor, using the same seed for both
    """
    setup_logging(False)
    pid = os.getpid()
    start_time = time()
    logger.info(f"Process {pid} - Starting simulations for seed {seed}...")

    caps = Capabilities(exploration=1.0, rescuing=1.0)

    sim_sup = Simulator(width, height, stoppingCondition)
    sim_sup.generate_ground_truth(config, seed)
    sim_sup.add_agent(TaskMpcAgent("tasker1", 0, 1, width, height, 0.01, 0.9, 4, False, caps))
    sim_sup.add_agent(TaskMpcAgent("tasker2", 0, 2, width, height, 0.01, 0.9, 4, False, caps))
    sim_sup.add_agent(TaskMpcAgent("tasker3", 0, 3, width, height, 0.01, 0.9, 4, False, caps))
    sim_sup.add_agent(TaskMpcAgent("tasker4", 1, 0, width, height, 0.01, 0.9, 4, False, caps))
    
    supervisor = Supervisor(width, height, 0.01, 8, False)
    sim_sup.set_supervisor(supervisor)
    sim_sup.run(300)
    
    sup_metrics = extract_useful_run_metrics(sim_sup.metrics.history)
    sup_metrics["seed"] = seed
    sup_metrics["mode"] = "Supervised"
    
    sim_no_sup = Simulator(width, height, stoppingCondition)
    sim_no_sup.generate_ground_truth(config, seed)
    sim_no_sup.add_agent(TaskMpcAgent("tasker1", 0, 1, width, height, 0.01, 0.9, 4, False, caps))
    sim_no_sup.add_agent(TaskMpcAgent("tasker2", 0, 2, width, height, 0.01, 0.9, 4, False, caps))
    sim_no_sup.add_agent(TaskMpcAgent("tasker3", 0, 3, width, height, 0.01, 0.9, 4, False, caps))
    sim_no_sup.add_agent(TaskMpcAgent("tasker4", 1, 0, width, height, 0.01, 0.9, 4, False, caps))
    
    sim_no_sup.run(300)
    
    no_sup_metrics = extract_useful_run_metrics(sim_no_sup.metrics.history)
    no_sup_metrics["seed"] = seed
    no_sup_metrics["mode"] = "Unsupervised"

    duration = time() - start_time
    logger.info(f"Process {pid} - Completed simulations for seed {seed} in {duration:.2f} seconds.")

    return (sup_metrics, no_sup_metrics)

def gather_data(width: int, height: int, 
                stoppingCondition: StoppingCondition, 
                num_runs: int, 
                config: MapConfig,
                seed_start: int | None = None
                ):
    """
    Runs multiple simulations, for each seed it runs: 1 simulation with, and 1 without a supervisor
    """
    start_time = time()

    if seed_start is None:
        seed_start = (int) (np.random.random() * 1_000_000_000)
    
    max_cores = os.cpu_count()
    logger.info(f"Starting data gathering for {num_runs} seeds for starting seed {seed_start} on max {max_cores} cores...")

    all_rows = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_single_seed, seed_start + i, width, height, stoppingCondition, config) 
            for i in range(num_runs)
        ]

        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            sup_metrics, no_sup_metrics = future.result()
            all_rows.append(sup_metrics)
            all_rows.append(no_sup_metrics)

            logger.info(f"Progress: {idx + 1}/{num_runs} seeds completed")
    
    df = pd.DataFrame(all_rows)
    cols = ['seed', 'mode', 'total_steps', 'final_exploration', 'time_to_50', 'time_to_70', 'time_to_80', 'time_to_85', 'time_to_90', 'time_to_all_found']
    df = df[cols]
    
    csv_filename = f"simulation_results_{num_runs}x_{width}x{height}_{time()}.csv"
    df.to_csv(csv_filename, index=False)
    logger.info(f"Data gathering completed in {time() - start_time:.2f} seconds.")
    logger.info(f"Data successfully saved to {csv_filename}")
    evaluate_csv_data(csv_filename)

def evaluate_csv_data(csv_filename: str):
    df = pd.read_csv(csv_filename)
    logger.info(f"Data loaded from {csv_filename}")

    summary = df.groupby('mode').mean().drop('seed', axis=1, errors='ignore')
    if 'Supervised' in summary.index and 'Unsupervised' in summary.index:
        summary.loc['Ratio (Sup/Unsup)'] = summary.loc['Supervised'] / summary.loc['Unsupervised']
    logger.info(f"Average metrics by mode:\n{summary.round(3)}")

    success_rates = df.groupby('mode')['time_to_all_found'].apply(lambda x: x.notna().mean() * 100)
    
    logger.info("\n--- VICTIM RETRIEVAL SUCCESS RATE ---")
    for mode, rate in success_rates.items():
        logger.info(f"{mode}: {rate:.1f}%")


if __name__ == '__main__':
    setup_logging(False)
    warmup_numba_compilers()
    
    config = MapConfig(num_rooms=12, num_victims=10, min_room_length=4, min_room_width=4, max_room_length=8, max_room_width=8, max_tunnel_thickness=3, min_tunnel_thickness=2
                       , initial_fire_points=1, room_vulnerability_probability=0.1, tunnel_vulnerability_probability=0.1, fire_spread_rate=0.01, fire_duration=100)
    gather_data(width=60, height=60, stoppingCondition=StoppingCondition.ALL_VICTIMS_FOUND, num_runs=1, config=config)
