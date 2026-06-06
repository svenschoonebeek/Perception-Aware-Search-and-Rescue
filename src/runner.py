import concurrent.futures
from datetime import date
import os
from time import time, strftime
import numpy as np
import pandas as pd
import logging

from .constants import MapConfig, StoppingCondition, Capabilities
from .simulator import Simulator
from .supervisor import Supervisor
from .agents.task_mpc import TaskMpcAgent
from .util import extract_useful_run_metrics, plot_boxes, setup_logging, warmup_numba_compilers, plot_total_spatial_conflicts, plot_victim_detection_times

logger = logging.getLogger(__name__)

def run_single_seed(seed: int, num_steps: int,
                    width: int, height: int, 
                    stoppingCondition, 
                    config: MapConfig):
    """
    Runs a single simulation with, and a single simulation without a supervisor, using the same seed for both
    """
    setup_logging(False)
    pid = os.getpid()
    logger.info(f"Process {pid} - Starting simulations for seed {seed}...")

    caps = Capabilities(exploration=1.0, rescuing=1.0)

    sim_sup = Simulator(width, height, stoppingCondition)
    sim_sup.generate_ground_truth(config, seed)
    sim_sup.add_agent(TaskMpcAgent("tasker1", 0, 0, width, height, 0.01, 0.9, 4, False, caps))
    sim_sup.add_agent(TaskMpcAgent("tasker2", 0, 1, width, height, 0.01, 0.9, 4, False, caps))
    sim_sup.add_agent(TaskMpcAgent("tasker3", 0, 2, width, height, 0.01, 0.9, 4, False, caps))
    sim_sup.add_agent(TaskMpcAgent("tasker4", 1, 0, width, height, 0.01, 0.9, 4, False, caps))
    
    supervisor = Supervisor(width, height, 0.01, 8, False)
    sup_start_time = time()
    sim_sup.set_supervisor(supervisor)
    sim_sup.run(num_steps)
    sup_duration = time() - sup_start_time

    sup_metrics = extract_useful_run_metrics(sim_sup.metrics.history)
    sup_metrics["seed"] = seed
    sup_metrics["mode"] = "Supervised"
    sup_ran_steps = sup_metrics.get("total_steps", num_steps)
    sup_metrics["avg_step_duration"] = sup_duration / sup_ran_steps if sup_ran_steps > 0 else 0

    sim_no_sup = Simulator(width, height, stoppingCondition)
    sim_no_sup.generate_ground_truth(config, seed)
    sim_no_sup.add_agent(TaskMpcAgent("tasker1", 0, 0, width, height, 0.01, 0.9, 4, False, caps))
    sim_no_sup.add_agent(TaskMpcAgent("tasker2", 0, 1, width, height, 0.01, 0.9, 4, False, caps))
    sim_no_sup.add_agent(TaskMpcAgent("tasker3", 0, 2, width, height, 0.01, 0.9, 4, False, caps))
    sim_no_sup.add_agent(TaskMpcAgent("tasker4", 1, 0, width, height, 0.01, 0.9, 4, False, caps))
    
    no_sup_start_time = time()
    sim_no_sup.run(num_steps)
    no_sup_duration = time() - no_sup_start_time
    
    no_sup_metrics = extract_useful_run_metrics(sim_no_sup.metrics.history)
    no_sup_metrics["seed"] = seed
    no_sup_metrics["mode"] = "Unsupervised"
    no_sup_ran_steps = no_sup_metrics.get("total_steps", num_steps)
    no_sup_metrics["avg_step_duration"] = no_sup_duration / no_sup_ran_steps if no_sup_ran_steps > 0 else 0

    logger.info(f"Process {pid} - Completed simulations for seed {seed}")

    return (sup_metrics, no_sup_metrics)

def gather_data(width: int, height: int, 
                stoppingCondition: StoppingCondition, 
                num_runs: int, steps_per_run: int,
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
    if max_cores is None:
        return
    cores_used = max_cores - 2
    logger.info(f"Starting data gathering for {num_runs} seeds for starting seed {seed_start} on max {cores_used} cores...")

    all_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=cores_used) as executor:
        futures = [
            executor.submit(run_single_seed, seed_start + i, steps_per_run, width, height, stoppingCondition, config) 
            for i in range(num_runs)
        ]

        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            sup_metrics, no_sup_metrics = future.result()
            all_rows.append(sup_metrics)
            all_rows.append(no_sup_metrics)

            logger.info(f"Progress: {idx + 1}/{num_runs} seeds completed")
    
    df = pd.DataFrame(all_rows)
    cols = ['seed', 'mode', 'total_steps', 'final_exploration', 
            'time_to_50', 'time_to_70', 'time_to_80', 'time_to_85', 'time_to_90', 'time_to_all_found',
            'avg_inter_agent_distance', 'avg_step_duration', 'total_spatial_conflicts', 'found_victims', 'individual_detection_times']
    df = df[cols]
    
    csv_filename = f"{date.today()}_{strftime('%H-%M-%S')}_{num_runs}x_{width}x{height}_sim_res.csv"
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(parent_dir, 'sim_data', csv_filename)
    logger.info(f"Saving data to {csv_path}.")
    df.to_csv(csv_path, index=False)

    logger.info(f"Data gathering completed in {time() - start_time:.2f} seconds.")
    logger.info(f"Data successfully saved to {csv_path}")
    evaluate_csv_data(csv_path)

def evaluate_csv_data(csv_filename: str):
    df = pd.read_csv(csv_filename)
    logger.info(f"Data loaded from {csv_filename}")

    summary = df.groupby('mode').mean(numeric_only=True).drop('seed', axis=1, errors='ignore')
    summary['std_time_all_found'] = df.groupby('mode')['time_to_all_found'].std()
    summary['std_avg_step_duration'] = df.groupby('mode')['avg_step_duration'].std()

    if 'Supervised' in summary.index and 'Unsupervised' in summary.index:
        summary.loc['Ratio (Sup/Unsup)'] = summary.loc['Supervised'] / summary.loc['Unsupervised']

    milestone_cols = ['time_to_50', 'time_to_70', 'time_to_80', 'time_to_85', 'time_to_90']
    spatial_cols = ['total_steps', 'final_exploration', 'avg_inter_agent_distance', 'total_spatial_conflicts', 'found_victims']
    perf_cols = ['time_to_all_found', 'std_time_all_found', 'avg_step_duration', 'std_avg_step_duration']

    logger.info("\n" + "="*50)
    logger.info("--- EXPLORATION MILESTONES (TIME STEPS) ---")
    logger.info(f"\n{summary[milestone_cols].round(3).to_string()}")

    logger.info("\n--- MISSION & SPATIAL METRICS ---")
    logger.info(f"\n{summary[spatial_cols].round(3).to_string()}")

    logger.info("\n--- DISCOVERY & COMPUTATION TIME ---")
    logger.info(f"\n{summary[perf_cols].round(3).to_string()}")
    logger.info("="*50)

    success_rates = df.groupby('mode')['time_to_all_found'].apply(lambda x: x.notna().mean() * 100)
    
    logger.info("\n--- VICTIM RETRIEVAL SUCCESS RATE ---")
    for mode, rate in success_rates.items():
        logger.info(f"{mode}: {rate:.1f}%")
    
    plot_boxes(csv_filename)

if __name__ == '__main__':
    setup_logging(False)
    warmup_numba_compilers()
    
    config = MapConfig(num_rooms=12, num_victims=20, min_room_length=4, min_room_width=4, max_room_length=8, max_room_width=8, max_tunnel_thickness=3, min_tunnel_thickness=2
                       , initial_fire_points=1, room_vulnerability_probability=0.1, tunnel_vulnerability_probability=0.1, fire_spread_rate=0.01, fire_duration=100)
    gather_data(width=60, height=60, stoppingCondition=StoppingCondition.ALL_VICTIMS_FOUND, num_runs=1, steps_per_run=1, config=config)
    # plot_total_spatial_conflicts("2026-06-05_16-24-46_108x_60x60_sim_res.csv")
    # evaluate_csv_data("2026-06-06_10-02-09_1x_60x60_sim_res.csv")