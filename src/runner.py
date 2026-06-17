import concurrent.futures
from datetime import date
import os
from time import time, strftime
import numpy as np
import pandas as pd
import logging
import scipy.stats as stats

from .constants import MapConfig, StoppingCondition, Capabilities
from .simulator import Simulator
from .supervisor import Supervisor
from .agents.task_mpc import TaskMpcAgent
from .util import extract_useful_run_metrics, plot_boxes, setup_logging, warmup_numba_compilers, plot_total_spatial_conflicts, plot_victim_detection_times
from .state import State

logger = logging.getLogger(__name__)

def run_single_seed(seed: int, num_steps: int,
                    width: int, height: int, 
                    stoppingCondition: StoppingCondition, 
                    config: MapConfig,
                    with_unsupervised: bool = False,
                    with_shared_map: bool = False,
                    ) -> tuple[dict, dict, dict]:
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
    no_sup_metrics = {}

    sup_metrics["seed"] = seed
    sup_metrics["mode"] = "Supervised"
    sup_metrics["num_agents"] = len(sim_sup.agents)
    sup_ran_steps = sup_metrics.get("total_steps", num_steps)
    sup_metrics["avg_step_duration"] = sup_duration / sup_ran_steps if sup_ran_steps > 0 else 0

    if with_unsupervised:
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
        no_sup_metrics["num_agents"] = len(sim_no_sup.agents)
        no_sup_ran_steps = no_sup_metrics.get("total_steps", num_steps)
        no_sup_metrics["avg_step_duration"] = no_sup_duration / no_sup_ran_steps if no_sup_ran_steps > 0 else 0

    if not with_shared_map:
        return (sup_metrics, no_sup_metrics, {})
    
    sim_no_sup_shared_map = Simulator(width, height, stoppingCondition)
    sim_no_sup_shared_map.generate_ground_truth(config, seed)
    sim_no_sup_shared_map.add_agent(tasker1 := TaskMpcAgent("tasker1", 0, 0, width, height, 0.01, 0.9, 4, False, caps))
    sim_no_sup_shared_map.add_agent(tasker2 := TaskMpcAgent("tasker2", 0, 1, width, height, 0.01, 0.9, 4, False, caps))
    sim_no_sup_shared_map.add_agent(tasker3 := TaskMpcAgent("tasker3", 0, 2, width, height, 0.01, 0.9, 4, False, caps))
    sim_no_sup_shared_map.add_agent(tasker4 := TaskMpcAgent("tasker4", 1, 0, width, height, 0.01, 0.9, 4, False, caps))
    # Share the perception map among all agents
    shared_perception = State(width, height)
    for agent in [tasker1, tasker2, tasker3, tasker4]:
        agent.perception = shared_perception
    
    no_sup_shared_map_start_time = time()
    sim_no_sup_shared_map.run(num_steps)
    no_sup_shared_map_duration = time() - no_sup_shared_map_start_time
    
    no_sup_shared_map_metrics = extract_useful_run_metrics(sim_no_sup_shared_map.metrics.history)
    no_sup_shared_map_metrics["seed"] = seed
    no_sup_shared_map_metrics["mode"] = "Shared"
    no_sup_shared_map_metrics["num_agents"] = len(sim_no_sup_shared_map.agents)
    no_sup_shared_map_ran_steps = no_sup_shared_map_metrics.get("total_steps", num_steps)
    no_sup_shared_map_metrics["avg_step_duration"] = no_sup_shared_map_duration / no_sup_shared_map_ran_steps if no_sup_shared_map_ran_steps > 0 else 0

    logger.info(f"Process {pid} - Completed simulations for seed {seed}")

    return (sup_metrics, no_sup_metrics, no_sup_shared_map_metrics)

def gather_data(width: int, height: int, 
                stoppingCondition: StoppingCondition, 
                num_runs: int, steps_per_run: int,
                config: MapConfig,
                seed_start: int | None = None,
                with_unsupervised: bool = False,
                with_shared_map: bool = False
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
    cores_used = max_cores# - 2
    logger.info(f"Starting data gathering for {num_runs} seeds for starting seed {seed_start} on max {cores_used} cores...")

    all_rows = []
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=cores_used)

    try:
        futures = [
            executor.submit(run_single_seed, seed_start + i, steps_per_run, width, height, stoppingCondition, config, with_unsupervised, with_shared_map) 
            for i in range(num_runs)
        ]

        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            sup_metrics, no_sup_metrics, no_sup_shared_map_metrics = future.result()
            all_rows.append(sup_metrics)
            all_rows.append(no_sup_metrics)
            all_rows.append(no_sup_shared_map_metrics)

            logger.info(f"Progress: {idx + 1}/{num_runs} seeds completed")
        executor.shutdown(wait=True)
    except KeyboardInterrupt:
        logger.warning("manual shutdown received. Saving data...")
        executor.shutdown(wait=False, cancel_futures=True)
    
    if not all_rows:
        logger.warning("No data was collected.")
        return
    
    df = pd.DataFrame(all_rows)
    cols = ['seed', 'mode', 'num_agents', 'total_steps', 'final_exploration', 
            'time_to_50', 'time_to_70', 'time_to_80', 'time_to_85', 'time_to_90', 'time_to_all_found',
            'avg_inter_agent_distance', 'avg_step_duration', 'total_spatial_conflicts', 'found_victims', 'individual_detection_times',
            'total_terrain_collisions', 'total_victim_collisions', 'total_fire_collisions', 'total_inter_agent_collisions', 'total_task_allocations',
            'exploration_redundancy',
            'total_event_resolve_time', 'total_event_resolves', 'total_wait_actions', 'total_damage',
            'total_time_in_vulnerable', 'total_time_in_high_risk', 'total_infeasible_states'
            ]
    df = df[cols]
    
    csv_filename = f"event_resolve_{date.today()}_{strftime('%H-%M-%S')}_{num_runs}x_{width}x{height}_{config.initial_fire_points}x_fire_sim_res.csv"
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(parent_dir, 'sim_data', csv_filename)
    logger.info(f"Saving data to {csv_path}.")
    df.to_csv(csv_path, index=False)

    logger.info(f"Data gathering completed in {time() - start_time:.2f} seconds.")
    logger.info(f"Data successfully saved to {csv_path}")
    # evaluate_csv_data(csv_path)

def evaluate_csv_data(csv_filename: str):
    df = pd.read_csv(csv_filename)
    logger.info(f"Data loaded from {csv_filename}")

    if 'final_exploration' in df.columns and 'total_steps' in df.columns:
        df['exploration_velocity'] = (df['final_exploration'] / df['total_steps']) * 100

    if 'total_time_in_high_risk' in df.columns and 'total_steps' in df.columns:
        agents = df['num_agents'] if 'num_agents' in df.columns else 1
        df['fraction_time_in_high_risk'] = df['total_time_in_high_risk'] / (df['total_steps'] * agents)
        df['fraction_time_in_vulnerable'] = df['total_time_in_vulnerable'] / (df['total_steps'] * agents)
    
    summary = df.groupby('mode').mean(numeric_only=True).drop('seed', axis=1, errors='ignore')
    summary['std_time_all_found'] = df.groupby('mode')['time_to_all_found'].std()
    summary['std_avg_step_duration'] = df.groupby('mode')['avg_step_duration'].std()
    summary['std_avg_inter_agent_distance'] = df.groupby('mode')['avg_inter_agent_distance'].std()
    
    summary['avg_event_resolve_time'] = df.groupby('mode')['total_event_resolve_time'].mean() / df.groupby('mode')['total_event_resolves'].mean()

    if 'Supervised' in summary.index and 'Unsupervised' in summary.index:
        summary.loc['Ratio (Sup/Unsup)'] = summary.loc['Supervised'] / summary.loc['Unsupervised']
    if 'Supervised' in summary.index and 'Shared' in summary.index:
        summary.loc['Ratio (Sup/Unsup_Shared)'] = summary.loc['Supervised'] / summary.loc['Shared']

    milestone_cols = [c for c in ['time_to_50', 'time_to_70', 'time_to_80', 'time_to_85', 'time_to_90'] if c in summary.columns]
    spatial_cols = [c for c in ['total_steps', 'final_exploration', 'avg_inter_agent_distance', 'total_spatial_conflicts', 'found_victims', 'exploration_velocity'] if c in summary.columns]
    perf_cols = [c for c in ['time_to_all_found', 'avg_step_duration', 'total_task_allocations', 'total_event_resolve_time', 'total_event_resolves','avg_event_resolve_time', 'exploration_redundancy'] if c in summary.columns]
    collision_cols = [c for c in ['total_terrain_collisions', 'total_victim_collisions', 'total_fire_collisions', 'total_inter_agent_collisions', 'total_infeasible_states','total_damage'] if c in summary.columns]
    safety_cols = [c for c in ['total_time_in_vulnerable', 'total_time_in_high_risk', 'fraction_time_in_vulnerable', 'fraction_time_in_high_risk', 'total_wait_actions'] if c in summary.columns]
    
    logger.info("\n" + "="*60)
    logger.info("--- EXPLORATION MILESTONES (TIME STEPS) ---")
    logger.info(f"\n{summary[milestone_cols].round(3).to_string()}")

    logger.info("\n--- MISSION & SPATIAL METRICS ---")
    logger.info(f"\n{summary[spatial_cols].round(3).to_string()}")

    logger.info("\n--- DISCOVERY & COMPUTATION TIME ---")
    logger.info(f"\n{summary[perf_cols].round(3).to_string()}")
    
    logger.info("\n--- SAFETY METRICS (COLLISIONS) ---")
    logger.info(f"\n{summary[collision_cols].round(3).to_string()}")

    logger.info("\n--- SAFETY METRICS (TIME IN VULNERABLE/HIGH RISK) ---")
    logger.info(f"\n{summary[safety_cols].round(3).to_string()}")

    logger.info("\n" + "="*60)
    logger.info("--- PAIRED STATISTICAL ANALYSIS ---")
    
    if 'seed' in df.columns:
        paired_df = df.pivot(index='seed', columns='mode')
        metrics_to_test = ['time_to_all_found', 'avg_step_duration', 
                           'avg_inter_agent_distance', 'exploration_velocity', 
                           'time_to_50', 'time_to_90',
                           'total_inter_agent_collisions', 'total_fire_collisions',
                           'total_terrain_collisions', 'total_infeasible_states',
                           'total_damage', 'fraction_time_in_vulnerable',
                           'fraction_time_in_high_risk' ]
        
        comparisons = [('Supervised', 'Unsupervised'), ('Supervised', 'Shared')]
        
        for comp_a, comp_b in comparisons:
            if comp_a in df['mode'].values and comp_b in df['mode'].values:
                logger.info(f"\n--- Comparing {comp_a} vs {comp_b} ---")
                for metric in metrics_to_test:
                    if metric not in paired_df.columns:
                        continue
                        
                    valid_pairs = paired_df[metric][[comp_a, comp_b]].dropna()
                    n_pairs = len(valid_pairs)
                    
                    if n_pairs < 2:
                        continue
                        
                    data_a = valid_pairs[comp_a]
                    data_b = valid_pairs[comp_b]
                    
                    diff = data_a - data_b
                    mean_diff = diff.mean()
                    median_diff = diff.median()
                    
                    sem = diff.std() / np.sqrt(n_pairs)
                    ci_margin = stats.t.ppf(0.975, n_pairs - 1) * sem
                    ci_lower, ci_upper = mean_diff - ci_margin, mean_diff + ci_margin
                    
                    t_stat, p_val = stats.ttest_rel(data_a, data_b)
                    p_str = "< 0.001" if p_val < 0.001 else f"= {p_val:.3f}"
                    
                    logger.info(f"\nMetric: {metric} (n={n_pairs})")
                    logger.info(f"  Mean Paired Diff   : {mean_diff:+.3f}")
                    logger.info(f"  Median Paired Diff : {median_diff:+.3f}")
                    logger.info(f"  95% CI of Diff     : [{ci_lower:+.3f}, {ci_upper:+.3f}]")
                    logger.info(f"  Paired t-test      : t-stat = {t_stat:.2f}, p-value {p_str}")
    else:
        logger.warning("Could not perform paired analysis: 'seed' column missing.")

    logger.info("\n" + "="*60)
    logger.info("--- PAPER-READY DIGITS ---")
    
    success_rates = df.groupby('mode')['time_to_all_found'].apply(lambda x: x.notna().mean() * 100)
    logger.info("1. Success Rates (Table 4):")
    for mode, rate in success_rates.items():
        logger.info(f"   - {mode}: {rate:.1f}%")
        
    if 'Supervised' in summary.index:
        logger.info("\n2. Time to All Found (Table 4):")
        logger.info(f"   - Supervised            : {summary.loc['Supervised', 'time_to_all_found']:.2f} ± {summary.loc['Supervised', 'std_time_all_found']:.2f} steps")
        
        if 'Unsupervised' in summary.index:
            logger.info(f"   - Unsupervised          : {summary.loc['Unsupervised', 'time_to_all_found']:.2f} ± {summary.loc['Unsupervised', 'std_time_all_found']:.2f} steps")
            percentage_faster = ((1/summary.loc['Ratio (Sup/Unsup)', 'time_to_all_found'] - 1) * 100) # type: ignore
            logger.info(f"   - Improvement vs Unsup  : {percentage_faster:.2f}% faster")
            std_percentage_lower = (summary.loc['Unsupervised', 'std_time_all_found'] / summary.loc['Supervised', 'std_time_all_found'] - 1) * 100# type: ignore
            logger.info(f"   - Consist. Imp. vs Unsup: {std_percentage_lower:.2f}% lower std dev")
            
        if 'Shared' in summary.index:
            logger.info(f"   - Unsupervised (Shared) : {summary.loc['Shared', 'time_to_all_found']:.2f} ± {summary.loc['Shared', 'std_time_all_found']:.2f} steps")
            percentage_faster_shared = ((1/summary.loc['Ratio (Sup/Unsup_Shared)', 'time_to_all_found'] - 1) * 100)# type: ignore
            logger.info(f"   - Improvement vs Shared : {percentage_faster_shared:.2f}% faster")
            std_percentage_lower_shared = (summary.loc['Shared', 'std_time_all_found'] / summary.loc['Supervised', 'std_time_all_found'] - 1) * 100# type: ignore
            logger.info(f"   - Consist. Imp. vs Share: {std_percentage_lower_shared:.2f}% lower std dev")

        logger.info("\n3. Swarm Dispersion / MIAD (Section 5.1):")
        miad_sup = summary.loc['Supervised', 'avg_inter_agent_distance']
        logger.info(f"   - Supervised MIAD       : {miad_sup:.2f}")
        
        if 'Unsupervised' in summary.index:
            miad_unsup = summary.loc['Unsupervised', 'avg_inter_agent_distance']
            miad_increase = ((miad_sup - miad_unsup) / miad_unsup) * 100# type: ignore
            logger.info(f"   - Unsupervised MIAD     : {miad_unsup:.2f} (+{miad_increase:.1f}%)")
            
        if 'Shared' in summary.index:
            miad_shared = summary.loc['Shared', 'avg_inter_agent_distance']
            miad_increase_shared = ((miad_sup - miad_shared) / miad_shared) * 100# type: ignore
            logger.info(f"   - Unsup (Shared) MIAD   : {miad_shared:.2f} (+{miad_increase_shared:.1f}%)")

        if 'exploration_velocity' in summary.columns:
            logger.info("\n4. Exploration Velocity (Section 5.1):")
            vel_sup = summary.loc['Supervised', 'exploration_velocity']
            logger.info(f"   - Supervised Velocity   : {vel_sup:.3f}% per step")
            
            if 'Unsupervised' in summary.index:
                vel_unsup = summary.loc['Unsupervised', 'exploration_velocity']
                vel_increase = ((vel_sup - vel_unsup) / vel_unsup) * 100# type: ignore
                logger.info(f"   - Unsupervised Velocity : {vel_unsup:.3f}% per step (+{vel_increase:.1f}%)")
                
            if 'Shared' in summary.index:
                vel_shared = summary.loc['Shared', 'exploration_velocity']
                vel_increase_shared = ((vel_sup - vel_shared) / vel_shared) * 100# type: ignore
                logger.info(f"   - Unsup (Shared) Vel    : {vel_shared:.3f}% per step (+{vel_increase_shared:.1f}%)")
                
        logger.info("\n5. Computation Time (Section 5.1):")
        logger.info(f"   - Supervised            : {summary.loc['Supervised', 'avg_step_duration']:.3f}s")
        if 'Unsupervised' in summary.index:
            logger.info(f"   - Unsupervised          : {summary.loc['Unsupervised', 'avg_step_duration']:.3f}s")
        if 'Shared' in summary.index:
            logger.info(f"   - Unsupervised (Shared) : {summary.loc['Shared', 'avg_step_duration']:.3f}s")

        if 'total_inter_agent_collisions' in summary.columns:
            logger.info("\n6. Safety Metrics (Section 5.1):")
            collisions_sup = summary.loc['Supervised', 'total_inter_agent_collisions']
            logger.info(f"   - Supervised Inter-Agent Collisions : {collisions_sup:.3f}")

            if 'Unsupervised' in summary.index:
                collisions_unsup = summary.loc['Unsupervised', 'total_inter_agent_collisions']
                collisions_reduction = ((collisions_unsup - collisions_sup) / collisions_unsup) * 100# type: ignore
                logger.info(f"   - Unsupervised Inter-Agent Collisions : {collisions_unsup:.3f} (-{collisions_reduction:.1f}%)")

            if 'Shared' in summary.index:
                collisions_shared = summary.loc['Shared', 'total_inter_agent_collisions']
                collisions_reduction_shared = ((collisions_shared - collisions_sup) / collisions_shared) * 100# type: ignore
                logger.info(f"   - Shared Inter-Agent Collisions : {collisions_shared:.3f} (-{collisions_reduction_shared:.1f}%)")

        if 'total_fire_collisions' in summary.columns:
            fire_sup = summary.loc['Supervised', 'total_fire_collisions']
            logger.info(f"   - Supervised Fire Collisions : {fire_sup:.3f}")

            if 'Unsupervised' in summary.index:
                fire_unsup = summary.loc['Unsupervised', 'total_fire_collisions']
                fire_reduction = ((fire_unsup - fire_sup) / fire_unsup) * 100# type: ignore
                logger.info(f"   - Unsupervised Fire Collisions : {fire_unsup:.3f} (-{fire_reduction:.1f}%)")

            if 'Shared' in summary.index:
                fire_shared = summary.loc['Shared', 'total_fire_collisions']
                fire_reduction_shared = ((fire_shared - fire_sup) / fire_shared) * 100# type: ignore
                logger.info(f"   - Shared Fire Collisions : {fire_shared:.3f} (-{fire_reduction_shared:.1f}%)")

        if 'total_infeasible_states' in summary.columns:
            infeasible_sup = summary.loc['Supervised', 'total_infeasible_states']
            logger.info(f"   - Supervised Infeasible States : {infeasible_sup:.3f}")

            if 'Unsupervised' in summary.index:
                infeasible_unsup = summary.loc['Unsupervised', 'total_infeasible_states']
                infeasible_reduction = ((infeasible_unsup - infeasible_sup) / infeasible_unsup) * 100# type: ignore
                logger.info(f"   - Unsupervised Infeasible States : {infeasible_unsup:.3f} (-{infeasible_reduction:.1f}%)")

            if 'Shared' in summary.index:
                infeasible_shared = summary.loc['Shared', 'total_infeasible_states']
                infeasible_reduction_shared = ((infeasible_shared - infeasible_sup) / infeasible_shared) * 100# type: ignore
                logger.info(f"   - Shared Infeasible States : {infeasible_shared:.3f} (-{infeasible_reduction_shared:.1f}%)")

        if 'total_damage' in summary.columns:
            damage_sup = summary.loc['Supervised', 'total_damage']
            logger.info(f"   - Supervised Total Damage : {damage_sup:.3f}")

            if 'Unsupervised' in summary.index:
                damage_unsup = summary.loc['Unsupervised', 'total_damage']
                damage_reduction = ((damage_unsup - damage_sup) / damage_unsup) * 100# type: ignore
                logger.info(f"   - Unsupervised Total Damage : {damage_unsup:.3f} (-{damage_reduction:.1f}%)")

            if 'Shared' in summary.index:
                damage_shared = summary.loc['Shared', 'total_damage']
                damage_reduction_shared = ((damage_shared - damage_sup) / damage_shared) * 100# type: ignore
                logger.info(f"   - Shared Total Damage : {damage_shared:.3f} (-{damage_reduction_shared:.1f}%)")

    logger.info("="*60 + "\n")

if __name__ == '__main__':
    setup_logging(False)
    # warmup_numba_compilers()
    
    # config = MapConfig(num_rooms=12, num_victims=20, min_room_length=4, min_room_width=4, max_room_length=8, max_room_width=8, max_tunnel_thickness=3, min_tunnel_thickness=2
    #                    , initial_fire_points=1, room_vulnerability_probability=0.1, tunnel_vulnerability_probability=0.1, fire_spread_rate=0.01, fire_duration=100)
    config = MapConfig(num_rooms=12, num_victims=20, min_room_length=4, min_room_width=4, max_room_length=8, max_room_width=8, max_tunnel_thickness=3, min_tunnel_thickness=2
                       , initial_fire_points=1, room_vulnerability_probability=0.3, tunnel_vulnerability_probability=0.4, room_vulnerability_severity=0.2, tunnel_vulnerability_severity=0.2,fire_spread_rate=0.01, fire_duration=100)
    
    starting_seed = 276895418
    # gather_data(width=60, height=60, stoppingCondition=StoppingCondition.ALL_VICTIMS_FOUND, num_runs=108, steps_per_run=300, 
    #             config=config, with_unsupervised=False, with_shared_map=False, seed_start=starting_seed)
    
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_filename = "event_resolve_2026-06-17_15-40-49_108x_60x60_1x_fire_sim_res.csv"
    csv_path = os.path.join(parent_dir, 'sim_data', csv_filename)
    evaluate_csv_data(csv_path)

    # csv_filename = "large_60x60_216x_3x_fire.csv"
    # csv_path = os.path.join(parent_dir, 'sim_data', csv_filename)
    # evaluate_csv_data(csv_path)

    # csv_filename = "large_60x60_216x_5x_fire.csv"
    # csv_path = os.path.join(parent_dir, 'sim_data', csv_filename)
    # evaluate_csv_data(csv_path)

    # csv_filename = "large_60x60_216x_10x_fire.csv"
    # csv_path = os.path.join(parent_dir, 'sim_data', csv_filename)
    # evaluate_csv_data(csv_path)
    # plot_boxes(csv_path)