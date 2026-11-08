import concurrent.futures
import logging
import os
from datetime import datetime
from time import time

import numpy as np
import pandas as pd

from .constants import Capabilities, MapConfig, StoppingCondition
from .simulator import Simulator
from .supervisor import Supervisor
from .agents.rescue import RescueAgent
from .util import setup_logging, warmup_numba_compilers

logger = logging.getLogger(__name__)

WIDTH, HEIGHT = 45, 45
NUM_STEPS = 300
AGENT_COUNTS = [0, 2, 4, 6, 8]

CONFIG = MapConfig(
    num_rooms=10, num_victims=10,
    room_vulnerability_probability=0.4, room_vulnerability_severity=0.2,
    tunnel_vulnerability_probability=0.1,
    min_room_length=5, min_room_width=5,
    max_room_length=10, max_room_width=10,
    min_tunnel_thickness=2, max_tunnel_thickness=3,
    fire_spread_rate=0.02, fire_duration=100, initial_fire_points=1,
)


def _agent_positions(n: int) -> list[tuple[int, int]]:
    """Fill top-left corner column by column, 3 rows per column."""
    positions = []
    col, row = 0, 0
    for _ in range(n):
        positions.append((col, row))
        row += 1
        if row == 3:
            row = 0
            col += 1
    return positions


def run_single(seed: int, num_agents: int) -> dict:
    setup_logging(False)
    pid = os.getpid()
    logger.info(f"[PID {pid}] seed={seed} num_agents={num_agents} starting")

    sim = Simulator(WIDTH, HEIGHT, StoppingCondition.ALL_VICTIMS_RESCUED)
    sim.generate_ground_truth(CONFIG, seed)

    for i, (x, y) in enumerate(_agent_positions(num_agents)):
        sim.add_agent(RescueAgent(
            f"rescuer{i+1}", x, y, WIDTH, HEIGHT,
            0.01, 0.9, 4, False, None,
            Capabilities(exploration=1.0, rescuing=1.0),
        ))

    sim.set_supervisor(Supervisor(WIDTH, HEIGHT, 0.01, 8, True))
    sim.run(NUM_STEPS)

    s = sim.metrics.history[-1]
    logger.info(f"[PID {pid}] seed={seed} num_agents={num_agents} done — saved={s.victims_saved} dead={s.victims_dead} step={s.step}")
    return {
        "seed": seed,
        "num_agents": num_agents,
        "outcome": s.outcome.value,
        "victims_saved": s.victims_saved,
        "victims_dead": s.victims_dead,
        "total_victims": s.total_victims,
        "time_to_first_saved": s.time_to_first_saved,
        "step": s.step,
    }


def gather_data(num_runs: int = 20, seed_start: int | None = None):
    setup_logging(False)
    warmup_numba_compilers()

    if seed_start is None:
        seed_start = int(np.random.random() * 1_000_000_000)

    seeds = [seed_start + i for i in range(num_runs)]
    tasks = [(seed, n) for n in AGENT_COUNTS for seed in seeds]
    total = len(tasks)

    logger.info(f"seed_start={seed_start}")
    logger.info(f"Queuing {total} simulations ({len(AGENT_COUNTS)} agent counts × {num_runs} seeds)")

    cores = max(1, (os.cpu_count() or 2) - 2)
    logger.info(f"Using {cores} worker processes")

    results = []
    start = time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        futures = {executor.submit(run_single, seed, n): (seed, n) for seed, n in tasks}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(future.result())
            logger.info(f"Progress: {done}/{total}")

    elapsed = time() - start
    logger.info(f"All simulations done in {elapsed:.1f}s")

    df = pd.DataFrame(results).sort_values(["num_agents", "seed"]).reset_index(drop=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"rescue_results_{timestamp}.xlsx"
    df.to_excel(out_path, index=False)
    logger.info(f"Results written to {out_path}")

    summary = df.groupby("num_agents")[["victims_saved", "victims_dead", "time_to_first_saved", "step"]].mean().round(2)
    summary.columns = ["avg_saved", "avg_dead", "avg_time_to_first_saved", "avg_steps"]
    logger.info(f"\nSummary (averages over {num_runs} seeds):\n{summary.to_string()}")


def run_single_ablation(seed: int, trust_aware: bool) -> dict:
    setup_logging(False)
    pid = os.getpid()
    arm = "trust_aware" if trust_aware else "trust_blind"
    logger.info(f"[PID {pid}] seed={seed} arm={arm} starting")

    sim = Simulator(WIDTH, HEIGHT, StoppingCondition.ALL_VICTIMS_RESCUED)
    sim.generate_ground_truth(CONFIG, seed)

    for i, (x, y) in enumerate(_agent_positions(4)):
        sim.add_agent(RescueAgent(
            f"rescuer{i+1}", x, y, WIDTH, HEIGHT,
            0.01, 0.9, 4, False, None,
            Capabilities(exploration=1.0, rescuing=1.0),
            trust_aware=trust_aware,
        ))

    sim.set_supervisor(Supervisor(WIDTH, HEIGHT, 0.01, 8, True))
    sim.run(NUM_STEPS)

    s = sim.metrics.history[-1]
    all_v = sim.all_victims

    final_trust = float(np.mean([v.trust for v in all_v])) if all_v else float("nan")

    per_avg = [v._trust_sum / v._trust_ticks for v in all_v if v._trust_ticks > 0]
    avg_trust = float(np.mean(per_avg)) if per_avg else float("nan")

    ttts = [v._ticks_to_threshold for v in all_v if v._ticks_to_threshold is not None]
    mean_ticks_to_threshold = float(np.mean(ttts)) if ttts else float("nan")
    n_crossed = len(ttts)

    total_trust_drops = float(np.mean([v._trust_drops for v in all_v])) if all_v else float("nan")

    ff = [v._follow_ticks / v._active_ticks for v in all_v if v._active_ticks > 0]
    follow_fraction = float(np.mean(ff)) if ff else float("nan")

    total_err_sum = sum(v._est_err_sum for v in all_v)
    total_err_count = sum(v._est_err_count for v in all_v)
    mean_est_error = total_err_sum / total_err_count if total_err_count > 0 else float("nan")

    total_signed_err_sum = sum(v._signed_err_sum for v in all_v)
    mean_signed_error = total_signed_err_sum / total_err_count if total_err_count > 0 else float("nan")

    total_overtrust_ticks = sum(v._overtrust_ticks for v in all_v)
    total_undertrust_ticks = sum(v._undertrust_ticks for v in all_v)
    overtrust_fraction = total_overtrust_ticks / total_err_count if total_err_count > 0 else float("nan")
    undertrust_fraction = total_undertrust_ticks / total_err_count if total_err_count > 0 else float("nan")

    logger.info(f"[PID {pid}] seed={seed} arm={arm} done — saved={s.victims_saved} dead={s.victims_dead} step={s.step}")
    return {
        "seed": seed,
        "arm": arm,
        "num_agents": 4,
        "outcome": s.outcome.value,
        "victims_saved": s.victims_saved,
        "victims_dead": s.victims_dead,
        "total_victims": s.total_victims,
        "time_to_first_saved": s.time_to_first_saved,
        "step": s.step,
        "final_trust": final_trust,
        "avg_trust": avg_trust,
        "mean_ticks_to_threshold": mean_ticks_to_threshold,
        "n_crossed": n_crossed,
        "total_trust_drops": total_trust_drops,
        "follow_fraction": follow_fraction,
        "mean_est_error": mean_est_error,
        "mean_signed_error": mean_signed_error,
        "overtrust_fraction": overtrust_fraction,
        "undertrust_fraction": undertrust_fraction,
    }


def gather_ablation_data(num_runs: int = 100, seed_start: int | None = None):
    setup_logging(False)
    warmup_numba_compilers()

    if seed_start is None:
        seed_start = int(np.random.random() * 1_000_000_000)

    seeds = [seed_start + i for i in range(num_runs)]
    tasks = [(seed, aware) for seed in seeds for aware in (True, False)]
    total = len(tasks)

    logger.info(f"seed_start={seed_start}")
    logger.info(f"Queuing {total} simulations (2 arms × {num_runs} seeds)")

    cores = max(1, (os.cpu_count() or 2) - 2)
    logger.info(f"Using {cores} worker processes")

    results = []
    start = time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        futures = {executor.submit(run_single_ablation, seed, aware): (seed, aware) for seed, aware in tasks}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(future.result())
            logger.info(f"Progress: {done}/{total}")

    elapsed = time() - start
    logger.info(f"All simulations done in {elapsed:.1f}s")

    df = pd.DataFrame(results).sort_values(["seed", "arm"]).reset_index(drop=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"rescue_ablation_{timestamp}.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Results written to {out_path}")

    print(df.to_string())
    return df


if __name__ == "__main__":
   #  gather_data(num_runs=100, seed_start=510656422)
    gather_ablation_data(num_runs=20, seed_start=510656422)
