"""Parallel held-out comparison: shard run_compare across all CPU cores, then merge.

run_compare runs episodes sequentially, which wastes a many-core box on the CPU-bound
MPC agents. This splits the seed range into shards and launches one subprocess per
(agent, shard), capped at --jobs concurrent workers, then concatenates each agent's
per-shard episode CSVs (in seed order) into the canonical runs/compare/<agent>_episodes.csv
and rebuilds summary.csv. Finally it runs the paired-difference / dominance analysis.

Children run CPU-only (the RL net is tiny; this frees the GPU for training and avoids
100 procs fighting over one device) with single-threaded BLAS so 1 shard == 1 core.

Usage (250 seeds, 5 agents, 30x30, sharded 10 seeds/proc across all cores):
  uv run python -m src.rl.run_compare_parallel --agents rl,mpc,tmpc,shield,shieldc \
      --rl-model runs/lstm_XXXX/scale_30x30 --lstm --deterministic \
      --size 30 --victims 10 --horizon 500 --episodes 250 --open --start center --shard 10
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from src.rl.run_compare import METRICS, ci95


def child_env():
    """CPU-only, single-threaded BLAS so each shard is exactly one core."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""          # keep the GPU for training; RL infers on CPU
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"
    return env


def shard_cmd(args, agent, seed0, n, outdir):
    cmd = [sys.executable, "-m", "src.rl.run_compare",
           "--agents", agent, "--size", str(args.size), "--victims", str(args.victims),
           "--horizon", str(args.horizon), "--episodes", str(n), "--seed", str(seed0),
           "--corridor", str(args.corridor), "--completion", str(args.completion),
           "--start", args.start, "--outdir", outdir]
    if args.rl_model:
        cmd += ["--rl-model", args.rl_model]
    if args.lstm:
        cmd.append("--lstm")
    if args.deterministic:
        cmd.append("--deterministic")
    if args.open:
        cmd.append("--open")
    return cmd


def run_shard(args, agent, idx, seed0, n, shard_root, env):
    outdir = str(shard_root / f"{agent}_{idx:03d}")
    log = Path(outdir) / "shard.log"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        rc = subprocess.run(shard_cmd(args, agent, seed0, n, outdir),
                            stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    return agent, idx, outdir, rc


def merge_agent(agent, shard_dirs, final_dir):
    """Concatenate per-shard <agent>_episodes.csv rows (seed order) -> one csv."""
    header, rows = None, []
    for d in shard_dirs:
        p = Path(d) / f"{agent}_episodes.csv"
        if not p.exists():
            continue
        lines = p.read_text().strip().splitlines()
        header = lines[0]
        rows += lines[1:]
    if header is None:
        return None
    (final_dir / f"{agent}_episodes.csv").write_text("\n".join([header] + rows))
    data = np.array([[float(v) for v in r.split(",")] for r in rows])
    return {m: ci95(data[:, i]) for i, m in enumerate(METRICS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default="rl,mpc,tmpc,shield,shieldc")
    ap.add_argument("--rl-model", default=None)
    ap.add_argument("--lstm", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--size", type=int, default=30)
    ap.add_argument("--victims", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--episodes", type=int, default=250)
    ap.add_argument("--seed", type=int, default=777_000)
    ap.add_argument("--corridor", type=int, default=1)
    ap.add_argument("--completion", type=int, default=0)
    ap.add_argument("--start", default="center")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--outdir", default="runs/compare")
    ap.add_argument("--shard", type=int, default=10, help="seeds per subprocess")
    ap.add_argument("--jobs", type=int, default=0, help="max concurrent procs (0 = cpu_count-2)")
    ap.add_argument("--no-analyze", action="store_true")
    args = ap.parse_args()

    agents = [a.strip() for a in args.agents.split(",")]
    jobs = args.jobs or max(1, (os.cpu_count() or 4) - 2)
    final_dir = Path(args.outdir)
    shard_root = final_dir / "_shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    # Build the (agent, shard) task list: contiguous seed chunks per agent.
    tasks = []
    for agent in agents:
        for idx, off in enumerate(range(0, args.episodes, args.shard)):
            n = min(args.shard, args.episodes - off)
            tasks.append((agent, idx, args.seed + off, n))
    print(f"[parallel] {len(tasks)} shards ({len(agents)} agents x "
          f"{len(tasks)//len(agents)} chunks of <= {args.shard}), jobs={jobs}, "
          f"{args.episodes} seeds, {args.size}x{args.size}", flush=True)

    env = child_env()
    done = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(run_shard, args, a, i, s, n, shard_root, env) for (a, i, s, n) in tasks]
        for fut in futs:
            agent, idx, outdir, rc = fut.result()
            done += 1
            flag = "ok" if rc == 0 else f"FAIL rc={rc}"
            print(f"[parallel] {done}/{len(tasks)} {agent} shard {idx}: {flag}", flush=True)

    # Merge per agent + rebuild summary.csv.
    summary = final_dir / "summary.csv"
    summary.write_text(",".join(["agent", "N"] + [f"{m}_mean" for m in METRICS]
                                + [f"{m}_ci95" for m in METRICS]) + "\n")
    print("\n[parallel] merged results:", flush=True)
    for agent in agents:
        n_shards = len(range(0, args.episodes, args.shard))
        shard_dirs = [shard_root / f"{agent}_{i:03d}" for i in range(n_shards)]
        agg = merge_agent(agent, shard_dirs, final_dir)
        if agg is None:
            print(f"  {agent}: no episodes produced (check _shards/*/shard.log)", flush=True)
            continue
        n_eps = sum(1 for _ in (final_dir / f"{agent}_episodes.csv").read_text().strip().splitlines()) - 1
        with summary.open("a") as f:
            f.write(",".join([agent, str(n_eps)]
                             + [f"{agg[m][0]:.4f}" for m in METRICS]
                             + [f"{agg[m][1]:.4f}" for m in METRICS]) + "\n")
        print(f"  {agent} (N={n_eps}): " + "  ".join(f"{m}={agg[m][0]:.3f}" for m in METRICS), flush=True)

    print(f"\n[parallel] summary -> {summary}", flush=True)
    if not args.no_analyze:
        print("\n[parallel] running paired-difference / dominance analysis...\n", flush=True)
        subprocess.run([sys.executable, "-m", "src.rl.analyze_compare", "--dir", str(final_dir),
                        "--agents", args.agents])


if __name__ == "__main__":
    main()
