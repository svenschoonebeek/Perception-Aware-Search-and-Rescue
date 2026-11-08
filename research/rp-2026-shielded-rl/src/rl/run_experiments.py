"""Unattended ablation batch: runs many configs back-to-back, logs a summary table.

Built from the 2-agent research finding that the ~0.63 coverage plateau is most
likely a discount-horizon bug (gamma=0.99 -> effective horizon ~100 << 700-step
episodes). This sweeps the high-ROI levers in one go so we learn what works:
  gamma (0.99/0.997/0.999) x random-start (off/on), plus reward / victim-count /
  horizon variations, then scales the best-guess config to 25x25 and 30x30.

Results stream to runs/batch_<ts>/results.csv after EVERY run (partial results
survive), and a formatted table prints at the end. Each run also saves its
best-by-success model in its own subdir.

Usage:
  N_ENVS=64 uv run python -m src.rl.run_experiments            # full batch (~2.5-3 h)
  N_ENVS=64 uv run python -m src.rl.run_experiments --smoke    # 150k steps each (pipeline check)
  N_ENVS=4  uv run python -m src.rl.run_experiments --smoke --first 2   # tiny local validation
"""
import gc
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize

from src.rl.env import SaREnv, N_STACK
from src.rl.reward import RewardWeights
from src.rl.train import GridCNN
from src.rl.callbacks import SuccessRateCallback, EntCoefAnneal
from src.rl.configs import sar_config
from src.rl.run_compare import open_config  # train on the SAME open maps we evaluate on

N_ENVS = int(os.environ.get("N_ENVS", "32"))
SMOKE = "--smoke" in sys.argv
# Optional: run only the first K experiments (local validation).
FIRST = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--first=")), None)
if "--first" in sys.argv:
    FIRST = int(sys.argv[sys.argv.index("--first") + 1])

MONITOR_KWARGS = dict(info_keywords=("outcome", "victims_found", "total_victims", "area_explored"))


def lin(start, end):
    """SB3 schedule callable: progress_remaining goes 1 -> 0."""
    return lambda pr: end + pr * (start - end)


def make_ppo(vec_env, seed, gamma):
    """v5 stabilised PPO; gamma is per-experiment (the key swept lever)."""
    return PPO(
        "MlpPolicy", vec_env,
        policy_kwargs=dict(
            features_extractor_class=GridCNN,
            features_extractor_kwargs=dict(features_dim=256),
            normalize_images=False,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ortho_init=True,
        ),
        learning_rate=lin(2.5e-4, 0.0),
        n_steps=1024, batch_size=2048, n_epochs=4,
        gamma=gamma, gae_lambda=0.95,
        clip_range=0.15, clip_range_vf=0.2,
        ent_coef=0.03, vf_coef=0.5, max_grad_norm=0.5,
        target_kl=0.02, normalize_advantage=True,
        verbose=0, seed=seed, device="auto",
    )


def train_one(exp, out_root):
    """Run one experiment; return a result row with final-window metrics."""
    label = exp["label"]
    steps = int(os.environ.get("SMOKE_STEPS", "150000")) if SMOKE else exp["steps"]
    out = out_root / label
    out.mkdir(parents=True, exist_ok=True)

    reward = RewardWeights(
        w_v=10.0, w_e=0.0, w_s=0.005, w_c_t=0.1, w_c_v=1.0,
        w_h_v=0.0, w_h_f_flam=0.0, w_h_f_burn=0.0,
        w_succ=50.0, w_tout=0.0, w_novelty=0.0,
        w_phi=exp["w_phi"], w_cov_term=exp["w_cov"],
        use_coverage_potential=exp["use_cov_pot"],
    )
    cfg = dict(steps=steps, **{k: exp[k] for k in
               ("size", "gamma", "random_start", "w_phi", "w_cov", "horizon",
                "num_victims", "corridor", "use_cov_pot")})
    (out / "config.txt").write_text(repr(cfg))

    t0 = time.time()
    env_kwargs = dict(
        width=exp["size"], height=exp["size"], max_episode_steps=exp["horizon"],
        config=open_config(exp["size"], exp["num_victims"]),
        reward_weights=reward, gamma=exp["gamma"], random_start=exp["random_start"],
    )
    vec_env = make_vec_env(SaREnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv,
                           env_kwargs=env_kwargs, monitor_kwargs=MONITOR_KWARGS)
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=exp["gamma"])
    vnorm = vec_env
    if N_STACK > 1:
        vec_env = VecFrameStack(vec_env, n_stack=N_STACK, channels_order="first")

    model = make_ppo(vec_env, seed=0, gamma=exp["gamma"])
    sr_cb = SuccessRateCallback(best_save_dir=out, vecnormalize=vnorm, verbose=0)
    callbacks = [sr_cb, EntCoefAnneal(start=0.03, end=0.005, total_timesteps=steps)]
    try:
        model.learn(total_timesteps=steps, callback=callbacks)
        model.save(out / "ppo_model")
    finally:
        try:
            vec_env.close()
        except Exception:
            pass

    def mean(dq):
        return round(sum(dq) / len(dq), 4) if len(dq) else 0.0

    row = dict(
        label=label, size=exp["size"], steps=steps, gamma=exp["gamma"],
        rs=int(exp["random_start"]), w_phi=exp["w_phi"], w_cov=exp["w_cov"],
        horizon=exp["horizon"], victims=exp["num_victims"], pot=("cov" if exp["use_cov_pot"] else "front"),
        final_success=mean(sr_cb._success), final_victims=mean(sr_cb._victims_frac),
        final_coverage=mean(sr_cb._coverage_frac), best_success=round(sr_cb.best, 4),
        minutes=round((time.time() - t0) / 60, 1),
    )
    del model, vec_env
    gc.collect()
    return row


# ---- Experiment matrix ----
# Core 3x2 grid (gamma x random-start) on 20x20 isolates the root-cause levers,
# then victim-count / horizon / reward / scale variations on the best-guess config.
def E(label, size=20, steps=2_000_000, gamma=0.99, random_start=False,
      w_phi=30.0, w_cov=50.0, horizon=700, num_victims=6, corridor=1, use_cov_pot=True):
    return dict(label=label, size=size, steps=steps, gamma=gamma, random_start=random_start,
                w_phi=w_phi, w_cov=w_cov, horizon=horizon, num_victims=num_victims,
                corridor=corridor, use_cov_pot=use_cov_pot)


EXPERIMENTS = [
    # --- core gamma x random-start grid (20x20, 6 victims) ---
    E("01_g099_rs0",  gamma=0.99),                          # control: reproduces ~0.63
    E("02_g0997_rs0", gamma=0.997),
    E("03_g0999_rs0", gamma=0.999),
    E("04_g099_rs1",  gamma=0.99,  random_start=True),
    E("05_g0997_rs1", gamma=0.997, random_start=True),
    E("06_g0999_rs1", gamma=0.999, random_start=True),
    # --- variations on the best-guess (g0997 + random start) ---
    E("07_g0997_rs1_v10",  gamma=0.997, random_start=True, num_victims=10),   # real target victim count
    E("08_g0999_rs1_h1000", gamma=0.999, random_start=True, horizon=1000),    # match eff. horizon to budget
    E("09_g0997_rs1_phi50", gamma=0.997, random_start=True, w_phi=50.0, w_cov=80.0),  # stronger coverage pull
    E("10_g0997_rs1_frontier", gamma=0.997, random_start=True, use_cov_pot=False),    # frontier-distance potential
    E("11_g0997_rs1_4M", gamma=0.997, random_start=True, steps=4_000_000),    # scale/stability check
    # --- scale the best-guess config up ---
    E("12_g0999_rs1_25x25", size=25, gamma=0.999, random_start=True, horizon=900, steps=3_000_000, num_victims=10),
    E("13_g0999_rs1_30x30", size=30, gamma=0.999, random_start=True, horizon=1200, steps=4_000_000, num_victims=10),
]

# Single frame-stack PPO matched to the LSTM's 20x20 config (gamma 0.999, random start,
# horizon 1000, 6 victims) -- the "no LSTM" baseline for the memory ablation table.
# Select this instead of EXPERIMENTS with the env var ABLATE=1.
ABLATION = [
    E("ablate_framestack_20x20", size=20, gamma=0.999, random_start=True,
      horizon=1000, num_victims=6, steps=2_000_000),
]


def print_table(rows):
    cols = ["label", "size", "gamma", "rs", "victims", "pot", "horizon",
            "final_coverage", "final_victims", "final_success", "best_success", "minutes"]
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0)) for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    print("\n" + line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def write_csv(rows, path):
    cols = ["label", "size", "steps", "gamma", "rs", "w_phi", "w_cov", "horizon",
            "victims", "pot", "final_coverage", "final_victims", "final_success",
            "best_success", "minutes"]
    lines = [",".join(cols)]
    lines += [",".join(str(r.get(c, "")) for c in cols) for r in rows]
    path.write_text("\n".join(lines))


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path("runs") / f"batch_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    base = ABLATION if os.environ.get("ABLATE") else EXPERIMENTS
    exps = base[:FIRST] if FIRST else base
    print(f"[batch] {len(exps)} experiments, n_envs={N_ENVS}, smoke={SMOKE}, "
          f"ablate={bool(os.environ.get('ABLATE'))}, out={out_root}", flush=True)

    rows = []
    t_all = time.time()
    for i, exp in enumerate(exps, 1):
        print(f"\n{'='*72}\n[batch] {i}/{len(exps)}: {exp['label']}  "
              f"(gamma={exp['gamma']}, rs={int(exp['random_start'])}, size={exp['size']})\n{'='*72}", flush=True)
        try:
            row = train_one(exp, out_root)
        except Exception:
            import traceback
            traceback.print_exc()
            row = dict(label=exp["label"], size=exp["size"], gamma=exp["gamma"],
                       rs=int(exp["random_start"]), final_coverage="ERR")
        rows.append(row)
        write_csv(rows, out_root / "results.csv")        # stream partial results
        print(f"[batch] {exp['label']}: coverage={row.get('final_coverage')} "
              f"victims={row.get('final_victims')} success={row.get('final_success')} "
              f"best={row.get('best_success')} ({row.get('minutes')} min)", flush=True)
        print_table(rows)

    print(f"\n[batch] ALL DONE in {(time.time()-t_all)/3600:.2f} h -> {out_root/'results.csv'}", flush=True)
    print_table(rows)


if __name__ == "__main__":
    main()
