"""LSTM (RecurrentPPO) batch: the last credible lever for the coverage plateau.

The 13-run PPO ablation proved flat PPO+frame-stack caps at ~60% coverage no
matter the gamma / reward / horizon / start. Per the literature (DRQN, POPGym,
Active Neural SLAM), the standard fix for "agent can't systematically sweep a
POMDP" is recurrent memory. This swaps frame-stacking for an LSTM hidden state
(RecurrentPPO, CnnLstmPolicy) keeping the same GridCNN extractor underneath.

First config mirrors the best PPO row (08: gamma=0.999, horizon=1000) for a
direct apples-to-apples comparison. Results stream to runs/lstm_<ts>/results.csv.

Usage:
  N_ENVS=64 uv run python run_lstm.py            # full LSTM batch
  N_ENVS=64 uv run python run_lstm.py --smoke    # short pipeline check
  N_ENVS=2  uv run python run_lstm.py --smoke --first 1   # tiny local validation
"""
import gc
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    sys.exit("sb3-contrib not installed. Run:  uv add sb3-contrib   (then re-run)")

from src.rl.env import SaREnv
from src.rl.reward import RewardWeights
from src.rl.train import GridCNN
from src.rl.callbacks import SuccessRateCallback, EntCoefAnneal
from src.rl.configs import sar_config
from src.rl.run_compare import open_config  # train on the SAME open maps we evaluate on

N_ENVS = int(os.environ.get("N_ENVS", "32"))
N_STEPS = int(os.environ.get("N_STEPS", "512"))   # shorter rollout (recurrent is memory-heavy)
SMOKE = "--smoke" in sys.argv
FIRST = None
if "--first" in sys.argv:
    FIRST = int(sys.argv[sys.argv.index("--first") + 1])

MONITOR_KWARGS = dict(info_keywords=("outcome", "victims_found", "total_victims", "area_explored"))


def lin(start, end):
    return lambda pr: end + pr * (start - end)


def make_rppo(vec_env, seed, gamma, enable_critic_lstm):
    """RecurrentPPO: GridCNN extractor -> LSTM -> policy/value heads."""
    return RecurrentPPO(
        "CnnLstmPolicy", vec_env,
        policy_kwargs=dict(
            features_extractor_class=GridCNN,
            features_extractor_kwargs=dict(features_dim=256),
            normalize_images=False,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ortho_init=True,
            lstm_hidden_size=256,
            n_lstm_layers=1,
            enable_critic_lstm=enable_critic_lstm,
            shared_lstm=not enable_critic_lstm,   # share if critic LSTM disabled
        ),
        learning_rate=lin(2.5e-4, 0.0),
        n_steps=N_STEPS, batch_size=2048, n_epochs=4,
        gamma=gamma, gae_lambda=0.95,
        clip_range=0.15, clip_range_vf=0.2,
        ent_coef=0.03, vf_coef=0.5, max_grad_norm=0.5,
        target_kl=0.02, normalize_advantage=True,
        verbose=0, seed=seed, device="auto",
    )


def train_one(exp, out_root):
    label = exp["label"]
    steps = int(os.environ.get("SMOKE_STEPS", "150000")) if SMOKE else exp["steps"]
    out = out_root / label
    out.mkdir(parents=True, exist_ok=True)

    reward = RewardWeights(
        w_v=10.0, w_e=0.0, w_s=0.005, w_c_t=0.1, w_c_v=1.0,
        w_h_v=0.0, w_h_f_flam=0.0, w_h_f_burn=0.0,
        w_succ=50.0, w_tout=0.0, w_novelty=0.0,
        w_phi=exp["w_phi"], w_cov_term=exp["w_cov"], use_coverage_potential=True,
    )
    (out / "config.txt").write_text(repr({**exp, "steps": steps, "lstm": True}))

    t0 = time.time()
    env_kwargs = dict(
        width=exp["size"], height=exp["size"], max_episode_steps=exp["horizon"],
        config=open_config(exp["size"], exp["num_victims"]),
        reward_weights=reward, gamma=exp["gamma"], random_start=exp["random_start"],
    )
    # NO VecFrameStack: the LSTM hidden state is the memory (obs stays 11-channel).
    vec_env = make_vec_env(SaREnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv,
                           env_kwargs=env_kwargs, monitor_kwargs=MONITOR_KWARGS,
                           seed=exp["seed"])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=exp["gamma"])

    model = make_rppo(vec_env, seed=exp["seed"], gamma=exp["gamma"],
                      enable_critic_lstm=exp["enable_critic_lstm"])
    sr_cb = SuccessRateCallback(best_save_dir=out, vecnormalize=vec_env, verbose=0)
    callbacks = [sr_cb, EntCoefAnneal(start=0.03, end=0.005, total_timesteps=steps)]
    try:
        model.learn(total_timesteps=steps, callback=callbacks)
        model.save(out / "rppo_model")
    finally:
        try:
            vec_env.close()
        except Exception:
            pass

    def mean(dq):
        return round(sum(dq) / len(dq), 4) if len(dq) else 0.0

    row = dict(
        label=label, size=exp["size"], steps=steps, gamma=exp["gamma"], seed=exp["seed"],
        rs=int(exp["random_start"]), horizon=exp["horizon"], victims=exp["num_victims"],
        critic_lstm=int(exp["enable_critic_lstm"]),
        final_coverage=mean(sr_cb._coverage_frac), final_victims=mean(sr_cb._victims_frac),
        final_success=mean(sr_cb._success), best_success=round(sr_cb.best, 4),
        minutes=round((time.time() - t0) / 60, 1),
    )
    del model, vec_env
    gc.collect()
    return row


def E(label, size=20, steps=2_000_000, gamma=0.999, random_start=True, seed=0,
      w_phi=30.0, w_cov=50.0, horizon=1000, num_victims=6, enable_critic_lstm=True):
    return dict(label=label, size=size, steps=steps, gamma=gamma, random_start=random_start,
                seed=seed, w_phi=w_phi, w_cov=w_cov, horizon=horizon, num_victims=num_victims,
                enable_critic_lstm=enable_critic_lstm)


# Day 2: confirm the LSTM win is real (3 seeds), test if it keeps climbing (4M),
# and whether memory scales better than flat PPO (which hit only 0.53 on 25x25).
EXPERIMENTS = [
    E("S0_2M", seed=0),                                              # reproduce seed-0 win (0.689)
    E("S1_2M", seed=1),                                              # confirm across seeds
    E("S2_2M", seed=2),
    E("long_4M", seed=0, steps=4_000_000),                          # is LSTM converged at 2M or still climbing?
    E("scale_25x25_3M", seed=0, size=25, horizon=900, num_victims=10, steps=3_000_000),  # does memory scale?
]

# 20/25/30 scaling study (retrain-per-size; horizon ~ map area, victims ~ area).
# Select this set instead of EXPERIMENTS with the env var SCALE=1.
SCALE_EXPERIMENTS = [
    E("scale_20x20", seed=0, size=20, horizon=1000, num_victims=6,  steps=2_000_000),
    E("scale_25x25", seed=0, size=25, horizon=1400, num_victims=8,  steps=3_000_000),
    E("scale_30x30", seed=0, size=30, horizon=2000, num_victims=10, steps=4_000_000),
]


def print_table(rows):
    cols = ["label", "size", "seed", "gamma", "rs", "horizon", "victims", "critic_lstm",
            "final_coverage", "final_victims", "final_success", "best_success", "minutes"]
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0)) for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    print("\n" + line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def write_csv(rows, path):
    cols = ["label", "size", "steps", "seed", "gamma", "rs", "horizon", "victims", "critic_lstm",
            "final_coverage", "final_victims", "final_success", "best_success", "minutes"]
    path.write_text("\n".join([",".join(cols)] +
                              [",".join(str(r.get(c, "")) for c in cols) for r in rows]))


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path("runs") / f"lstm_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    base = SCALE_EXPERIMENTS if os.environ.get("SCALE") else EXPERIMENTS
    exps = base[:FIRST] if FIRST else base
    print(f"[lstm] {len(exps)} experiments, n_envs={N_ENVS}, n_steps={N_STEPS}, "
          f"smoke={SMOKE}, scale={bool(os.environ.get('SCALE'))}, out={out_root}", flush=True)

    rows = []
    t_all = time.time()
    for i, exp in enumerate(exps, 1):
        print(f"\n{'='*72}\n[lstm] {i}/{len(exps)}: {exp['label']}\n{'='*72}", flush=True)
        try:
            row = train_one(exp, out_root)
        except Exception:
            import traceback
            traceback.print_exc()
            row = dict(label=exp["label"], final_coverage="ERR")
        rows.append(row)
        write_csv(rows, out_root / "results.csv")
        print(f"[lstm] {exp['label']}: coverage={row.get('final_coverage')} "
              f"victims={row.get('final_victims')} success={row.get('final_success')} "
              f"best={row.get('best_success')} ({row.get('minutes')} min)", flush=True)
        print_table(rows)

    print(f"\n[lstm] ALL DONE in {(time.time()-t_all)/3600:.2f} h -> {out_root/'results.csv'}", flush=True)
    print_table(rows)


if __name__ == "__main__":
    main()
