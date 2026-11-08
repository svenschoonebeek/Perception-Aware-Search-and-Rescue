"""Small-env diagnostic for the pure-RL baseline.

Trains PPO on a deliberately trivial instance — 12x12 grid, 1 victim, no fire,
no vulnerability — to test whether the policy can learn to complete a rescue at
all. Used to confirm that a failure to solve the full 40x40 task is caused by
scale/sparsity rather than a broken reward or observation.

Result (seed 0, 200k steps): ep_rew_mean +4.5 -> +23, ep_len_mean 76 -> 34 —
the policy does learn to find and rescue efficiently, so the reward + algorithm
are sound. Reuses GridCNN + the PPO setup from train.py.
"""
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from src.rl.env import SaREnv
from src.rl.train import GridCNN
from src.constants import MapConfig


def main(total_timesteps: int = 200_000, n_envs: int = 8):
    # Trivial, fully-learnable instance: small grid, single victim, no hazards.
    small_cfg = MapConfig(
        num_rooms=2, num_victims=1, num_agents=0,
        initial_fire_points=0, fire_spread_rate=0.0,
        room_vulnerability_probability=0.0, tunnel_vulnerability_probability=0.0,
        min_room_width=3, max_room_width=5,
        min_room_length=3, max_room_length=5,
        max_tunnel_thickness=1,
    )
    env_kwargs = dict(width=12, height=12, config=small_cfg, max_episode_steps=100)

    vec_env = make_vec_env(SaREnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)

    model = PPO(
        "MlpPolicy", vec_env,
        policy_kwargs=dict(
            features_extractor_class=GridCNN,
            features_extractor_kwargs=dict(features_dim=128),
            normalize_images=False,
        ),
        n_steps=1024, batch_size=64, ent_coef=0.01, verbose=1, seed=0,
    )
    model.learn(total_timesteps=total_timesteps)

    os.makedirs("runs", exist_ok=True)
    model.save("runs/ppo_diag_small")
    print("saved diagnostic model to runs/ppo_diag_small.zip")


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    main(total_timesteps=steps)
