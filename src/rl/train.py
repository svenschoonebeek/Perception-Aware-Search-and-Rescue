"""
Minimal sanity PPO training for the pure-RL baseline. 8 parallel envs + reward 
normalization, small CNN over the perception grid, ~500k steps. 
"""

import os

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from src.rl.env import SaREnv


class GridCNN(BaseFeaturesExtractor):
    """Small CNN ovre the (channels, H, W) perception grid."""

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim) # Let the parent class know (SB3 needs this)
        n_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # stride halves the grid
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU()) # from 6400 feature vector to the desired output size

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations)) # the forward pass obs (B, C, H, W) -> features (B, features_dim) -> linear
                                                  # -> (B, features_dim) output for the policy/value net


# TODO: add eval callback, checkpoints
def main(total_timesteps: int = 500_000, n_envs: int = 8):
    # 8 parallel envs in subprocesses: more data/sec and lower-variance gradients.
    vec_env = make_vec_env(SaREnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    # Normalize the reward stream only (obs already in [0, 1])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)

    model = PPO(
        "MlpPolicy", 
        vec_env,
        policy_kwargs=dict(
            features_extractor_class=GridCNN,
            features_extractor_kwargs=dict(features_dim=256),
            normalize_images=False,  # obs already in [0, 1]
        ),
        n_steps=1024,
        batch_size=64,
        ent_coef=0.01,
        verbose=1,
        seed=0,
    )

    model.learn(total_timesteps=total_timesteps)

    os.makedirs("runs", exist_ok=True)
    model.save("runs/ppo_sanity")
    vec_env.save("runs/vecnormalize.pkl")
    print("saved model + vecnormalize stats to runs/")


if __name__ == "__main__":
    main()
