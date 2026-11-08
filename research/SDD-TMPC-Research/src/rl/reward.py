from dataclasses import dataclass

from src.constants import FireLevel


@dataclass
class RewardWeights:
    """Pure-RL weights"""
    w_v: float = 10.0          # + per new victim found
    w_e: float = 1.0           # + newly-explored traversable cells
    w_s: float = 0.01          # - per step
    w_c_t: float = 0.1         # - terrain collision 
    w_c_v: float = 1.0         # - victim collision
    w_h_v: float = 0.5         # - vulnerability of current cell
    w_h_f_flam: float = 0.1    # - Flammable
    w_h_f_burn: float = 5.0    # - Burning 
    w_succ: float = 20.0       # + on success
    w_tout: float = 5.0        # - on timeout


def compute_reward(
    *,
    delta_victims: int,
    delta_explored: int,
    total_traversable: int,
    delta_terrain_coll: int,
    delta_victim_coll: int,
    vulnerability_at_agent: float,
    fire_at_agent: int,
    terminated: bool,
    timeout: bool,
    weights: RewardWeights = RewardWeights(),
) -> float:
    fire_term = 0.0
    if fire_at_agent == FireLevel.FLAMMABLE:
        fire_term = weights.w_h_f_flam
    elif fire_at_agent == FireLevel.BURNING:
        fire_term = weights.w_h_f_burn

    explore_term = 0.0
    if total_traversable > 0:
        explore_term = weights.w_e * delta_explored / total_traversable

    return (
        weights.w_v * delta_victims
        + explore_term
        - weights.w_s
        - weights.w_c_t * delta_terrain_coll
        - weights.w_c_v * delta_victim_coll
        - weights.w_h_v * vulnerability_at_agent
        - fire_term
        + (weights.w_succ if terminated else 0.0)
        - (weights.w_tout if timeout else 0.0)
    )
