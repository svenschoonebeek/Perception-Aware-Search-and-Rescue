from dataclasses import dataclass

from src.constants import FireLevel


@dataclass
class RewardWeights:
    """Pure-RL weights"""
    # Active in deployment 
    w_v: float = 10.0          # + per new victim found
    w_s: float = 0.005         # - per step (encourages speed)
    w_c_t: float = 0.1         # - terrain collision 
    w_c_v: float = 1.0         # - victim collision   
    w_succ: float = 50.0       # + on success 
    # Potential-based shaping weight (Ng, Harada & Russell, 1999). Scales the
    # potential Phi added to the reward via F = gamma*Phi(s') - Phi(s). 0.0 means off.
    w_phi: float = 30.0
    # Gives a bonus proportional to the coverage fraction at the end of the episode
    w_cov_term: float = 50.0
    # global coverage potential Phi = -w_phi*(1-coverage) 
    # When on, set w_novelty = w_e = 0 
    use_coverage_potential: bool = True

    # Kept but inactive for now
    w_e: float = 0.0          # + newly-explored traversable cells
    w_h_v: float = 0.0         # - vulnerability  -> ZEROED, shield handles
    w_h_f_flam: float = 0.0    # - Flammable      -> ZEROED, shield handles
    w_h_f_burn: float = 0.0    # - Burning        -> ZEROED, shield handles
    w_tout: float = 0.0       # - on timeout
    # Count-based intrinsic motivation: bonus the first time the agent
    # physically steps on a cell each episode. Movement-conditional, so it
    # directly breaks the "sit and wait" basin (Bellemare 2016; Andres 2025).
    w_novelty: float = 0.0
    
    
    


def compute_reward(
    *,
    delta_victims: int,
    delta_explored: int,
    total_traversable: int,
    delta_terrain_coll: int,
    delta_victim_coll: int,
    vulnerability_at_agent: float,
    fire_at_agent: int,
    first_visit: bool,
    terminated: bool,
    timeout: bool,
    shaping: float = 0.0,
    coverage_fraction: float = 0.0,
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
        + (weights.w_novelty if first_visit else 0.0)
        + shaping 
        + (weights.w_cov_term * coverage_fraction if (terminated or timeout) else 0.0)
    )
