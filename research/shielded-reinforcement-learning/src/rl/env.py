import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.rl.agent import RLAgent
from src.constants import (
    AgentAction,
    FireLevel,
    MapConfig,
    RunOutcome,
    TraversabilityLevel,
    VictimPresence,
)
from src.simulator import Simulator
from src.rl.reward import RewardWeights, compute_reward
from src.rl.frontier import nearest_frontier_distance, frontier_distance_field


N_CHANNELS = 11
N_STACK = 4  # frame-stack size for the policy (only for non LSTM policy)


def build_observation(agent) -> np.ndarray:
    """Build the (N_CHANNELS, H, W) observation 'images' from an agent's belief."""
    p = agent.perception
    h, w = agent.world_height, agent.world_width
    obs = np.zeros((N_CHANNELS, h, w), dtype=np.float32)
    obs[0] = p.traversability.matrix.astype(np.float32) # which cells are floor vs wall
    obs[1] = p.vulnerability.matrix.astype(np.float32) # hazard level per cell
    obs[2] = p.victims.matrix.astype(np.float32) # where agent saw victims
    obs[3] = p.agents.matrix.astype(np.float32) # where the agent is now
    fire = p.fire.matrix.astype(int)
    obs[4] = (fire == FireLevel.SAFE).astype(np.float32)    # safe 
    obs[5] = (fire == FireLevel.FLAMMABLE).astype(np.float32) # flammable
    obs[6] = (fire == FireLevel.BURNING).astype(np.float32) # burning
    obs[7] = (fire == FireLevel.BURNT).astype(np.float32) # burnt
    obs[8] = p.confidence.matrix.astype(np.float32) # the scan certainty per cell that decays over time
    obs[9] = agent.explored.astype(np.float32) # 0 - cell not scanned 1 - cell scaned at some point
    # Distance-to-nearest-unexplored field (a perceptual frontier gradient, from belief).
    trav = (p.traversability.matrix == TraversabilityLevel.TRAVERSIBLE)
    obs[10] = frontier_distance_field(agent.explored, trav) # BFS distance to nearest unexplored 
    return obs


class SaREnv(gym.Env):
    """
    Gymnasium environment wrapping the search-and-rescue Simulator for single-agent pure-RL training.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        width: int = 40,
        height: int = 40,
        config: MapConfig | None = None,
        max_episode_steps: int = 300,
        start_x: int = 0,
        start_y: int = 0,
        agent_decay: float = 0.01,
        agent_scan_accuracy: float = 0.9,
        agent_scan_radius: int = 3,
        agent_scan_falloff: bool = True,
        reward_weights: RewardWeights | None = None,
        gamma: float = 0.99,
        ego_crop: int = 0,
        random_start: bool = False,
    ):
        super().__init__()
        self.width = width
        self.height = height
        # "Balanced" 40x40 config: enough rooms to fill the grid, slow fire, light hazards. (not same in deployment
        # I pass sar_config(20/25/30) to overwrite)
        self.config = config or MapConfig(
            num_rooms=8, num_victims=3, num_agents=0,
            min_room_width=5, max_room_width=10,
            min_room_length=5, max_room_length=10,
            max_tunnel_thickness=2,
            initial_fire_points=1, fire_spread_rate=0.1, fire_duration=8,
            room_vulnerability_probability=0.25, room_vulnerability_severity=0.3,
            tunnel_vulnerability_probability=0.2, tunnel_vulnerability_severity=0.3,
        )
        if self.config.num_agents != 0:
            raise ValueError("SaREnv places its own agent; set config.num_agents=0")

        self.max_episode_steps = max_episode_steps
        self.start_x, self.start_y = start_x, start_y
        self.agent_kwargs = dict(
            decay=agent_decay,
            scan_accuracy=agent_scan_accuracy,
            scan_radius=agent_scan_radius,
            scan_falloff=agent_scan_falloff,
        )
        self.weights = reward_weights or RewardWeights()
        # Discount factor for potential-based shaping.
        self.gamma = gamma
        # Egocentric crop size. 0 = full-grid observation.
        self.ego_crop = ego_crop
        # Random traversable start cell each reset (to break the corner overfit).
        self.random_start = random_start

        obs_h, obs_w = (ego_crop, ego_crop) if ego_crop > 0 else (height, width)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_CHANNELS, obs_h, obs_w), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(AgentAction))

        self.sim: Simulator | None = None
        self.agent: RLAgent | None = None
        self.step_count = 0
        self._prev_explored = 0
        # Per-episode visit mask for count-based intrinsic motivation (Bellemare 2016, Andres 2025).
        self.visited: np.ndarray | None = None
        # Cached potential for Ng-1999 frontier shaping.
        self._prev_phi: float = 0.0

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.sim = Simulator(self.width, self.height)
        self.sim.generate_ground_truth(self.config, seed=seed)
        sx, sy = self.start_x, self.start_y
        if self.random_start:
            # random cell for start point
            trav = (self.sim.ground_truth.traversability.matrix == TraversabilityLevel.TRAVERSIBLE)
            free = trav & (self.sim.ground_truth.victims.matrix != VictimPresence.PRESENT)
            ys, xs = np.nonzero(free)
            if len(xs) > 0:
                i = int(self.np_random.integers(len(xs)))
                sx, sy = int(xs[i]), int(ys[i])
        self.agent = RLAgent(
            name="rl",
            x=sx,
            y=sy,
            width=self.width,
            height=self.height,
            **self.agent_kwargs,
        )
        self.sim.add_agent(self.agent)
        # first scan so perception is populated for the first observation.
        self.agent.scan(self.sim.ground_truth)
        self.step_count = 0
        self._prev_explored = int(self.agent.explored.sum())
        # Reset visit mask, mark the start cell as already visited so the first
        # novelty bonus is only earned when the agent steps onto a NEW cell.
        self.visited = np.zeros((self.height, self.width), dtype=bool)
        self.visited[self.agent.y, self.agent.x] = True
        # Cache the starting potential for the first shaping delta.
        self._prev_phi = self._potential()
        return self._build_obs(), self._build_info(curr=None)

    def step(self, action):
        assert self.agent is not None and self.sim is not None, "call reset() first"
        self.agent._next_action = AgentAction(int(action))
        self.sim.step()
        self.step_count += 1

        history = self.sim.metrics.history
        curr = history[-1]
        prev = history[-2] if len(history) >= 2 else None
        
        # since metrics are cumulative we compute the deltas by subtracting the previous step's metrics from current
        if prev is None:
            delta_victims = curr.victims_found
            delta_terr = curr.terrain_collisions.get(self.agent, 0)
            delta_vict = curr.victim_collisions.get(self.agent, 0)
        else:
            delta_victims = curr.victims_found - prev.victims_found
            delta_terr = (
                curr.terrain_collisions.get(self.agent, 0)
                - prev.terrain_collisions.get(self.agent, 0)
            )
            delta_vict = (
                curr.victim_collisions.get(self.agent, 0)
                - prev.victim_collisions.get(self.agent, 0)
            )

        explored_now = int(self.agent.explored.sum())
        delta_explored = explored_now - self._prev_explored
        self._prev_explored = explored_now

        v_at = float(self.sim.ground_truth.vulnerability[self.agent.y][self.agent.x])
        f_at = int(self.sim.ground_truth.fire[self.agent.y][self.agent.x])

        # Count-based novelty, checking if agent has been on current cell before or not
        cell = (self.agent.y, self.agent.x)
        first_visit = not bool(self.visited[cell])
        self.visited[cell] = True

        terminated = curr.outcome == RunOutcome.SUCCESS
        # hit the step budget
        timeout = (self.step_count >= self.max_episode_steps) and not terminated

        # Potential-based shaping (Ng 1999). Phi is forced to 0 at
        # terminal states (success or timeout) so the shaping cannot alter the
        # optimal policy, only accelerate learning.
        phi_next = 0.0 if (terminated or timeout) else self._potential()
        shaping = self.gamma * phi_next - self._prev_phi
        self._prev_phi = phi_next

        # Map coverage at episdeo end, which drives the terminal coverage
        # bonus, so even a timeout that covers a lot of ground teaches the agent.
        coverage_fraction = (
            float(curr.area_explored.get(self.agent, 0.0)) if (terminated or timeout) else 0.0
        )

        reward = compute_reward(
            delta_victims=delta_victims,
            delta_explored=delta_explored,
            total_traversable=curr.total_traversable,
            delta_terrain_coll=delta_terr,
            delta_victim_coll=delta_vict,
            vulnerability_at_agent=v_at,
            fire_at_agent=f_at,
            first_visit=first_visit,
            terminated=terminated,
            timeout=timeout,
            shaping=shaping,
            coverage_fraction=coverage_fraction,
            weights=self.weights,
        )

        return self._build_obs(), reward, terminated, timeout, self._build_info(curr=curr)

    def _build_obs(self) -> np.ndarray:
        obs = build_observation(self.agent)
        if self.ego_crop > 0:
            obs = self._egocentric_crop(obs)
        return obs

    def _egocentric_crop(self, obs: np.ndarray) -> np.ndarray:
        """Crop a fixed K x K window centred on the agent. Out-of-bounds padded so edges look like solid bounds."""
        k = self.ego_crop
        r = k // 2
        pad = np.zeros(N_CHANNELS, dtype=np.float32)
        pad[0] = pad[9] = pad[10] = 1.0  # traversability(wall), explored, frontier-far
        padded = np.stack([
            np.pad(obs[c], r, constant_values=float(pad[c])) for c in range(N_CHANNELS)
        ])
        ay, ax = self.agent.y, self.agent.x
        return padded[:, ay:ay + k, ax:ax + k].astype(np.float32)

    def _frontier_potential(self) -> float:
        """Following Ng-1999, compute a potential using BFS to find the frontier distance (not used in deployed version)"""
        if self.weights.w_phi == 0.0 or self.agent is None:
            return 0.0
        traversable = (
            self.agent.perception.traversability.matrix == TraversabilityLevel.TRAVERSIBLE
        )
        d = nearest_frontier_distance(
            self.agent.explored, traversable, self.agent.y, self.agent.x
        )
        return -self.weights.w_phi * d / (self.height + self.width)

    def _coverage_potential(self) -> float:
        """Phi = -w_phi * (1 - coverage), where
        coverage is the fraction of traversable cells the agent has explored.
        """
        if self.weights.w_phi == 0.0 or self.agent is None or self.sim is None:
            return 0.0
        traversable = (
            self.sim.ground_truth.traversability.matrix == TraversabilityLevel.TRAVERSIBLE
        )
        total = int(traversable.sum())
        if total == 0:
            return 0.0
        explored_trav = int((self.agent.explored & traversable).sum())
        return -self.weights.w_phi * (1.0 - explored_trav / total)

    def _potential(self) -> float:
        """Dispatch: coverage potential (v5) or nearest-frontier potential (v4)."""
        if self.weights.use_coverage_potential:
            return self._coverage_potential()
        return self._frontier_potential()

    def _build_info(self, curr) -> dict:
        # for logging and run analysis 
        if curr is None:
            return {
                "outcome": RunOutcome.IN_PROGRESS.value,
                "victims_found": 0,
                "total_victims": 0,
                "area_explored": 0.0,
                "terrain_collisions": 0,
                "victim_collisions": 0,
                "damage": 0,
                "steps_taken": 0,
            }
        return {
            "outcome": curr.outcome.value,
            "victims_found": int(curr.victims_found),
            "total_victims": int(curr.total_victims),
            "area_explored": float(curr.area_explored.get(self.agent, 0.0)),
            "terrain_collisions": int(curr.terrain_collisions.get(self.agent, 0)),
            "victim_collisions": int(curr.victim_collisions.get(self.agent, 0)),
            "damage": int(curr.damage.get(self.agent, 0)),
            "steps_taken": int(curr.step),
        }


if __name__ == "__main__":
    env = SaREnv(reward_weights=RewardWeights(
        w_phi=8.0, w_cov_term=20.0, use_coverage_potential=True))
    obs, info = env.reset(seed=42)
    print(
        f"obs shape: {obs.shape}, dtype: {obs.dtype}, "
        f"min: {obs.min():.3f}, max: {obs.max():.3f}"
    )
    total = 0.0
    last_info = info
    t = 0
    for t in range(50):
        action = env.action_space.sample()
        obs, r, term, timeout, last_info = env.step(action)
        total += r
        if term or timeout:
            break
    print(
        f"after {t+1} steps: total_reward={total:.3f}, "
        f"victims={last_info['victims_found']}/{last_info['total_victims']}, "
        f"outcome={last_info['outcome']}, "
        f"area_explored={last_info['area_explored']:.3f}"
    )
