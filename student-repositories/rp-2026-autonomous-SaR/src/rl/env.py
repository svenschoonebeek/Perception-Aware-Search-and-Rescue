import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.rl.agent import RLAgent
from src.constants import (
    AgentAction,
    FireLevel,
    MapConfig,
    RunOutcome,
)
from src.simulator import Simulator
from src.rl.reward import RewardWeights, compute_reward


N_CHANNELS = 10


class SaREnv(gym.Env):
    """
    Gymnasium environment wrapping the search-and-rescue Simulator for
    single-agent pure-RL training.
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
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.config = config or MapConfig(
            num_rooms=3,
            num_victims=5,
            num_agents=0,
            min_room_length=4, min_room_width=4,
            max_room_length=7, max_room_width=7,
            max_tunnel_thickness=1,
        )
        if self.config.num_agents != 0:
            # The wrapper places its own RLAgent; auto-placement would put
            # an extra unowned agent on the grid.
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

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(N_CHANNELS, height, width), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(AgentAction))

        self.sim: Simulator | None = None
        self.agent: RLAgent | None = None
        self.step_count = 0
        self._prev_explored = 0

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.sim = Simulator(self.width, self.height)
        self.sim.generate_ground_truth(self.config, seed=seed)
        self.agent = RLAgent(
            name="rl",
            x=self.start_x,
            y=self.start_y,
            width=self.width,
            height=self.height,
            **self.agent_kwargs,
        )
        self.sim.add_agent(self.agent)
        # First scan so perception is populated for the first observation.
        self.agent.scan(self.sim.ground_truth)
        self.step_count = 0
        self._prev_explored = int(self.agent.explored.sum())
        return self._build_obs(), self._build_info(curr=None)

    def step(self, action):
        assert self.agent is not None and self.sim is not None, "call reset() first"
        self.agent._next_action = AgentAction(int(action))
        self.sim.step()
        self.step_count += 1

        history = self.sim.metrics.history
        curr = history[-1]
        prev = history[-2] if len(history) >= 2 else None

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

        terminated = curr.outcome == RunOutcome.SUCCESS
        # hit the step budget (reported in gymnasium's 4th step() return value)
        timeout = (self.step_count >= self.max_episode_steps) and not terminated

        reward = compute_reward(
            delta_victims=delta_victims,
            delta_explored=delta_explored,
            total_traversable=curr.total_traversable,
            delta_terrain_coll=delta_terr,
            delta_victim_coll=delta_vict,
            vulnerability_at_agent=v_at,
            fire_at_agent=f_at,
            terminated=terminated,
            timeout=timeout,
            weights=self.weights,
        )

        return self._build_obs(), reward, terminated, timeout, self._build_info(curr=curr)

    def _build_obs(self) -> np.ndarray:
        p = self.agent.perception
        obs = np.zeros((N_CHANNELS, self.height, self.width), dtype=np.float32)
        obs[0] = p.traversability.matrix.astype(np.float32)
        obs[1] = p.vulnerability.matrix.astype(np.float32)
        obs[2] = p.victims.matrix.astype(np.float32)
        obs[3] = p.agents.matrix.astype(np.float32)
        fire = p.fire.matrix.astype(int)
        obs[4] = (fire == FireLevel.SAFE).astype(np.float32)
        obs[5] = (fire == FireLevel.FLAMMABLE).astype(np.float32)
        obs[6] = (fire == FireLevel.BURNING).astype(np.float32)
        obs[7] = (fire == FireLevel.BURNT).astype(np.float32)
        obs[8] = p.confidence.matrix.astype(np.float32)
        obs[9] = self.agent.explored.astype(np.float32)
        return obs

    def _build_info(self, curr) -> dict:
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
    # Smoke check: random actions for 50 steps, prove the env runs end-to-end.
    env = SaREnv()
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
