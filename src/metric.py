from dataclasses import dataclass

import random

from .agent import Agent
from .constants import RunOutcome

@dataclass(frozen=True)
class MetricInTime:
    """copy of the metric state at a given time step"""
    step: int
    outcome: RunOutcome
    victims_found: int
    total_victims: int
    time_to_first_found: int | None
    time_to_all_found: int | None
    total_traversable: int
    terrain_collisions: dict[Agent, int]
    victim_collisions: dict[Agent, int]
    inter_agent_collisions: dict[Agent, int]
    wait_actions: dict[Agent, int]
    damage: dict[Agent, int]
    area_explored: dict[Agent, float]


class Metric:
    def __init__(self):
        self.terrain_collisions: dict[Agent, int] = {} # Number of times agent hits a wall
        self.victim_collisions: dict[Agent, int] = {} # Number of times it hits a victim
        self.inter_agent_collisions: dict[Agent, int] = {} # Number of times it collides with another agent
        self.wait_actions: dict[Agent, int] = {}  # How many times the agent chooses to wait
        self.damage: dict[Agent, int] = {}  # (terrain collisions + victim collisions + inter-agent collisions) 
                                            # * damage per collision (for now all same)

        self.outcome: RunOutcome = RunOutcome.IN_PROGRESS
        self.steps_taken: int = 0
        self.victims_found: int = 0
        self.total_victims: int = 0
        self.time_to_first_found: int | None = None
        self.time_to_all_found: int | None = None

        # Fraction of traversable cells the agent has ever observed. [0, 1] per agent.
        self.area_explored: dict[Agent, float] = {}
        self.total_traversable: int = 0

        # per time step history of all the metrics
        self.history: list[MetricInTime] = []

    def register_agent(self, agent: Agent) -> None:
        self.terrain_collisions[agent] = 0
        self.victim_collisions[agent] = 0
        self.inter_agent_collisions[agent] = 0
        self.wait_actions[agent] = 0
        self.damage[agent] = 0
        self.area_explored[agent] = 0.0

    def record_terrain_collision(self, agent: Agent) -> None:
        self.terrain_collisions[agent] += 1
        self.damage[agent] += 1

    def record_victim_collision(self, agent: Agent) -> None:
        self.victim_collisions[agent] += 1
        self.damage[agent] += 1

    def record_inter_agent_collision(self, agent: Agent) -> None:
        self.inter_agent_collisions[agent] += 1
        self.damage[agent] += 1

    def record_vulnerable_collision(self, agent: Agent, vulnerable: float) -> None:
        rnd = random.random()
        if (rnd < vulnerable):
            self.damage[agent] += 1

    def record_wait(self, agent: Agent) -> None:
        self.wait_actions[agent] += 1

    def snapshot(self) -> MetricInTime:
        """return the current state without modifying it"""
        return MetricInTime(
            step=self.steps_taken,
            outcome=self.outcome,
            victims_found=self.victims_found,
            total_victims=self.total_victims,
            time_to_first_found=self.time_to_first_found,
            time_to_all_found=self.time_to_all_found,
            total_traversable=self.total_traversable,
            terrain_collisions=dict(self.terrain_collisions),
            victim_collisions=dict(self.victim_collisions),
            inter_agent_collisions=dict(self.inter_agent_collisions),
            wait_actions=dict(self.wait_actions),
            damage=dict(self.damage),
            area_explored=dict(self.area_explored),
        )

    def record_snapshot(self) -> None:
        """append the current metric state to the history log"""
        self.history.append(self.snapshot())
