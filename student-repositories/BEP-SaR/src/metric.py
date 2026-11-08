from dataclasses import dataclass

import numpy as np

from .agent import Agent
from .constants import RunOutcome

@dataclass(frozen=True)
class MetricInTime:
    """copy of the metric state at a given time step"""
    step: int
    outcome: RunOutcome
    victims_found: int
    total_victims: int
    victims_saved: int
    victims_dead: int
    time_to_first_found: int | None
    time_to_all_found: int | None
    time_to_first_saved: int | None
    total_traversable: int
    terrain_collisions: dict[Agent, int]
    victim_collisions: dict[Agent, int]
    inter_agent_collisions: dict[Agent, int]
    fire_collisions: dict[Agent, int]
    wait_actions: dict[Agent, int]
    damage: dict[Agent, int]
    area_explored: dict[Agent, float]
    total_explored: float
    inter_agent_distance: float
    total_spatial_conflicts: int
    individual_detection_times: list[int]


class Metric:
    def __init__(self):
        self.terrain_collisions: dict[Agent, int] = {} # Number of times agent hits a wall
        self.victim_collisions: dict[Agent, int] = {} # Number of times it hits a victim
        self.inter_agent_collisions: dict[Agent, int] = {} # Number of times it collides with another agent
        self.fire_collisions: dict[Agent, int] = {} # Number of steps it spends in burning tiles
        self.wait_actions: dict[Agent, int] = {}  # How many times the agent chooses to wait
        self.damage: dict[Agent, int] = {}  # (terrain collisions + victim collisions + inter-agent collisions) 
                                            # * damage per collision (for now all same)

        self.outcome: RunOutcome = RunOutcome.IN_PROGRESS
        self.steps_taken: int = 0
        self.victims_found: int = 0
        self.total_victims: int = 0
        self.victims_saved: int = 0
        self.victims_dead: int = 0
        self.time_to_first_found: int | None = None
        self.time_to_all_found: int | None = None
        self.time_to_first_saved: int | None = None

        # Fraction of traversable cells the agent has ever observed. [0, 1] per agent.
        self.area_explored: dict[Agent, float] = {}
        self.total_traversable: int = 0

        # Supervisory metrics
        self.total_explored: float = 0.0 # fraction of total traversable area explored by all agents combined, accounting for overlaps
        self.average_inter_agent_distance: float = 0.0
        self.total_spatial_conflicts: int = 0
        self.individual_detection_times: list[int] = [] # time steps at which each victim was found for the first time

        # Number of times each agent has reached a state defined as "infeasible" by its controller
        self.infeasible_states: dict[Agent, int] = {}

        # per time step history of all the metrics
        self.history: list[MetricInTime] = []

    def register_agent(self, agent: Agent) -> None:
        self.terrain_collisions[agent] = 0
        self.victim_collisions[agent] = 0
        self.inter_agent_collisions[agent] = 0
        self.fire_collisions[agent] = 0
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

    def record_fire_collision(self, agent: Agent) -> None:
        self.fire_collisions[agent] += 1
        self.damage[agent] += 1

    def record_vulnerable_collision(self, agent: Agent, vulnerable: float) -> None:
        rnd = np.random.random()
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
            victims_saved=self.victims_saved,
            victims_dead=self.victims_dead,
            time_to_first_found=self.time_to_first_found,
            time_to_all_found=self.time_to_all_found,
            time_to_first_saved=self.time_to_first_saved,
            total_traversable=self.total_traversable,
            total_explored=self.total_explored,
            inter_agent_distance=self.average_inter_agent_distance,
            total_spatial_conflicts=self.total_spatial_conflicts,
            terrain_collisions=dict(self.terrain_collisions),
            victim_collisions=dict(self.victim_collisions),
            inter_agent_collisions=dict(self.inter_agent_collisions),
            fire_collisions=dict(self.fire_collisions),
            wait_actions=dict(self.wait_actions),
            damage=dict(self.damage),
            area_explored=dict(self.area_explored),
            individual_detection_times=list(self.individual_detection_times),
        )

    def record_snapshot(self) -> None:
        """append the current metric state to the history log"""
        self.history.append(self.snapshot())
