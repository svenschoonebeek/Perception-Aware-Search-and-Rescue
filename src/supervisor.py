import numpy as np

from .agent import Agent
from .state import State
from .constants import *

class Supervisor:
    def __init__(self, width: int, height: int, sigma: float):
        self.shared_belief = State(width, height)
        self.active_tasks = []

        self.sigma = sigma # Certainty decay per time step [0,1]
        self.needs_reallocation = False # intervention flag


    def update_belief(self, agent_scans: list[list[Tile]]) -> None:
        """
        Updates the shared belief state based on the agent scans received
        """
        self.shared_belief.confidence.matrix = np.maximum(self.shared_belief.confidence.matrix - self.sigma, 0) # vectorized confidence decay

        for agent_scan in agent_scans:
            for tile in agent_scan:
                y = tile.y
                x = tile.x

                is_more_confident = tile.confidence > self.shared_belief.confidence[y][x]
                if is_more_confident:
                    self.shared_belief.confidence[y][x] = tile.confidence
                    self.shared_belief.traversability[y][x] = tile.traversability
                    self.shared_belief.vulnerability[y][x] = tile.vulnerability
                    self.shared_belief.fire[y][x] = tile.fire
                    self.shared_belief.victims[y][x] = tile.victims
                    self.shared_belief.agents[y][x] = tile.agents
        
    def check_for_conflicts(self) -> None:
        """
        Checks for conflicts between active tasks and locations of agents
        """
        pass

    def assign_tasks(self) -> None:
        """
        Assigns tasks to agents based on the shared belief and active tasks
        """
        pass