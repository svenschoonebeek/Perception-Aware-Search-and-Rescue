import numpy as np

from .agent import Agent
from .state import State

class Supervisor:
    def __init__(self, width: int, height: int, agents: list[Agent], sigma: float):
        self.agents = agents
        self.shared_belief = State(width, height)
        self.active_tasks = []

        self.sigma = sigma # Certainty decay per time step [0,1]


    def update_belief(self, agent_scans: list[list[dict]]) -> None:
        """
        Updates the shared belief state based on the agent states
        """
        self.shared_belief.confidence.matrix = np.maximum(self.shared_belief.confidence.matrix - self.sigma, 0) # vectorized confidence decay

        for agent_scan in agent_scans:
            for tile in agent_scan:
                is_more_confident = tile["confidence"] > self.shared_belief.confidence[tile["y"]][tile["x"]]
                if is_more_confident:
                    self.shared_belief.confidence[tile["y"]][tile["x"]] = tile["confidence"]
                    self.shared_belief.traversability[tile["y"]][tile["x"]] = tile["traversability"]
                    self.shared_belief.victims[tile["y"]][tile["x"]] = tile["victims"]
                    self.shared_belief.agents[tile["y"]][tile["x"]] = tile["agents"]
        
    
    def assign_tasks(self) -> None:
        """
        Assigns tasks to agents based on the shared belief and active tasks
        """
        for agent in self.agents:
            # 
            pass