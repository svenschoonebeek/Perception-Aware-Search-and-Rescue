import itertools
import numpy as np
import random
from copy import deepcopy

from src.agent import Agent, AgentAction
from src.state import State
from src.constants import FireLevel, TraversabilityLevel

class MpcAgent(Agent):
    def __init__(self, name: str, x: int, y: int, width: int, height: int, 
                 decay: float, scan_accuracy: float, scan_radius: int, scan_falloff: bool):
        super().__init__(name, x, y, width, height, decay, scan_accuracy, scan_radius, scan_falloff)
        
        # How many steps ahead to simulate.
        # The number of deep-copies of the agent made when calculating the action to take will be 5^horizon
        self.horizon = 3 
        self.gamma = 0.9 # Discount factor, nearer moves are more important than later moves

    def get_action(self) -> AgentAction:
        """
        Decides what action to perform using exhaustive search

        Returns:
        The action the agent wants to perform, decided using MPC
        """
        best_sequence = [AgentAction.WAIT] * self.horizon
        best_objective = float('-inf')

        for sequence in itertools.product(AgentAction, repeat=self.horizon):
            model_state = deepcopy(self)

            total_objective = 0.0
            feasible = True

            for step, action in enumerate(sequence):
                if not model_state._is_feasible(action):
                    feasible = False
                    break

                model_state = model_state._predict_next_state(action)
                total_objective += (self.gamma ** step) * model_state._objective()
            
            if feasible and total_objective > best_objective:
                best_sequence = sequence
                best_objective = total_objective

        return best_sequence[0]

    def _objective(self) -> float:
        """
        Calculates the agents perceived objective value

        Returns:
        The objective value
        """
        w_exploration, w_safety, w_confidence = 400, 1, 100 # To be adjusted

        exploration =  w_exploration * self._exploration_score()
        safety      = -w_safety * self._safety_penalty()
        confidence  =  w_confidence * self._confidence_score()

        # print("Exploration:", exploration, "Safety:", safety, "Confidence:", confidence)

        return exploration + safety + confidence

    def _predict_next_state(self, action: AgentAction) -> "MpcAgent": # <- Apparently how to do forward declaration
        """
        Creates a prediction of the state after performing the given action.
        This assumes the action is feasible (see self._is_feasible)

        Parameters:
        action: The action to perform

        Returns:
        The predicted next state
        """
        new_agent = deepcopy(self)

        match action:
            case AgentAction.MOVE_UP:
                new_agent.perception.agents[self.y - 1][self.x] = 1
                new_agent.perception.agents[self.y][self.x] = 0
                new_agent.y -= 1
            case AgentAction.MOVE_DOWN:
                new_agent.perception.agents[self.y + 1][self.x] = 1
                new_agent.perception.agents[self.y][self.x] = 0
                new_agent.y += 1
            case AgentAction.MOVE_LEFT:
                new_agent.perception.agents[self.y][self.x - 1] = 1
                new_agent.perception.agents[self.y][self.x] = 0
                new_agent.x -= 1
            case AgentAction.MOVE_RIGHT:
                new_agent.perception.agents[self.y][self.x + 1] = 1
                new_agent.perception.agents[self.y][self.x] = 0
                new_agent.x += 1

        new_agent.perception.fire.matrix = self._predict_fire_spread()        
        new_agent.scan(new_agent.perception)

        return new_agent

    def _predict_fire_spread(self) -> np.ndarray:
        """
        Predicts the spread of fire.
        Does so probabilistically (where random.random() should be seeded), and with an educated guess on spread rate

        Returns:
        A predicted fire matrix
        """
        # Probabilistically (Randomly (seeded) ignite, educated guess on spread rate)
        predicted_spread_rate = 0.3
        predicted_fire = deepcopy(self.perception.fire.matrix)

        for y in range(self.world_height):
            for x in range(self.world_width):
                if self.perception.fire[y][x] == FireLevel.FLAMMABLE and random.random() < predicted_spread_rate:
                    predicted_fire[y][x] = FireLevel.BURNING
                    
                    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        
                        if not (0 <= nx < self.world_width and 0 <= ny < self.world_height):
                            continue
                        if self.perception.traversability[ny, nx] == TraversabilityLevel.UNTRAVERSIBLE:
                            continue

                        if self.perception.fire[ny, nx] == FireLevel.SAFE:
                            predicted_fire[ny, nx] = FireLevel.FLAMMABLE
        
        return predicted_fire

    def _is_feasible(self, action: AgentAction) -> bool:
        """
        Decides whether a given action will result in a feasible position
        (e.g. not walking into a wall)

        Parameters:
        action: The action to check

        Returns:
        Whether the action is feasible
        """
        target_cell_x, target_cell_y = self._target_cell(action)
        if (target_cell_x < 0 or target_cell_x >= self.world_width or 
            target_cell_y < 0 or target_cell_y >= self.world_height): # Out of bounds
            return False

        if self.perception.victims[target_cell_y][target_cell_x] == 1: # Hit victim
            return False
        
        return self.perception.traversability[target_cell_y][target_cell_x] == 0 # Didn't hit wall


    def _target_cell(self, action: AgentAction) -> tuple[int, int]:
        """
        Calculates the cell targeted by the agent given its action.

        Parameters:
        action: The action the agent wants to perform

        Returns:
        The target cell indices
        """
        match action:
            case AgentAction.MOVE_UP:
                return self.x, self.y - 1
            case AgentAction.MOVE_DOWN:
                return self.x, self.y + 1
            case AgentAction.MOVE_LEFT:
                return self.x - 1, self.y
            case AgentAction.MOVE_RIGHT:
                return self.x + 1, self.y
        
        return self.x, self.y # Wait action
    
    def _exploration_score(self) -> float:
        """
        Calculates a score in regards to how much area is explored.

        Returns:
        The score, normalized to [0, 1]
        """
        explored = np.count_nonzero(self.explored)
        unexplored = np.argwhere(self.explored == False)
    
        if len(unexplored) == 0:
            return explored / (self.world_height * self.world_width)
        
        # Heading towards unexplored tiles is more important
        distances = np.linalg.norm(unexplored - np.array([self.y, self.x]), axis=1)
        min_distance = np.min(distances)
        proximity_bonus = 1.0 / (1.0 + min_distance)

        return (explored + proximity_bonus) / (self.world_height * self.world_width + 1)

    def _safety_penalty(self) -> float:
        """
        Calculates a score in regards to how unsafe the agent is

        Returns:
        The score, normalized to [0, 1]
        """
        # Penalty for being on vulnerable tile
        vulnerability_penalty = self.perception.vulnerability[self.y][self.x]
        fire_penalty = (
            2 if self.perception.fire[self.y][self.x] == FireLevel.BURNING else 
            1 if self.perception.fire[self.y][self.x] == FireLevel.FLAMMABLE else 
            0
        )

        return (vulnerability_penalty + fire_penalty) / 3
        
    def _confidence_score(self) -> float:
        """
        Calculates a score in regards to how confident the agent is

        Returns:
        The score, normalized to [0, 1]
        """
        return np.sum(self.perception.confidence.matrix) / (self.world_height * self.world_width * self.scan_accuracy)