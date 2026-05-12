import numpy as np
from numpy.typing import NDArray

from typing import Self

from scipy.ndimage import binary_dilation

from src.agent import Agent, AgentAction
from src.constants import FireLevel, TraversabilityLevel

class MpcAgent(Agent):
    def __init__(self, name: str, x: int, y: int, width: int, height: int, 
                 decay: float, scan_accuracy: float, scan_radius: int, scan_falloff: bool):
        super().__init__(name, x, y, width, height, decay, scan_accuracy, scan_radius, scan_falloff)
        
        # How many steps ahead to simulate.
        # The number of deep-copies of the agent made when calculating the action to take will be 5^horizon
        self.horizon = 3
        self.discount = 0.9 # Discount factor, nearer moves are more important than later moves
        self.fire_spread_rate = 0.3
    
    def copy(self) -> Self:
        copy = super().copy()

        copy.horizon = self.horizon
        copy.discount = self.discount
        copy.fire_spread_rate = self.fire_spread_rate

        return copy

    def get_action(self) -> AgentAction:
        """
        Decides what action to perform using exhaustive search

        Returns:
        The action the agent wants to perform, decided using MPC
        """
        def dfs(model_state: MpcAgent, depth: int, objective: float) -> tuple[float, list]:
            if depth > self.horizon:
                return objective, []

            best_objective = float('-inf')
            best_sequence = []

            for action in AgentAction:
                if not model_state._is_feasible(action):
                    continue

                next_state = model_state._predict_next_state(action)
                next_objective = objective + (self.discount ** depth * next_state._objective())
                
                future_obj, future_seq = dfs(next_state, depth + 1, next_objective)
                
                if future_obj > best_objective:
                    best_objective = future_obj
                    best_sequence = [action] + future_seq

            return best_objective, best_sequence

        best_objective, best_sequence = dfs(self.copy(), 1, 0)

        return best_sequence[0]

    def _objective(self) -> float:
        """
        Calculates the agents perceived objective value

        Returns:
        The objective value
        """
        w_exploration, w_safety, w_confidence = 100, 20, 1 # To be adjusted

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
        new_agent = self.copy()

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

    def _predict_fire_spread(self) -> NDArray:
        """
        Predicts the spread of fire.
        Does so probabilistically (where random.random() should be seeded), and with an educated guess on spread rate

        Returns:
        The predicted fire matrix
        """
        # Probabilistically (Randomly (seeded) ignite, educated guess on spread rate)
        predicted_fire = np.copy(self.perception.fire.matrix)
        
        random_noise = np.random.rand(self.world_height, self.world_width)

        ignited_mask = (predicted_fire == FireLevel.FLAMMABLE) & (random_noise < self.fire_spread_rate)
        predicted_fire[ignited_mask] = FireLevel.BURNING

        adjacent_mask = binary_dilation(ignited_mask)
        safe_mask = predicted_fire == FireLevel.SAFE
        traversable_mask = self.perception.traversability == TraversabilityLevel.TRAVERSIBLE
        exposed_mask = adjacent_mask & safe_mask & traversable_mask
        predicted_fire[exposed_mask] = FireLevel.FLAMMABLE

        return predicted_fire
    
    def _predict_fire_spread_horizon(self) -> NDArray:
        """
        Predicts the spread of fire over every step until the horizon, since fire spread is atm the same for every branch
        Does so probabilistically (where random.random() should be seeded), and with an educated guess on spread rate

        Returns:
        A matrix containing <horizon> sequential fire matrices
        """
        # Probabilistically (Randomly (seeded) ignite, educated guess on spread rate)
        predictions = np.empty((self.horizon, self.world_height, self.world_width))
        current_fire = np.copy(self.perception.fire.matrix)

        for i in range(self.horizon):
            random_noise = np.random.rand(self.world_height, self.world_width)

            ignited_mask = (current_fire == FireLevel.FLAMMABLE) & (random_noise < self.fire_spread_rate)
            current_fire[ignited_mask] = FireLevel.BURNING

            adjacent_mask = binary_dilation(ignited_mask)
            safe_mask = current_fire == FireLevel.SAFE
            traversable_mask = self.perception.traversability == TraversabilityLevel.TRAVERSIBLE
            exposed_mask = adjacent_mask & safe_mask & traversable_mask
            current_fire[exposed_mask] = FireLevel.FLAMMABLE

            predictions[i] = current_fire

        return predictions

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
        burning_penalty = int(self.perception.fire[self.y][self.x] == FireLevel.BURNING) 
        flammable_penalty = int(self.perception.fire[self.y][self.x] == FireLevel.FLAMMABLE)

        w_vuln = 1.0
        w_burn = 50.0
        w_flam = 0.2

        return (vulnerability_penalty * w_vuln + burning_penalty * w_burn + flammable_penalty * w_flam) / (w_vuln + w_burn + w_flam)
        
    def _confidence_score(self) -> float:
        """
        Calculates a score in regards to how confident the agent is

        Returns:
        The score, normalized to [0, 1]
        """
        return np.sum(self.perception.confidence.matrix) / (self.world_height * self.world_width * self.scan_accuracy)