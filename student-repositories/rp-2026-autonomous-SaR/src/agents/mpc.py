import numpy as np
from numpy.typing import NDArray
import logging
from typing import Self
from scipy.ndimage import binary_dilation

from src.agent import Agent
from src.constants import AgentAction, AgentPresence, FireLevel, TraversabilityLevel, VictimPresence

logger = logging.getLogger(__name__)

class MpcAgent(Agent):
    def __init__(self, name: str, x: int, y: int, width: int, height: int, 
                 decay: float, scan_accuracy: float, scan_radius: int, scan_falloff: bool):
        super().__init__(name, x, y, width, height, decay, scan_accuracy, scan_radius, scan_falloff)
        
        # How many steps ahead to simulate.
        # The number of deep-copies of the agent made when calculating the action to take will be 5^horizon
        self.horizon = 4
        self.discount = 0.9 # Discount factor, nearer moves are more important than later moves
        self.fire_spread_rate = 0.3

        self.frontier_distances: NDArray = np.full((self.world_height, self.world_width), -1, dtype=np.int16)
        self.fire_prediction: NDArray = np.zeros((self.world_height, self.world_width), dtype=FireLevel)
    
    def copy(self) -> Self:
        copy = super().copy()

        copy.horizon = self.horizon
        copy.discount = self.discount
        copy.fire_spread_rate = self.fire_spread_rate

        copy.frontier_distances = np.copy(self.frontier_distances)
        copy.fire_prediction = np.copy(self.fire_prediction)

        return copy

    def get_action(self) -> AgentAction:
        """
        Decides what action to perform using exhaustive search

        Returns:
        The action the agent wants to perform, decided using MPC
        """
        self.frontier_distances = self._closest_unexplored()
        self.fire_prediction = self._predict_fire_spread_horizon()
        
        def dfs(model_state: MpcAgent, depth: int, objective: float) -> tuple[float, list]:
            if depth >= self.horizon:
                return objective + model_state._objective_terminal(), []

            best_objective = float('-inf')
            best_sequence = []

            for action in AgentAction:
                next_state = model_state._predict_next_state(action, depth)
                if next_state is None or not next_state._is_feasible():
                    continue
                next_state.scan(next_state.perception)

                next_objective = objective + ((self.discount ** depth) * next_state._objective_stage())
                
                future_obj, future_seq = dfs(next_state, depth + 1, next_objective)

                if future_obj > best_objective:
                    best_objective = future_obj
                    best_sequence = [action] + future_seq

            return best_objective, best_sequence

        best_objective, best_sequence = dfs(self.copy(), 0, 0)

        if len(best_sequence) == 0:
            self.infeasible_states += 1
            logger.debug("INFEASIBLE STATE REACHED ==========================")
            logger.debug("Dscv:", self._discovery_score())
            logger.debug("Safety:", self._safety_penalty())
            logger.debug("Conf:", self._confidence_score())
            logger.debug("Expl:", self._exploration_penalty())
            return AgentAction.WAIT

        return best_sequence[0]
    
    def _predict_next_state(self, action: AgentAction, depth: int) -> Self | None:
        """
        Predicts the state resulting from the given action, unless said prediction would fall outside the admissible set X
        Does not check whether the predicted state violates constraints (see self._is_feasible)

        Parameters:
        action: The action to perform

        Returns:
        The predicted next state, or None if the action would result in a state not in X (go out of bounds)
        """
        target_x, target_y = self._target_cell(action)
        if not(0 <= target_x < self.world_width and 0 <= target_y < self.world_height):
            return None

        new_agent = self.copy()

        current = self.y, self.x
        target = target_y, target_x
        new_agent.perception.agents[current] = 0
        new_agent.perception.agents[target] = 1
        new_agent.y, new_agent.x = target

        new_agent.perception.fire.matrix = self.fire_prediction[depth]        

        return new_agent

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

            adjacent_mask = binary_dilation(current_fire == FireLevel.BURNING)
            ignited_mask = adjacent_mask & (current_fire == FireLevel.FLAMMABLE) & (random_noise < self.fire_spread_rate)
            current_fire[ignited_mask] = FireLevel.BURNING

            adjacent_mask = binary_dilation(ignited_mask)
            safe_mask = current_fire == FireLevel.SAFE
            traversable_mask = self.perception.traversability == TraversabilityLevel.TRAVERSABLE
            exposed_mask = adjacent_mask & safe_mask & traversable_mask
            current_fire[exposed_mask] = FireLevel.FLAMMABLE

            burning_mask = current_fire == FireLevel.BURNING
            burnout_mask = burning_mask & (self.fire_durations >= self.fire_duration)
            current_fire[burnout_mask] = FireLevel.BURNT

            predictions[i] = current_fire

        return predictions

    def _is_feasible(self) -> bool:
        """
        Checks whether a given state violates constraints (e.g. not walking into a wall)
        Assumes the given state is in the admissible set X

        Returns:
        True if no constraints were violated, else False
        """
        if self.perception.victims[self.y, self.x] == 1: # Hit victim
            return False

        # if self.perception.agents[self.y, self.x] == 1: # Hit agent
        #     return False
        
        if self.perception.fire.matrix[self.y, self.x] == FireLevel.BURNING: # On fire
            return False
        
        return self.perception.traversability[self.y, self.x] == 0 # Didn't hit anything
    
    def _objective_stage(self) -> float:
        """
        Calculates the agent's predicted stage reward - calculated for every step

        Returns:
        The objective value
        """
        w_discovery, w_safety, w_confidence = 100, 0, 3 # To be adjusted

        discovery   =  w_discovery * self._discovery_score()
        safety      = -w_safety * self._safety_penalty()
        confidence  =  w_confidence * self._confidence_score()

        logger.debug("Exploration: %f, Safety: %f, Confidence: %f", discovery, safety, confidence)

        return discovery + safety + confidence
    
    def _objective_terminal(self) -> float:
        """
        Calculates the agent's predicted terminal reward - calculated only for the last step

        Returns:
        The objective value
        """
        w_exploration = 300

        exploration = -w_exploration * self._exploration_penalty()

        return exploration * self.horizon
    
    def _discovery_score(self) -> float:
        """
        Calculates a score in regards to how much area is discovered this turn.

        Returns:
        The score, normalized to [0, 1]
        """
        newly_explored = np.count_nonzero(self.discovered)

        return float(newly_explored / ((np.pi * self.scan_radius ** 2) / 2))

    def _safety_penalty(self) -> float:
        """
        Calculates a penalty in regards to how unsafe the agent is

        Returns:
        The score, normalized to [0, 1]
        """
        return self.perception.vulnerability[self.y][self.x]
        
    def _confidence_score(self) -> float:
        """
        Calculates a score in regards to how confident the agent is

        Returns:
        The score, normalized to [0, 1]
        """
        return np.sum(self.perception.confidence.matrix) / (self.world_height * self.world_width)
    
    def _exploration_penalty(self) -> float:
        """
        Calculates a penalty in regards to how far from an unexplored tile the agent is 

        Returns:
        The score, normalised to [0, 1+] (Will go over one if the agent has to go through a maze to get there)
        """
        return self.frontier_distances[self.y, self.x] / (self.world_height + self.world_width)
    
    def _closest_unexplored(self) -> NDArray:
        """
        Uses multi-source BFS to calculate the shortest distance from each explored tile to closest unexplored tile
        should be JIT-compiled with numba for performance - currently infeasible

        Returns:
        A world-sized ndarray where each tile is:
            0 if the tile is unexplored or a wall
            x if the tile is explored, where x is the true shortest distance to the closest unexplored tile
        """
        big_M = 32767
        distance = np.full((self.world_height, self.world_width), big_M, dtype=np.int16)

        # pre-allocate queue so numba doesn't do any dynamic memory allocation
        # queue x and y together for cache locality
        max_queue_size = self.world_height * self.world_width
        queue = np.zeros((max_queue_size, 2), dtype=np.int16)

        head = 0
        tail = 0

        for y in range(self.world_height):
            for x in range(self.world_width):
                if not self.explored[y, x]:
                    distance[y, x] = 0
                    queue[tail] = (y, x)
                    tail += 1
        tail -= 1

        # up, down, left, right
        dy = np.array([-1, 1, 0, 0])
        dx = np.array([0, 0, -1, 1])

        moving_state = self.copy()
        while head < tail:
            (y, x) = queue[head]
            head += 1

            for i in range(4):
                ny: int = y + dy[i]
                nx: int = x + dx[i]

                if 0 <= ny < self.world_height and 0 <= nx < self.world_width:
                    moving_state.y, moving_state.x = ny, nx
                    if distance[ny, nx] == big_M and moving_state._is_feasible():
                        distance[ny, nx] = distance[y, x] + 1
                        queue[tail] = (ny, nx)
                        tail += 1

        return distance