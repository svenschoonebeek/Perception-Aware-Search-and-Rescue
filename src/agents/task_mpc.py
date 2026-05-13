from collections import deque
import logging
import numpy as np
from numpy.typing import NDArray
from typing import Self
from scipy.ndimage import binary_dilation

from src.agent import Agent, AgentAction
from src.constants import EventPriority, FireLevel, InterventionEvent, Task, TaskType, TraversabilityLevel
from src.supervisor import Supervisor

logger = logging.getLogger(__name__)

class TaskMpcAgent(Agent):
    def __init__(self, name: str, x: int, y: int, width: int, height: int, 
                 decay: float, scan_accuracy: float, scan_radius: int, scan_falloff: bool):
        super().__init__(name, x, y, width, height, decay, scan_accuracy, scan_radius, scan_falloff)
        
        # How many steps ahead to simulate.
        # The number of deep-copies of the agent made when calculating the action to take will be 5^horizon
        self.horizon = 3
        self.discount = 0.9 # Discount factor, nearer moves are more important than later moves
        self.fire_spread_rate = 0.3

        # Task info
        self.current_task = Task(TaskType.IDLE, -1, -1)
        self.current_path: list[tuple[int, int]] = []

        self.impatience_counter = 0
    
    def copy(self) -> Self:
        copy = super().copy()

        copy.horizon = self.horizon
        copy.discount = self.discount
        copy.fire_spread_rate = self.fire_spread_rate

        copy.current_path = list(self.current_path)
        copy.current_task = Task(self.current_task.task_type, self.current_task.x, self.current_task.y)
        copy.impatience_counter = self.impatience_counter

        return copy

    def get_current_task(self) -> Task:
        return self.current_task
    
    def set_current_task(self, task: Task) -> None:
        self.current_task = task

    def get_current_path(self) -> list[tuple[int, int]]:
        return self.current_path
    
    def set_current_path(self, path: list[tuple[int, int]]) -> None:
        self.current_path = path
    
    def get_action(self) -> AgentAction: 
        """
        Decides what action to perform using exhaustive search

        Returns:
        The action the agent wants to perform, decided using MPC
        """
        def dfs(model_state: TaskMpcAgent, depth: int, objective: float) -> tuple[float, list]:
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
    
    def check_task_status(self, supervisor: Supervisor | None) -> None:
        """
        Checks whether the current task is completed, and if so, switches to idle
        """
        if supervisor is None:
            return
        
        if self.current_task.task_type == TaskType.EXPLORE:
            distance_to_target = np.linalg.norm(np.array([self.current_task.y, self.current_task.x]) - np.array([self.y, self.x]))
            if distance_to_target < self.scan_radius / 2.0:
                self._return_to_idle(supervisor)

    def _is_path_blocked(self) -> bool:
        """
        Checks if the currently planned path contains any newly discovered walls or fires.
        """
        for x, y in self.current_path:
            # Did we just discover a wall here?
            if self.perception.traversability[y][x] == TraversabilityLevel.UNTRAVERSABLE:
                return True
            #? should count fire as blocking? I don't think so
            # if self.perception.fire[y][x] == FireLevel.BURNING:
            #     return True
                
        return False
    
    def _return_to_idle(self, supervisor: Supervisor) -> None:
        """
        Helper function to return to idle state after completing a task
        """
        supervisor.trigger_event(InterventionEvent(
            priority=EventPriority.TASK_COMPLETED,
            description=f"{self.name} completed exploration task at ({self.current_task.x}, {self.current_task.y})",
            data={"agent": self.name}
        ))
        self.current_task = Task(TaskType.IDLE, -1, -1)
        self.current_path = []
    
    def _objective(self) -> float:
        if (self.current_task.task_type == TaskType.EXPLORE):
            return self._objective_explore()
        
        return self._objective_wander()

    def _objective_wander(self) -> float:
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
    
    def _objective_explore(self) -> float:
        """
        Goes to a certain objective tile (e.g. a frontier) and then switches to wander mode
        """
        w_safety, w_confidence, w_path = 20, 1, 500 # To be adjusted

        safety      = -w_safety * self._safety_penalty()
        confidence  =  w_confidence * self._confidence_score()

        carrot_x, carrot_y = self.get_carrot_waypoint()
        carrot_distance = np.linalg.norm(np.array([carrot_y, carrot_x]) - np.array([self.y, self.x]))
        path_score = w_path * float(1.0 / (1.0 + carrot_distance))
        # TODO maybe here we also do exploration? such that it explores on the way
        # print("path_score:", path_score, "Safety:", safety, "Confidence:", confidence)

        return safety + confidence + path_score

    def get_carrot_waypoint(self) -> tuple[int, int]:        
        if len(self.current_path) == 0:
            return (self.x, self.y)
        
        current_idx = 0
        min_dist = float('inf')
        for i, (px, py) in enumerate(self.current_path):
            dist = np.linalg.norm(np.array([py, px]) - np.array([self.y, self.x]))
            if dist < min_dist:
                min_dist = dist
                current_idx = i
        
        # look 2 steps ahead on the path, this will be the carrot
        lookahead_idx = min(current_idx + 2, len(self.current_path) - 1)
        return self.current_path[lookahead_idx]

    def _predict_next_state(self, action: AgentAction) -> "TaskMpcAgent": # <- Apparently how to do forward declaration
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
        traversable_mask = self.perception.traversability == TraversabilityLevel.TRAVERSABLE
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
            traversable_mask = self.perception.traversability == TraversabilityLevel.TRAVERSABLE
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
        
        if self.perception.agents[target_cell_y][target_cell_x] == 1:
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
            return float(explored) / (self.world_height * self.world_width)
        
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