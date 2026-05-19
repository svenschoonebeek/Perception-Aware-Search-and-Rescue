from typing import Self
import numpy as np
import logging

from .mpc import MpcAgent
from src.state import State
from src.constants import *
from src.supervisor import Supervisor

logger = logging.getLogger(__name__)

class TaskMpcAgent(MpcAgent):
    def __init__(self, name: str, x: int, y: int, width: int, height: int, 
                 decay: float, scan_accuracy: float, scan_radius: int, scan_falloff: bool):
        
        super().__init__(name, x, y, width, height, decay, scan_accuracy, scan_radius, scan_falloff)
        
        # Task info
        self.current_task = Task(TaskType.IDLE, -1, -1)
        self.current_path: list[tuple[int, int]] = []
        
        self.conflict_timeout = 0
    
    def copy(self) -> Self:
        copy = super().copy()

        copy.current_path = list(self.current_path)
        copy.current_task = Task(self.current_task.task_type, self.current_task.x, self.current_task.y)
        copy.conflict_timeout = self.conflict_timeout

        return copy

    def get_current_task(self) -> Task:
        return self.current_task
    
    def set_current_task(self, task: Task) -> None:
        self.current_task = task

    def get_current_path(self) -> list[tuple[int, int]]:
        return self.current_path
    
    def set_current_path(self, path: list[tuple[int, int]]) -> None:
        self.current_path = path
    
    def check_status(self, supervisor: Supervisor | None, state: State) -> None:
        """
        Checks whether the status of the agent, like if a target has been met, and triggers the corresponding event for the supervisor.

        Parameters:
        supervisor: the supervisor to trigger
        state: the true world (used for checking if close to another agent)
        """
        if supervisor is None:
            return
        
        if self.current_task.task_type == TaskType.EXPLORE:
            distance_to_target = np.linalg.norm(np.array([self.current_task.y, self.current_task.x]) - np.array([self.y, self.x]))
            if distance_to_target < self.scan_radius / 2.0:
                self._return_to_idle(supervisor)
        
        self.conflict_timeout += 1
        if (self.conflict_timeout >= 3):
            self._check_for_spatial_conflict(supervisor, state)

    def _check_for_spatial_conflict(self, supervisor: Supervisor, state: State):
        conflict_distance = self.scan_radius # could change this to be smaller

        visible = self._tiles_in_radius() & self._tiles_in_los(state)
        conflict_mask = visible & (self.perception.agents.matrix > 0)
        agent_ys, agent_xs = np.nonzero(conflict_mask)

        for y, x in zip(agent_ys, agent_xs):
            if x == self.x and y == self.y:
                continue
            
            distance = np.sqrt((self.x - x)**2 + (self.y - y)**2)
            if distance < conflict_distance:
                logger.debug(f"{self.name} detected a spatial conflict with another agent.")
                supervisor.trigger_event(InterventionEvent(
                    priority=EventPriority.SPATIAL_CONFLICT,
                    description=f"{self.name} detected a spatial conflict locally.",
                    data={"agent": self.name}
                    ))
                self.conflict_timeout = 0
                return
    
    def _is_path_blocked(self) -> bool:
        """
        Checks if the currently planned path contains any newly discovered walls or fires.
        """
        for x, y in self.current_path:
            if self.perception.traversability[y][x] == TraversabilityLevel.UNTRAVERSABLE:
                return True
            #? should count fire as blocking?
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
    
    def _objective_terminal(self) -> float:
        return 0
    
    def _objective_stage(self) -> float:
        if (self.current_task.task_type == TaskType.EXPLORE):
            return self._objective_explore()
        
        return self._objective_wander()

    def _objective_wander(self) -> float:
        """
        Calculates the agents perceived objective value

        Returns:
        The objective value
        """
        w_discovery, w_safety, w_confidence = 100, 10, 3 # To be adjusted

        discovery   =  w_discovery * self._discovery_score()
        safety      = -w_safety * self._safety_penalty()
        confidence  =  w_confidence * self._confidence_score()

        # print("Discovery:", discovery, "Safety:", safety, "Confidence:", confidence)

        return discovery + safety + confidence
    
    def _objective_explore(self) -> float:
        """
        Goes to a certain objective tile (e.g. a frontier) and then switches to wander mode
        """
        w_discovery, w_safety, w_confidence, w_path = 100, 20, 1, 100 # To be adjusted

        safety      = -w_safety * self._safety_penalty()
        confidence  =  w_confidence * self._confidence_score()
        # TODO maybe here we also do discovery? such that it explores on the way
        # discovery =  w_discovery * self._discovery_score()

        # carrot_x, carrot_y = self.get_carrot_waypoint()
        # carrot_distance = np.linalg.norm(np.array([carrot_y, carrot_x]) - np.array([self.y, self.x]))
        # path_score = w_path * float(1.0 / (1.0 + carrot_distance))

        path_score = w_path * self._calculate_path_score()

        # print("path_score:", path_score, "Safety:", safety, "Confidence:", confidence)

        return safety + confidence + path_score # + exploration
    
    def _calculate_path_score(self) -> float:
        """
        Calculates a raw, normalized score for how well the agent is adhering to its path.
        """
        if len(self.current_path) == 0:
            return 0.0
        
        min_dist = float('inf')
        closest_idx = 0
        
        for i, (px, py) in enumerate(self.current_path):
            dist = np.sqrt((px - self.x)**2 + (py - self.y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        if min_dist == 0: # exactly on the path
            return 4.0 + (closest_idx * 0.1)
        else: # off the path
            return 3.0 - (min_dist * 0.2)


    def _get_carrot_waypoint(self) -> tuple[int, int]:        
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
        res = self.current_path[lookahead_idx]
        return res