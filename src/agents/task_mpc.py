from typing import Self
import numpy as np

from .mpc import MpcAgent
from src.constants import *
from src.supervisor import Supervisor

class TaskMpcAgent(MpcAgent):
    def __init__(self, name: str, x: int, y: int, width: int, height: int, 
                 decay: float, scan_accuracy: float, scan_radius: int, scan_falloff: bool):
        
        super().__init__(name, x, y, width, height, decay, scan_accuracy, scan_radius, scan_falloff)
        
        # Task info
        self.current_task = Task(TaskType.IDLE, -1, -1)
        self.current_path: list[tuple[int, int]] = []

        self.impatience_counter = 0
    
    def copy(self) -> Self:
        copy = super().copy()

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
        w_exploration, w_safety, w_confidence, w_path = 100, 20, 1, 200 # To be adjusted

        safety      = -w_safety * self._safety_penalty()
        confidence  =  w_confidence * self._confidence_score()
        # TODO maybe here we also do exploration? such that it explores on the way
        # exploration =  w_exploration * self._exploration_score()

        carrot_x, carrot_y = self.get_carrot_waypoint()
        carrot_distance = np.linalg.norm(np.array([carrot_y, carrot_x]) - np.array([self.y, self.x]))
        path_score = w_path * float(1.0 / (1.0 + carrot_distance))

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