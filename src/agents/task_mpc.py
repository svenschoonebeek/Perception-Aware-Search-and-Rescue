from typing import Self
from numba import njit
import numpy as np
import logging

from .mpc import MpcAgent
from src.state import State
from src.constants import *
from src.supervisor import Supervisor
from src.util import fast_bfs_array

logger = logging.getLogger(__name__)

class TaskMpcAgent(MpcAgent):
    def __init__(self, name: str, x: int, y: int, width: int, height: int, 
                 decay: float, scan_accuracy: float, scan_radius: int, scan_falloff: bool, capabilities: Capabilities):
        
        super().__init__(name, x, y, width, height, decay, scan_accuracy, scan_radius, scan_falloff)
        
        # Task info
        self.current_task = Task(TaskType.IDLE, -1, -1)
        self.current_path: list[tuple[int, int]] = []
        
        self.capabilities = capabilities # how good the agent is at exploring vs rescuing
        self.conflict_timeout = 0
        self.spatial_conflicts = 0

        self.assignable = True # can be assigned tasks by the supervisor
    
    def copy(self) -> Self:
        copy = super().copy()

        copy.current_path = list(self.current_path)
        copy.current_task = Task(self.current_task.task_type, self.current_task.x, self.current_task.y)
        copy.capabilities = Capabilities(self.capabilities.exploration, self.capabilities.rescuing)
        copy.conflict_timeout = self.conflict_timeout
        copy.spatial_conflicts = self.spatial_conflicts
        copy.assignable = self.assignable

        return copy

    def get_current_task(self) -> Task:
        return self.current_task
    
    def set_current_task(self, task: Task) -> None:
        self.current_task = task

    def get_current_path(self) -> list[tuple[int, int]]:
        return self.current_path
    
    def set_current_path(self, path: list[tuple[int, int]]) -> None:
        self.current_path = path
    
    def get_capabilities(self) -> Capabilities:
        """
        Returns the capabilities of the agent, to be used by the supervisor for task allocation
        """
        return self.capabilities
    
    def get_spatial_conflicts(self) -> int:
        """
        Returns the number of spatial conflicts the agent has been involved in, to be used by the supervisor for task allocation
        """
        return self.spatial_conflicts
    
    def check_status(self, supervisor: Supervisor | None, state: State) -> None:
        """
        Checks whether the status of the agent, like if a target has been met, and triggers the corresponding event for the supervisor.

        Parameters:
        supervisor: the supervisor to trigger
        state: the true world (used for checking if close to another agent)
        """
        self.conflict_timeout += 1
        if (self.conflict_timeout >= 3):
            self._check_for_spatial_conflict(supervisor, state)
        
        if supervisor is None:
            return
        
        if self.current_task.task_type == TaskType.EXPLORE:
            path_valid = self._check_path_blocked(supervisor)
            if not path_valid:
                return
            distance_to_target = np.linalg.norm(np.array([self.current_task.y, self.current_task.x]) - np.array([self.y, self.x]))
            if distance_to_target <= 1.0:
                supervisor.trigger_event(InterventionEvent(
                    type=EventType.TASK_COMPLETED,
                    description=f"{self.name} completed exploration task at ({self.current_task.x}, {self.current_task.y})",
                    data={"agent": self.name}
                ))
                self._return_to_idle()
        elif self.current_task.task_type == TaskType.RESCUE:
            distance_to_target = np.linalg.norm(np.array([self.current_task.y, self.current_task.x]) - np.array([self.y, self.x]))
            if distance_to_target <= 1.0:
                if self.current_task.search_status == SearchStatus.SEARCHING:
                    self.current_task.search_status = SearchStatus.RESCUING
                    self.current_task.x, self.current_task.y = 0, 0 # go to exit, currently hardcoded 0,0
                    self.current_path = self._calculate_path(self.x, self.y, self.current_task.x, self.current_task.y)
                    # TODO use maria's implementation, also think about how simulator will handle this
                elif self.current_task.search_status == SearchStatus.RESCUING:
                    supervisor.trigger_event(InterventionEvent(
                        type=EventType.TASK_COMPLETED,
                        description=f"{self.name} completed rescue task at ({self.current_task.x}, {self.current_task.y})",
                        data={"agent": self.name}
                    ))
                    self._return_to_idle()

    def _check_path_blocked(self, supervisor: Supervisor | None):
        """
        Updates the path of the agent by removing already passed nodes and checks whether the path is blocked by a new obstacle/hazard.
        If it is, it returns to idle and triggers an event for the supervisor.
        """
        if supervisor is None:
            return False
        
        if len(self.current_path) > 0:
            min_dist = float('inf')
            closest_idx = 0

            search_window = min(10, len(self.current_path))

            for i in range(search_window):
                px, py = self.current_path[i]
                dist = np.sqrt((px - self.x)**2 + (py - self.y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
                    
            # remove nodes the agent has already passed
            self.current_path = self.current_path[closest_idx:]

        if len(self.current_path) > 0:
            path_blocked = False
            for px, py in self.current_path[:self.scan_radius]: 
                if int(px) == self.x and int(py) == self.y:
                    continue

                if not self._is_passible((px, py)):
                    logger.debug(f"Agent {self.name} detected an obstacle/hazard on its upcoming path at ({px}, {py})! Requesting new task/path.")
                    path_blocked = True
                    break
                    
            if path_blocked:
                self._return_to_idle() 
                supervisor.trigger_event(InterventionEvent(type=EventType.PATH_BLOCKED, 
                                                        description=f"{self.name} detected a blocked path.",
                                                        data={"agent": self.name}))
                return False
        return True

    def _predict_next_state(self, action: AgentAction, depth: int) -> Self | None:
        target_x, target_y = self._target_cell(action)
        
        if 0 <= target_x < self.world_width and 0 <= target_y < self.world_height:
            if (self.x, self.y) != (target_x, target_y) and self.perception.agents.matrix[target_y, target_x] > 0:
                return None
                
        return super()._predict_next_state(action, depth)
    
    def _is_passible(self, target_cell: tuple[int, int]) -> bool:
        """
        Decides whether a cell is passible
        (e.g. not walking into a wall)

        Parameters:
        target_cell: The cell to check (x, y)

        Returns:
        Whether the cell is passible
        """
        target_cell_x, target_cell_y = target_cell
        if (target_cell_x < 0 or target_cell_x >= self.world_width or 
            target_cell_y < 0 or target_cell_y >= self.world_height): # Out of bounds
            return False

        if self.perception.victims[target_cell_y][target_cell_x] == VictimPresence.PRESENT: # Hit victim
            return False

        if self.perception.fire.matrix[target_cell_y][target_cell_x] == FireLevel.BURNING: # On fire
            return False
        
        if self.perception.agents[target_cell_y][target_cell_x] == AgentPresence.PRESENT: # Another agent is there
            return False
        
        return self.perception.traversability[target_cell_y][target_cell_x] == TraversabilityLevel.TRAVERSABLE # Didn't hit wall


    def _check_for_spatial_conflict(self, supervisor: Supervisor | None, state: State):
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
                if supervisor is not None:
                    supervisor.trigger_event(InterventionEvent(
                        type=EventType.SPATIAL_CONFLICT,
                        description=f"{self.name} detected a spatial conflict locally.",
                        data={"agent": self.name}
                        ))
                self.conflict_timeout = 0
                self.spatial_conflicts += 1
                return
    
    def _return_to_idle(self) -> None:
        """
        Helper function to return to idle state after completing a task
        """
        self.current_task = Task(TaskType.IDLE, -1, -1)
        self.current_path = []
    
    def _objective_terminal(self) -> float:
        """
        Calculates the agent's predicted terminal reward - calculated only for the last step

        Returns:
        The objective value
        """
        if (self.current_task.task_type == TaskType.EXPLORE or self.current_task.task_type == TaskType.RESCUE):
            return 0
        
        w_exploration = 300
        exploration = -w_exploration * self._exploration_penalty()

        return exploration * self.horizon
    
    def _objective_stage(self) -> float:
        """
        Calculates the agent's predicted stage reward - calculated for every step

        Returns:
        The objective value
        """
        if (self.current_task.task_type == TaskType.EXPLORE or self.current_task.task_type == TaskType.RESCUE):
            return self._objective_targeted()
        
        return self._objective_wander()

    def _objective_wander(self) -> float:
        """
        Calculates the agents perceived objective value

        Returns:
        The objective value
        """
        w_discovery, w_safety, w_confidence, w_repulsion = 100, 10, 3, 50 # To be adjusted

        discovery   =  w_discovery * self._discovery_score()
        safety      = -w_safety * self._safety_penalty()
        confidence  =  w_confidence * self._confidence_score()
        repulsion   =  -w_repulsion * self._agent_repulsion_penalty()

        # print("Discovery:", discovery, "Safety:", safety, "Confidence:", confidence, "Repulsion:", repulsion)

        return discovery + safety + confidence + repulsion
    
    def _objective_targeted(self) -> float:
        """
        Goes to a certain objective tile (e.g. a frontier) and then switches to wander mode
        """
        w_discovery, w_safety, w_confidence, w_path, w_repulsion = 10, 5, 1, 100, 200 # To be adjusted

        safety      = -w_safety * self._safety_penalty()
        confidence  =  w_confidence * self._confidence_score()
        path_score = w_path * self._calculate_path_score()
        repulsion   =  -w_repulsion * self._agent_repulsion_penalty()

        #? maybe here we also do discovery? such that it explores on the way
        # discovery =  w_discovery * self._discovery_score()
        
        # print("path_score:", path_score, "Safety:", safety, "Confidence:", confidence, "Repulsion:", repulsion)

        return safety + confidence + path_score + repulsion # + discovery
    
    def _agent_repulsion_penalty(self) -> float:
        """
        Calculates a penalty based on proximity to other agents in the current evaluated state.
        This forces the MPC to route around other agents instead of stepping into them.
        """
        penalty = 0.0        
        agent_ys, agent_xs = np.nonzero(self.perception.agents.matrix)

        for y, x in zip(agent_ys, agent_xs):
            # skip self
            dist = np.sqrt((self.x - x)**2 + (self.y - y)**2)
            if dist == 0:
                continue

            # heavy penalty for stepping orthogonally or diagonally adjacent to another agent
            if dist <= 1.5: 
                penalty += 1.0
            # moderate penalty for being two tiles away
            elif dist <= 2.5:
                penalty += 0.2
                
        return min(1.0, penalty)
        
    def _calculate_path_score(self) -> float:
        """
        Calculates a continuous score that rewards forward progress along the path,
        even if the agent has to temporarily step off the path to avoid an obstacle.
        """
        if len(self.current_path) == 0:
            return 0.0
        
        min_dist = float('inf')
        closest_idx = 0
        search_window = min(10, len(self.current_path))
        
        for i in range(search_window):
            px, py = self.current_path[i]
            dist = np.sqrt((px - self.x)**2 + (py - self.y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        base_score = 0.5 
        progress_reward = closest_idx * 0.1
        detour_penalty = min_dist * 0.05 
        
        score = base_score + progress_reward - detour_penalty
        
        return min(1.0, max(0.0, score))
    
    def _calculate_path(self, start_x: int, start_y: int, target_x: int, target_y: int) -> list[tuple[int, int]]:
        """
        Calculates a path from the current position to the target position using BFS. Returns a list of (x, y) tuples representing the path.
        """
        path = fast_bfs_array(start_x, start_y,
                             target_x, target_y,
                             self.perception.traversability.matrix,
                             self.perception.vulnerability.matrix,
                             self.perception.fire.matrix,
                             TraversabilityLevel.UNTRAVERSABLE.value, VictimPresence.PRESENT.value, 
                             FireLevel.BURNING.value, AgentPresence.PRESENT.value)
        return [(int(pt[0]), int(pt[1])) for pt in path]