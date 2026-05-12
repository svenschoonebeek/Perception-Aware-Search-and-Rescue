from collections import deque

import numpy as np

from .state import State
from .constants import AgentAction, FireLevel, Scan, Task, TaskType, TraversabilityLevel
import numpy as np

from numpy.typing import NDArray

from dataclasses import dataclass
from typing import Self

from tcod.map import compute_fov
from tcod.libtcodpy import FOV_RESTRICTIVE

# Threshold for what value of traversability causes a tile to become 'opaque' to the LoS check
LOS_THRESHOLD = 1

class Agent:
    def __init__(self, name: str, x: int, y: int, width: int, height: int, 
                 decay: float, scan_accuracy: float, scan_radius: int, scan_falloff: bool):
        # Basic setup
        self.name = name
        self.x = x
        self.y = y
        self.world_width = width
        self.world_height = height
        self.perception = State(width, height)
        self.perception.agents[y][x] = 1

        # Scan parameters
        self.decay = decay # Certainty decay per time step [0,1]
        self.scan_accuracy = scan_accuracy # Scan accuracy [0,1]
        self.scan_radius = scan_radius # How far the agent can see when it scans
        self.scan_falloff = scan_falloff # Whether scan accuracy should decrease with distance

        # Metrics
        self.illegal_moves = 0
        self.move_history: list[tuple[int, int]] = [(x, y)] # Cells the agent has been to
        self.explored: np.ndarray = np.zeros((height, width), dtype=bool) # Cells the agent has scanned
        
        # Task info
        self.current_task = Task(TaskType.IDLE, -1, -1)
        self.current_path: list[tuple[int, int]] = []
    
    def copy(self) -> Self:
        cls = type(self)
        copy = cls.__new__(cls)

        copy.name = self.name
        copy.x = self.x
        copy.y = self.y
        copy.world_width = self.world_width
        copy.world_height = self.world_height
        copy.perception = self.perception.copy()

        copy.decay = self.decay
        copy.scan_accuracy = self.scan_accuracy
        copy.scan_radius = self.scan_radius
        copy.scan_falloff = self.scan_falloff

        copy.illegal_moves = self.illegal_moves
        copy.move_history = self.move_history.copy()
        copy.explored = np.copy(self.explored)

        copy.current_task = Task(self.current_task.task_type, self.current_task.x, self.current_task.y)
        return copy

    def scan(self, state: State) -> Scan:
        """
        Retrieves information from the state passed in (with some noise/uncertainties?)

        Parameters:
        state: The true world to scan from
        """
        visible = self._tiles_in_radius() & self._tiles_in_los(state)
        confidence_bounds = np.where(visible, self._tile_accuracy(), 0)

        self.perception.traversability[visible] = trvs = state.traversability[visible]
        self.perception.vulnerability[visible] = vuln = state.vulnerability[visible]
        self.perception.victims[visible] = vict = state.victims[visible]
        self.perception.agents[visible] = agnt = state.agents[visible]
        self.perception.fire[visible] = fire = state.fire[visible]
        self.perception.confidence.matrix = np.maximum(self.perception.confidence.matrix - self.decay, confidence_bounds)

        self.explored[visible] = True

        ys, xs = np.nonzero(visible)
        conf = self.perception.confidence.matrix[visible]

        return Scan(self.name, self.x, self.y, xs, ys, trvs, vuln, vict, agnt, fire, conf)
    
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

    def move_to(self, x: int, y: int) -> None:
        """
        Commits the agent to a position, after the simulator does the validation.
        """
        self.perception.agents[self.y][self.x] = 0
        self.x, self.y = x, y
        self.move_history.append((self.x, self.y))
        self.perception.agents[self.y][self.x] = 1

    def get_action(self) -> AgentAction:
        """
        To be overwritten for each research question.

        Returns:
        The action the agent wants to perform given its internal perception matrix
        """
        return AgentAction.MOVE_RIGHT

    def check_task_status(self) -> None:
        """
        Checks whether the current task is completed, and if so, switches to idle
        """
        if self.current_task.task_type == TaskType.IDLE:
            return
        
        target_pos = (self.current_task.x, self.current_task.y)
        if self.current_task.task_type == TaskType.EXPLORE:
            if (self.x, self.y) == (self.current_task.x, self.current_task.y):
                self._return_to_idle()
    
    def _calculate_path(self, target_x: int, target_y: int) -> list[tuple[int, int]]:
        """
        Calculates a path from the agent's current position to the target using BFS.
        Returns a list of (x, y) coordinates representing the path.
        """
        start = (self.x, self.y)
        target = (target_x, target_y)
        
        if start == target:
            return [start]
            
        queue = deque([[start]])
        seen = set([start])
        
        while queue:
            path = queue.popleft()
            x, y = path[-1]
            
            if (x, y) == target:
                return path
                
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < self.world_width and 0 <= ny < self.world_height:
                    if (nx, ny) not in seen:
                        if self.perception.traversability[ny][nx] != TraversabilityLevel.UNTRAVERSIBLE:
                            seen.add((nx, ny))
                            queue.append(path + [(nx, ny)])
                            
        return []
    
    def _is_path_blocked(self) -> bool:
        """
        Checks if the currently planned path contains any newly discovered walls or fires.
        """
        for x, y in self.current_path:
            # Did we just discover a wall here?
            if self.perception.traversability[y][x] == TraversabilityLevel.UNTRAVERSIBLE:
                return True
            #? should count fire as blocking? I don't think so
            # if self.perception.fire[y][x] == FireLevel.BURNING:
            #     return True
                
        return False
    
    def _return_to_idle(self) -> None:
        """
        Helper function to return to idle state after completing a task
        """
        self.current_task = Task(TaskType.IDLE, -1, -1)
        self.current_path = []
    
    def _tiles_in_radius(self) -> NDArray[np.bool]:
        """
        Returns a mask of all tiles within scan_radius of the agent
        """
        y, x = np.ogrid[:self.world_height, :self.world_width]
        return (y - self.y)**2 + (x - self.x)**2 <= self.scan_radius**2 # euclidean distance
    
    def _tiles_in_los(self, state: State) -> NDArray[np.bool]:
        """
        Basically just TCOD's Restrictive Precise Angle Shadowcasting FOV calculation

        Returns a mask of tiles which are within line of sight from the agent. Note that TCOD's definition
        of radius is different to our own, so we need to combine it with a radius mask later. 
        """
        transparency = np.where(state.traversability.matrix >= LOS_THRESHOLD, 0, 1)
        return compute_fov(transparency, (self.y, self.x), self.scan_radius, True, FOV_RESTRICTIVE)
    
    def _tile_accuracy(self) -> NDArray[np.float64]:
        if self.scan_falloff:
            var = (self. scan_radius / 3) ** 2
            y, x = np.ogrid[:self.world_height, :self.world_width]
            return self.scan_accuracy * np.exp(-0.5 * ((x - self.x)**2 + (y - self.y)**2) / var)
        else:
            return np.full((self.world_height, self.world_width), self.scan_accuracy)