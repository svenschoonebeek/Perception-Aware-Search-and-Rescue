from __future__ import annotations
from collections import deque
from .state import State
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self
from tcod.map import compute_fov
from tcod.libtcodpy import FOV_RESTRICTIVE

from .constants import AgentAction, Capabilities, FireLevel, Scan, Task, TaskType, TraversabilityLevel

if TYPE_CHECKING:
    from .supervisor import Supervisor
    from .agents.victim import Victim

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
        self.infeasible_states = 0
        self.move_history: list[tuple[int, int]] = [(x, y)] # Cells the agent has been to
        self.explored: np.ndarray = np.zeros((height, width), dtype=bool) # Cells the agent has scanned at any point in the run
        self.discovered: np.ndarray = np.zeros((height, width), dtype=bool) # Cells the agent scanned for the first time this step

        self.assignable = False # Whether the supervisor can assign a task to this agent

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
        copy.infeasible_states = self.infeasible_states
        copy.move_history = self.move_history.copy()
        copy.explored = np.copy(self.explored)
        copy.discovered = np.copy(self.discovered)

        copy.assignable = self.assignable

        return copy

    def scan(self, state: State) -> Scan:
        """
        Retrieves information from the state passed in (with some noise/uncertainties?)

        Parameters:
        state: The true world to scan from
        """
        visible = self._tiles_in_radius() & self._tiles_in_los(state)

        confidence, accuracy = self.perception.confidence.matrix, self._tile_accuracy()
        confidence_bounds = np.where(visible, confidence + accuracy * (1 - confidence), 0)
        self.perception.confidence.matrix = np.maximum(self.perception.confidence.matrix - self.decay, confidence_bounds)

        self.perception.traversability[visible] = trvs = state.traversability[visible]
        self.perception.vulnerability[visible] = vuln = state.vulnerability[visible]
        self.perception.victims[visible] = vict = state.victims[visible]
        self.perception.agents[visible] = agnt = state.agents[visible]
        self.perception.fire[visible] = fire = state.fire[visible]

        self.discovered = visible & ~self.explored
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
        The target cell indices (x, y)
        """
        # up, down, left, right, wait
        dy = np.array([-1, 1, 0, 0, 0])
        dx = np.array([0, 0, -1, 1, 0])
        return self.x + dx[action], self.y + dy[action]

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
    
    def check_status(self, supervisor: Supervisor | None, state: State) -> None:
        """
        Checks whether the agent has completed its current task, base Agent doesn't use it
        """
        pass

    def get_current_task(self) -> Task | None:
        """
        Returns the current task of the agent, to be used by the supervisor for task allocation
        """
        return None
    
    def set_current_task(self, task: Task) -> None:
        """
        Sets the current task of the agent, to be used by the supervisor for task allocation
        """
        pass

    def get_current_path(self) -> list[tuple[int, int]] | None:
        """
        Returns the current path of the agent, to be used by the supervisor for task allocation
        """
        return None

    def set_current_path(self, path: list[tuple[int, int]]) -> None:
        """
        Sets the current path of the agent, to be used by the supervisor for task allocation
        """
        pass

    def get_spatial_conflicts(self) -> int:
        """
        Returns the number of spatial conflicts the agent has been involved in, to be used by the supervisor for task allocation
        """
        return 0

    def get_capabilities(self) -> Capabilities:
        """
        Returns the capabilities of the agent, to be used by the supervisor for task allocation
        """
        return Capabilities()
    
    def set_assigned_victim(self, victim: Victim) -> None:
        """
        Sets assigned victim, to be used in rescue agent
        """
        pass
    
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