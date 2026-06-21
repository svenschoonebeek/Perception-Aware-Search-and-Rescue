import logging
import numpy as np
from typing import Self
from numpy.typing import NDArray

from .mpc import MpcAgent
from ..constants import AgentAction, FireLevel, VulnerabilityLevel

logger = logging.getLogger(__name__)

class SddTmpcAgent(MpcAgent):
    def __init__(self, name: str, x: int, y: int, width: int, height: int, 
                 decay: float, scan_accuracy: float, scan_radius: int, scan_falloff: bool):
        super().__init__(name, x, y, width, height, decay, scan_accuracy, scan_radius, scan_falloff)
        self.fire_spread_rate = 1.0

    def get_action(self) -> AgentAction:
        action = super().get_action()
        x, y = self._target_cell(action)
        
        vuln = self.perception.vulnerability[y, x]
        conf = self.perception.confidence[y, x]
        tube_size = 1 if vuln - conf > 0 else 0
        self.tube_size += tube_size
        
        return action

    def _is_feasible(self) -> bool:
        """
        Checks whether a given state violates constraints (e.g. not walking into a wall)
        Assumes given state (centre of the tube) is in the admissible set X
        For SDD-TMPC, this now also checks whether tripping from this position results violates constraints

        Returns:
        True if no constraints were violated, else False
        """

        def cell_feasible(x: int, y: int) -> bool:
            wall = self.perception.traversability[y, x] == 1
            victim = self.perception.victims[y, x] == 1
            fire = self.perception.fire[y, x] == FireLevel.BURNING
            return not(wall or victim or fire)

        if not cell_feasible(self.x, self.y):
            return False

        # Could the agent trip?
        vulnerability = self.perception.vulnerability[self.y, self.x]
        confidence = self.perception.confidence[self.y, self.x]
        if vulnerability - confidence <= 0: # No chance of tripping
            return True

        # Check if tripping after this move results in an infeasible state
        # [up, down, left, right]
        dy = [-1, 1, 0, 0]
        dx = [0, 0, -1, 1]
        for i in range(4):
            new_x, new_y = self.x + dx[i], self.y + dy[i]
            out_of_bounds = not (0 <= new_x < self.world_width and 0 <= new_y < self.world_height)
            if out_of_bounds: return False  # Return False if element of tube not in admissible set
            if not cell_feasible(new_x, new_y):
                return False
            
        return True
    
    def _closest_unexplored(self) -> NDArray:
        """
        Uses multi-source BFS to calculate the shortest distance from each explored tile to closest unexplored tile
        should be JIT-compiled with numba for performance - currently infeasible

        Returns:
        A world-sized ndarray where each tile is:
            0 if the tile is unexplored or a wall
            x if the tile is explored, where x is the true shortest distance to the closest unexplored tile
        """
        distance = np.full((self.world_height, self.world_width), -1, dtype=np.int16)

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
        moving_state.perception.confidence.matrix = np.full((self.world_height, self.world_width), 0.999)
        while head < tail:
            (y, x) = queue[head]
            head += 1

            for i in range(4):
                ny: int = y + dy[i]
                nx: int = x + dx[i]

                if 0 <= ny < self.world_height and 0 <= nx < self.world_width:
                    moving_state.y, moving_state.x = ny, nx
                    if distance[ny, nx] == -1 and moving_state._is_feasible():
                        distance[ny, nx] = distance[y, x] + 1
                        queue[tail] = (ny, nx)
                        tail += 1

        return distance