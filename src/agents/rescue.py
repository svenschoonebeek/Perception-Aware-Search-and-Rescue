from collections import deque

import numpy as np
from numpy.typing import NDArray
from typing import Self
import logging

from ..agent import Agent
from .task_mpc import TaskMpcAgent
from ..constants import *
from .victim import Victim
from src.supervisor import Supervisor
from src.state import State

EXIT_X, EXIT_Y  = 0, 0
EXIT_ZONE       = 3     # victims despawn in the EXIT_ZONE × EXIT_ZONE square at (0,0)
TRUST_THRESHOLD      = 0.4   # switch from approach-mode to lead-mode above this
MAX_FOLLOW_DIST      = 5.0   # leash: wait if victim falls this many cells behind
SIGMA_ROBOT          = 0.1   # how fast the trust estimate updates per unit of Δdistance
PERSONAL_SPACE_DIST  = 1.5   # victim retreats when robot within 1 cell; retreats land ≤ this
LEASH_TIMEOUT        = 10    # ticks of leash-wait before abandoning rescue
NO_PROGRESS_TIMEOUT  = 15    # ticks without moving (while rescuing) before abandoning rescue

logger = logging.getLogger(__name__)

class RescueAgent(TaskMpcAgent):
    """
    Two-mode rescue robot.

    Mode 1 - approach  (estimated_trust < TRUST_THRESHOLD):
        Move toward the victim to build trust via proximity.
        Uses a sweet-spot strategy: if already within the victim's perception
        radius, nudge exitward instead of crowding the victim.

    Mode 2 - lead      (estimated_trust >= TRUST_THRESHOLD):
        Follow the BFS exit map toward (0,0), but wait if the victim
        falls more than MAX_FOLLOW_DIST cells behind (leash condition).
    """

    def __init__(self, name: str, x: int, y: int, width: int, height: int,
                 decay: float, scan_accuracy: float, scan_radius: int, scan_falloff: bool,
                 assigned_victim: Victim | None, capabilities: Capabilities):
        super().__init__(name, x, y, width, height, decay, scan_accuracy, scan_radius, scan_falloff, capabilities)
        self._assigned_victim: Victim | None = assigned_victim
        self._estimated_trust: float = 0.0
        self._prev_victim_dist: float | None = None
        self._exit_dist: NDArray = np.full((height, width), -1, dtype=np.int32)
        self._leash_wait_ticks: int = 0
        self._no_progress_ticks: int = 0
        self._last_rescue_pos: tuple[int, int] | None = None

    def copy(self) -> Self:
        copy = super().copy()
        copy._assigned_victim = self._assigned_victim
        copy._estimated_trust = self._estimated_trust
        copy._prev_victim_dist = self._prev_victim_dist
        copy._exit_dist = np.copy(self._exit_dist)
        copy._leash_wait_ticks = self._leash_wait_ticks
        copy._no_progress_ticks = self._no_progress_ticks
        copy._last_rescue_pos = self._last_rescue_pos
        return copy
    
    def set_assigned_victim(self, victim: Victim) -> None:
        """
        Sets assigned victim, to be used in rescue agent
        """
        self._assigned_victim = victim

    def check_status(self, supervisor: Supervisor | None, state: State) -> None:
        logger.debug(f"rescue agent checking status with victim {self._assigned_victim.name if self._assigned_victim else None} at ({self._assigned_victim.x}, {self._assigned_victim.y})" if self._assigned_victim else "rescue agent checking status with no assigned victim")
        self.conflict_timeout += 1
        if (self.conflict_timeout >= 3):
            self._check_for_spatial_conflict(supervisor, state)
        
        if supervisor is None:
            return
        
        if self.current_task.type == TaskType.EXPLORE:
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
        elif self.current_task.type == TaskType.RESCUE:
            distance_to_target = np.linalg.norm(np.array([self.current_task.y, self.current_task.x]) - np.array([self.y, self.x]))
            if self.current_task.search_status == SearchStatus.SEARCHING:
                if distance_to_target <= 3.0:
                    self.current_task.search_status = SearchStatus.RESCUING
                    self._leash_wait_ticks = 0
                    self._no_progress_ticks = 0
                    self._last_rescue_pos = None
                    logger.debug(f"{self.name} is starting rescue for victim {self.current_task.victim_id} at ({self.current_task.x}, {self.current_task.y})")
                    supervisor.trigger_event(InterventionEvent(
                        type=EventType.INITIALIZING_RESCUE,
                        description=f"{self.name} found a victim at ({self.current_task.x}, {self.current_task.y}) and is initializing rescue.",
                        data={"agent": self.name, "victim_id": self.current_task.victim_id}
                    ))
                    self.current_task.x, self.current_task.y = 0, 0 # go to exit, currently hardcoded 0,0
                    self.current_path = self._calculate_path(self.x, self.y, self.current_task.x, self.current_task.y)
                    # TODO use maria's implementation and her "completed rescue" indication
                
            elif self.current_task.search_status == SearchStatus.RESCUING:
                if self._assigned_victim is not None and self._assigned_victim.x < EXIT_ZONE and self._assigned_victim.y < EXIT_ZONE:
                    logger.debug(f"{self.name} is completing rescue for victim {self.current_task.victim_id} at ({self.current_task.x}, {self.current_task.y})")
                    supervisor.trigger_event(InterventionEvent(
                        type=EventType.VICTIM_RESCUED,
                        description=f"{self.name} completed rescue task at ({self.current_task.x}, {self.current_task.y})",
                        data={"agent": self.name, "victim_id": self.current_task.victim_id}
                    ))
                    self._return_to_idle()
                    return

                # No-progress timeout: agent position unchanged while rescuing
                curr_pos = (self.x, self.y)
                if self._last_rescue_pos == curr_pos:
                    self._no_progress_ticks += 1
                else:
                    self._no_progress_ticks = 0
                self._last_rescue_pos = curr_pos

                timed_out = (
                    self._leash_wait_ticks >= LEASH_TIMEOUT
                    or self._no_progress_ticks >= NO_PROGRESS_TIMEOUT
                )
                if timed_out:
                    reason = "leash wait" if self._leash_wait_ticks >= LEASH_TIMEOUT else "no movement progress"
                    logger.debug(f"{self.name} abandoning rescue of {self.current_task.victim_id} after {reason} timeout")
                    self._leash_wait_ticks = 0
                    self._no_progress_ticks = 0
                    self._last_rescue_pos = None
                    supervisor.trigger_event(InterventionEvent(
                        type=EventType.RESCUE_ABANDONED,
                        description=f"{self.name} abandoned rescue of {self.current_task.victim_id} due to {reason}",
                        data={"agent": self.name, "victim_id": self.current_task.victim_id}
                    ))
                    self._return_to_idle()

    def get_action(self) -> AgentAction:
        if (self.current_task.type == TaskType.RESCUE and self.current_task.search_status == SearchStatus.RESCUING):
            return self.get_rescue_action()
        else:
            return super().get_action()
    
    # ------------------------------------------------------------------
    # BDI entry point
    # ------------------------------------------------------------------
    def get_rescue_action(self) -> AgentAction:
        self._rebuild_exit_bfs()
        estimated_trust = self._estimate_trust()

        if estimated_trust < TRUST_THRESHOLD:
            return self._approach()
        else:
            return self._lead_to_exit()

    # ------------------------------------------------------------------
    # Belief: estimate victim's trust from proximity change
    # ------------------------------------------------------------------

    def _estimate_trust(self) -> float:
        """
        τ̂(k+1) = clamp(τ̂(k) + σ_robot · Δdist, 0, 1)

        Victim moved closer          → positive Δ → estimated trust rises.
        Victim moved away (far)      → negative Δ → estimated trust falls.
        Victim retreated (≤1 cell)   → personal space, not distrust — treated as neutral.
        Victim stationary            → tiny positive nudge.
        """
        curr_dist = self._dist_to_victim()
        if self._prev_victim_dist is None:
            self._prev_victim_dist = curr_dist
            return self._estimated_trust

        delta = self._prev_victim_dist - curr_dist  # positive = victim came closer
        if delta < 0 and self._prev_victim_dist <= PERSONAL_SPACE_DIST:
            delta = 0.01  # victim was adjacent and retreated — personal space, not distrust
        elif delta == 0:
            delta = 0.01
        self._estimated_trust = max(0.0, min(1.0, self._estimated_trust + delta * SIGMA_ROBOT))
        self._prev_victim_dist = curr_dist
        return self._estimated_trust

    # ------------------------------------------------------------------
    # Mode 1: approach — build trust via proximity
    # ------------------------------------------------------------------

    def _approach(self) -> AgentAction:
        """
        Sweet-spot approach (Chen et al. 2017 viewpoint maintenance):
          - Outside victim's perception radius: move directly toward victim
          - Inside radius: nudge exitward as a directional hint while remaining
            visible — this demonstrates the path without crowding the victim
        """
        if self._assigned_victim is None:
            logger.debug("approach mode but no assigned victim, returning WAIT")
            return AgentAction.WAIT

        dist = self._dist_to_victim()
        if dist > self._assigned_victim.scan_radius:
            return self._step_towards(self._assigned_victim.x, self._assigned_victim.y)
        else:
            return self._follow_bfs_to_exit()

    # ------------------------------------------------------------------
    # Mode 2: lead — guide victim to exit along BFS path
    # ------------------------------------------------------------------

    def _lead_to_exit(self) -> AgentAction:
        """
        Follow BFS toward (0,0). If the victim falls behind the leash distance,
        wait rather than racing ahead and leaving them.
        """
        if self._dist_to_victim() > MAX_FOLLOW_DIST:
            self._leash_wait_ticks += 1
            return AgentAction.WAIT
        self._leash_wait_ticks = 0
        return self._follow_bfs_to_exit()

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _follow_bfs_to_exit(self) -> AgentAction:
        """Move to the adjacent cell with the smallest BFS distance to exit."""
        if self.x == EXIT_X and self.y == EXIT_Y:
            return AgentAction.WAIT

        best_action = AgentAction.WAIT
        best_dist = float('inf')

        for action in AgentAction:
            if action == AgentAction.WAIT:
                continue
            tx, ty = self._target_cell(action)
            if not (0 <= tx < self.world_width and 0 <= ty < self.world_height):
                continue
            d = self._exit_dist[ty][tx]
            if d < 0:
                continue
            if d < best_dist:
                best_dist = d
                best_action = action

        return best_action

    def _step_towards(self, tx: int, ty: int) -> AgentAction:
        """One step toward (tx, ty) by Manhattan distance, through traversable cells only."""
        best_action = AgentAction.WAIT
        best_dist = float('inf')

        for action in AgentAction:
            if action == AgentAction.WAIT:
                continue
            nx, ny = self._target_cell(action)
            if not (0 <= nx < self.world_width and 0 <= ny < self.world_height):
                continue
            if self.perception.traversability[ny][nx] == TraversabilityLevel.UNTRAVERSABLE:
                continue
            dist = abs(nx - tx) + abs(ny - ty)
            if dist < best_dist:
                best_dist = dist
                best_action = action

        return best_action

    def _dist_to_victim(self) -> float:
        v = self._assigned_victim
        if v is None:
            return float('inf')
        return float(np.sqrt((self.x - v.x) ** 2 + (self.y - v.y) ** 2))

    def _rebuild_exit_bfs(self) -> None:
        """
        BFS from exit (0,0) across all traversable cells in the robot's current
        perception. exit_dist[y][x] = min steps to reach (0,0); -1 = unreachable.
        Rebuilt every step so newly scanned traversability is reflected immediately.
        """
        dist = np.full((self.world_height, self.world_width), -1, dtype=np.int32)

        if self.perception.traversability[EXIT_Y][EXIT_X] != TraversabilityLevel.TRAVERSABLE:
            self._exit_dist = dist
            return

        dist[EXIT_Y][EXIT_X] = 0
        queue: deque[tuple[int, int]] = deque([(EXIT_X, EXIT_Y)])

        while queue:
            cx, cy = queue.popleft()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.world_width and 0 <= ny < self.world_height):
                    continue
                if dist[ny][nx] != -1:
                    continue
                if self.perception.traversability[ny][nx] != TraversabilityLevel.TRAVERSABLE:
                    continue
                dist[ny][nx] = dist[cy][cx] + 1
                queue.append((nx, ny))

        self._exit_dist = dist
