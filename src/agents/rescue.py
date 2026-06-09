from collections import deque

import numpy as np
from numpy.typing import NDArray
from typing import Self

from ..agent import Agent
from ..constants import AgentAction, TraversabilityLevel
from .victim import Victim

EXIT_X, EXIT_Y  = 0, 0
EXIT_ZONE       = 3     # victims despawn in the EXIT_ZONE × EXIT_ZONE square at (0,0)
TRUST_THRESHOLD = 0.4   # switch from approach-mode to lead-mode above this
MAX_FOLLOW_DIST = 5.0   # leash: wait if victim falls this many cells behind
SIGMA_ROBOT     = 0.1   # how fast the trust estimate updates per unit of Δdistance


class RescueAgent(Agent):
    """
    Two-mode rescue robot.

    Mode 1 – approach  (estimated_trust < TRUST_THRESHOLD):
        Move toward the victim to build trust via proximity.
        Uses a sweet-spot strategy: if already within the victim's perception
        radius, nudge exitward instead of crowding the victim.

    Mode 2 – lead      (estimated_trust >= TRUST_THRESHOLD):
        Follow the BFS exit map toward (0,0), but wait if the victim
        falls more than MAX_FOLLOW_DIST cells behind (leash condition).
    """

    def __init__(self, name: str, x: int, y: int, width: int, height: int,
                 decay: float, scan_accuracy: float, scan_radius: int, scan_falloff: bool,
                 assigned_victim: Victim):
        super().__init__(name, x, y, width, height, decay, scan_accuracy, scan_radius, scan_falloff)
        self._assigned_victim: Victim = assigned_victim
        self._estimated_trust: float = 0.0
        self._prev_victim_dist: float | None = None
        self._exit_dist: NDArray = np.full((height, width), -1, dtype=np.int32)

    def copy(self) -> Self:
        copy = super().copy()
        copy._assigned_victim = self._assigned_victim
        copy._estimated_trust = self._estimated_trust
        copy._prev_victim_dist = self._prev_victim_dist
        copy._exit_dist = np.copy(self._exit_dist)
        return copy

    # ------------------------------------------------------------------
    # BDI entry point
    # ------------------------------------------------------------------

    def get_action(self) -> AgentAction:
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

        Victim moved closer → positive Δ → estimated trust rises.
        Victim moved away   → negative Δ → estimated trust falls.
        Victim stationary   → tiny positive nudge (sustained presence still builds trust).
        """
        curr_dist = self._dist_to_victim()
        if self._prev_victim_dist is None:
            self._prev_victim_dist = curr_dist
            return self._estimated_trust

        delta = self._prev_victim_dist - curr_dist  # positive = victim came closer
        if delta == 0:
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
            return AgentAction.WAIT
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
            if self.perception.traversability[ny][nx] == TraversabilityLevel.UNTRAVERSIBLE:
                continue
            dist = abs(nx - tx) + abs(ny - ty)
            if dist < best_dist:
                best_dist = dist
                best_action = action

        return best_action

    def _dist_to_victim(self) -> float:
        v = self._assigned_victim
        return float(np.sqrt((self.x - v.x) ** 2 + (self.y - v.y) ** 2))

    def _rebuild_exit_bfs(self) -> None:
        """
        BFS from exit (0,0) across all traversable cells in the robot's current
        perception. exit_dist[y][x] = min steps to reach (0,0); -1 = unreachable.
        Rebuilt every step so newly scanned traversability is reflected immediately.
        """
        dist = np.full((self.world_height, self.world_width), -1, dtype=np.int32)

        if self.perception.traversability[EXIT_Y][EXIT_X] != TraversabilityLevel.TRAVERSIBLE:
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
                if self.perception.traversability[ny][nx] != TraversabilityLevel.TRAVERSIBLE:
                    continue
                dist[ny][nx] = dist[cy][cx] + 1
                queue.append((nx, ny))

        self._exit_dist = dist
