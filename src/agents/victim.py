from enum import Enum
from typing import Self

import numpy as np

from ..agent import Agent
from ..constants import AgentAction, AgentPresence, FireLevel, VulnerabilityLevel, TraversabilityLevel, VictimPresence

EXIT_X, EXIT_Y = 0, 0

# Speed as move-probability per step, derived from paper's m/s ranges (Table 2)
_BASE_MOVE_PROB = {
    "child":   0.525,
    "adult":   0.90,
    "elderly": 0.65,
}
_DANGER_SPEED_MULTIPLIER = 1.5  # panic boost; capped at 1.0

# Trust model  τᵢ(k+1) = clamp(τᵢ(k) + δᵢ·σ(k), 0, τᵢᵐᵃˣ)
_TRUST_INIT  = 0.5
_TRUST_MAX   = 1.0
# δᵢ: propensity to trust — altruists open up faster, selfish people barely update
_TRUST_SENSITIVITY = {"altruist": 1.2, "egoist": 1.0, "selfish": 0.6}
# σ weights — asymmetric: danger loss is 3× stronger than safety gain (trust is fragile)
_SIGMA_PROX_WEIGHT        = 0.05   # per unit of distance the robot moved closer
_SIGMA_DANGER_GAIN_WEIGHT = 0.10   # per danger-unit improvement (slow recovery)
_SIGMA_DANGER_LOSS_WEIGHT = 0.30   # per danger-unit worsening  (fast erosion)
_TRUST_FOLLOW_THRESHOLD   = 0.4    # above this, victim defers to robot and follows it


class Personality(Enum):
    ALTRUIST = "altruist"   # wanders to group others; flees only when in danger
    EGOIST   = "egoist"     # like altruist, but flips to selfish upon conflict with a selfish
    SELFISH  = "selfish"    # always moves to exit as fast as possible


class AgeGroup(Enum):
    CHILD   = "child"
    ADULT   = "adult"
    ELDERLY = "elderly"


class Victim(Agent):
    def __init__(self, name: str, x: int, y: int, width: int, height: int,
                 personality: Personality = Personality.EGOIST,
                 age_group: AgeGroup = AgeGroup.ADULT,
                 is_static: bool = False,
                 seed: int | None = None):
        super().__init__(
            name=name, x=x, y=y, width=width, height=height,
            decay=0.0,
            scan_accuracy=1.0,
            scan_radius=3,
            scan_falloff=False,
        )
        self.personality = personality
        self.age_group = age_group
        self._base_move_prob: float = _BASE_MOVE_PROB[age_group.value] if not is_static else 0.0
        self.rng = np.random.default_rng(seed)
        self._base_move_prob: float = _BASE_MOVE_PROB[age_group.value]
        self.trust: float = _TRUST_INIT
        self.trust_max: float = _TRUST_MAX
        self._trust_sensitivity: float = _TRUST_SENSITIVITY[personality.value]
        self._prev_danger: float | None = None
        self._prev_robot_dist: float | None = None
        self._curr_robot_pos: tuple[int, int] | None = None

    def copy(self) -> Self:
        copy = super().copy()
        copy.personality = self.personality
        copy.age_group = self.age_group
        copy._base_move_prob = self._base_move_prob
        copy.trust = self.trust
        copy.trust_max = self.trust_max
        copy._trust_sensitivity = self._trust_sensitivity
        copy._prev_danger = self._prev_danger
        copy._prev_robot_dist = self._prev_robot_dist
        copy._curr_robot_pos = self._curr_robot_pos
        return copy

    # ------------------------------------------------------------------
    # BDI entry point
    # ------------------------------------------------------------------

    def get_action(self) -> AgentAction:
        self._update_trust()
        in_danger = self._perceives_danger()

        # Personal space: always retreat if a robot is cardinally adjacent, regardless of trust
        adjacent_robot = self._nearest_adjacent_robot()
        if adjacent_robot is not None:
            action = self._move_away_from(*adjacent_robot)
            if action is not None:
                return action

        # Speed: danger triggers panic → higher move probability
        move_prob = min(1.0, self._base_move_prob * (_DANGER_SPEED_MULTIPLIER if in_danger else 1.0))
        if self.rng.random() > move_prob:
            return AgentAction.WAIT

        # High trust: follow the robot (overrides personality-based logic)
        if self.trust >= _TRUST_FOLLOW_THRESHOLD and self._curr_robot_pos is not None:
            action = self._step_towards(*self._curr_robot_pos)
            if action is not None:
                return action

        # SELFISH always flees; EGOIST and ALTRUIST wander until danger (or until
        # EGOIST is mutated to SELFISH by the simulator's conflict resolution)
        flee = in_danger or self.personality == Personality.SELFISH
        return self._move_towards_exit_safely() if flee else self._wander_or_seek()

    # ------------------------------------------------------------------
    # Personal space
    # ------------------------------------------------------------------

    def _nearest_adjacent_robot(self) -> tuple[int, int] | None:
        """Returns position of the first cardinally adjacent rescue agent, or None."""
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = self.x + dx, self.y + dy
            if (0 <= nx < self.world_width and 0 <= ny < self.world_height
                    and self.perception.agents[ny, nx] == AgentPresence.RESCUE_AGENT):
                return (nx, ny)
        return None

    def _move_away_from(self, rx: int, ry: int) -> AgentAction | None:
        """Pick the safe action that maximises manhattan distance from (rx, ry)."""
        best_action: AgentAction | None = None
        best_dist = -1
        for action in AgentAction:
            if action == AgentAction.WAIT:
                continue
            nx, ny = self._target_cell(action)
            if self._cell_safety_score(nx, ny) < 1:
                continue
            dist = abs(nx - rx) + abs(ny - ry)
            if dist > best_dist:
                best_dist = dist
                best_action = action
        return best_action

    # ------------------------------------------------------------------
    # Belief: danger perception
    # ------------------------------------------------------------------

    def _perceives_danger(self) -> bool:
        return self._cell_safety_score(self.x, self.y) <= 2

    # ------------------------------------------------------------------
    # Trust: update + helpers
    # ------------------------------------------------------------------

    def _cell_danger_level(self) -> float:
        """Converts safety score to danger level: 0.0 (safe) → 4.0 (burning)."""
        score = self._cell_safety_score(self.x, self.y)
        return 4.0 if score < 0 else float(4 - score)

    def _nearest_robot(self) -> tuple[float, tuple[int, int]] | None:
        """Returns (distance, (x, y)) of the nearest visible rescue agent, or None."""
        positions = np.argwhere(self.perception.agents.matrix == AgentPresence.RESCUE_AGENT)
        if len(positions) == 0:
            return None
        distances = np.linalg.norm(positions - np.array([self.y, self.x]), axis=1)
        idx = int(np.argmin(distances))
        return float(distances[idx]), (int(positions[idx][1]), int(positions[idx][0]))

    def _update_trust(self) -> None:
        """
        τᵢ(k+1) = clamp(τᵢ(k) + δᵢ·σ(k), 0, τᵢᵐᵃˣ)

        σ is built from two signals:
          proximity — only contributes when robot moves closer (one-directional)
          danger    — asymmetric: improvement is weighted 3× less than worsening
        No update at all when robot is out of perception range.
        """
        curr_danger = self._cell_danger_level()
        robot_info = self._nearest_robot()
        curr_dist = robot_info[0] if robot_info is not None else None
        self._curr_robot_pos = robot_info[1] if robot_info is not None else None

        if self._prev_danger is None:
            self._prev_danger = curr_danger
            self._prev_robot_dist = curr_dist
            return

        if curr_dist is None:
            self._prev_danger = curr_danger
            self._prev_robot_dist = None
            return

        sigma = 0.0

        # Proximity: positive sigma only when robot moved closer
        if self._prev_robot_dist is not None:
            dist_delta = curr_dist - self._prev_robot_dist
            if dist_delta < 0:
                sigma += -dist_delta * _SIGMA_PROX_WEIGHT

        # Danger delta: slow gain, fast loss
        danger_delta = curr_danger - self._prev_danger
        if danger_delta < 0:
            sigma += -danger_delta * _SIGMA_DANGER_GAIN_WEIGHT
        elif danger_delta > 0:
            sigma += -danger_delta * _SIGMA_DANGER_LOSS_WEIGHT

        self.trust = max(0.0, min(self.trust + self._trust_sensitivity * sigma, self.trust_max))
        self._prev_danger = curr_danger
        self._prev_robot_dist = curr_dist

    # ------------------------------------------------------------------
    # Cell safety scoring (higher = safer; -1 = impassable)
    # ------------------------------------------------------------------

    def _cell_safety_score(self, x: int, y: int) -> int:
        if not (0 <= x < self.world_width and 0 <= y < self.world_height):
            return -1
        if self.perception.traversability[y][x] != TraversabilityLevel.TRAVERSABLE:
            return -1
        fire = self.perception.fire[y][x]
        vuln = self.perception.vulnerability[y][x]
        if fire >= FireLevel.BURNING:
            return 0
        if vuln >= VulnerabilityLevel.HIGH_RISK:
            return 1
        if fire >= FireLevel.FLAMMABLE:
            return 2
        if vuln >= VulnerabilityLevel.VULNERABLE:
            return 3
        return 4

    # ------------------------------------------------------------------
    # Intention: move towards exit (selfish / in danger)
    # ------------------------------------------------------------------

    def _move_towards_exit_safely(self) -> AgentAction:
        if self.x == EXIT_X and self.y == EXIT_Y:
            return AgentAction.WAIT

        dx = EXIT_X - self.x
        dy = EXIT_Y - self.y

        preferred: list[AgentAction] = []
        if dx < 0:
            preferred.append(AgentAction.MOVE_LEFT)
        elif dx > 0:
            preferred.append(AgentAction.MOVE_RIGHT)
        if dy < 0:
            preferred.append(AgentAction.MOVE_UP)
        elif dy > 0:
            preferred.append(AgentAction.MOVE_DOWN)

        other = [a for a in AgentAction if a != AgentAction.WAIT and a not in preferred]

        # 1. Preferred direction, fully safe
        for action in preferred:
            if self._cell_safety_score(*self._target_cell(action)) >= 3:
                return action

        # 2. Any safe lateral move (navigate around obstacles)
        for action in other:
            if self._cell_safety_score(*self._target_cell(action)) >= 3:
                return action

        # 3. Preferred direction even if risky (not burning)
        for action in preferred:
            if self._cell_safety_score(*self._target_cell(action)) >= 1:
                return action

        # 4. Any traversable cell as last resort
        for action in preferred + other:
            if self._cell_safety_score(*self._target_cell(action)) >= 0:
                return action

        return AgentAction.WAIT

    # ------------------------------------------------------------------
    # Intention: wander / seek other victims (altruist / egoist)
    # ------------------------------------------------------------------

    def _wander_or_seek(self) -> AgentAction:
       victims = self.perception.victims.matrix
       positions = np.argwhere(victims == VictimPresence.PRESENT)
       positions = positions[(positions[:, 0] != self.y) | (positions[:, 1] != self.x)]

       if len(positions) > 0:
          distances = np.linalg.norm(positions - np.array([self.y, self.x]), axis=1)
          nearest_dist = distances.min()
         
          # If we've reached a victim (adjacent), form a group and head to exit
          if nearest_dist <= 1.5:  # adjacent cell
                return self._move_towards_exit_safely()
         
          # Otherwise keep seeking
          target = positions[np.argmin(distances)]
          action = self._step_towards(int(target[1]), int(target[0]))
          if action is not None:
                return action

       # No victims found: wander safely
       safe_moves = [
            a for a in AgentAction
            if a != AgentAction.WAIT and self._cell_safety_score(*self._target_cell(a)) >= 3
        ]
       if safe_moves:
           return self.rng.choice(safe_moves)

       return AgentAction.WAIT

    def _step_towards(self, tx: int, ty: int) -> AgentAction | None:
        candidates: list[tuple[int, AgentAction]] = []
        for action in AgentAction:
            if action == AgentAction.WAIT:
                continue
            nx, ny = self._target_cell(action)
            if self._cell_safety_score(nx, ny) < 1:
                continue
            dist = abs(nx - tx) + abs(ny - ty)
            candidates.append((dist, action))
        if not candidates:
            return None
        candidates.sort(key=lambda c: c[0])
        return candidates[0][1]
