from enum import Enum
from typing import Self

import numpy as np

from ..agent import Agent
from ..constants import AgentAction, FireLevel, VulnerabilityLevel, TraversabilityLevel, VictimPresence

EXIT_X, EXIT_Y = 0, 0

# Speed as move-probability per step, derived from paper's m/s ranges (Table 2)
_BASE_MOVE_PROB = {
    "child":   0.60,
    "adult":   0.90,
    "elderly": 0.70,
}
_DANGER_SPEED_MULTIPLIER = 1.5  # panic boost; capped at 1.0


class Personality(Enum):
    ALTRUIST = "altruist"   # wanders to help; only flees when in danger
    SELFLESS = "selfless"   # like altruist but mimics others after conflict (future)
    EGOIST   = "egoist"     # moderate; moves to exit
    SELFISH  = "selfish"    # always moves to exit as fast as possible


class AgeGroup(Enum):
    CHILD   = "child"
    ADULT   = "adult"
    ELDERLY = "elderly"


class Victim(Agent):
    def __init__(self, name: str, x: int, y: int, width: int, height: int,
                 personality: Personality = Personality.SELFLESS,
                 age_group: AgeGroup = AgeGroup.ADULT):
        super().__init__(
            name=name, x=x, y=y, width=width, height=height,
            decay=0.0,
            scan_accuracy=1.0,
            scan_radius=3,
            scan_falloff=False,
        )
        self.personality = personality
        self.age_group = age_group
        self._base_move_prob: float = _BASE_MOVE_PROB[age_group.value]

    def copy(self) -> Self:
        copy = super().copy()
        copy.personality = self.personality
        copy.age_group = self.age_group
        copy._base_move_prob = self._base_move_prob
        return copy

    # ------------------------------------------------------------------
    # BDI entry point
    # ------------------------------------------------------------------

    def get_action(self) -> AgentAction:
        in_danger = self._perceives_danger()

        # Speed: danger triggers panic → higher move probability
        move_prob = min(1.0, self._base_move_prob * (_DANGER_SPEED_MULTIPLIER if in_danger else 1.0))
        if np.random.random() > move_prob:
            return AgentAction.WAIT

        # Personality shifts selfish when danger is perceived (paper §IV-A)
        flee = in_danger or self.personality in (Personality.SELFISH, Personality.EGOIST)
        return self._move_towards_exit_safely() if flee else self._wander_or_seek()

    # ------------------------------------------------------------------
    # Belief: danger perception
    # ------------------------------------------------------------------

    def _perceives_danger(self) -> bool:
        fire = self.perception.fire.matrix
        vuln = self.perception.vulnerability.matrix

        if fire[self.y][self.x] >= FireLevel.BURNING:
            return True
        if vuln[self.y][self.x] >= VulnerabilityLevel.HIGH_RISK:
            return True

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = self.y + dy, self.x + dx
            if 0 <= ny < self.world_height and 0 <= nx < self.world_width:
                if fire[ny][nx] >= FireLevel.BURNING:
                    return True
        return False

    # ------------------------------------------------------------------
    # Cell safety scoring (higher = safer; -1 = impassable)
    # ------------------------------------------------------------------

    def _cell_safety_score(self, x: int, y: int) -> int:
        if not (0 <= x < self.world_width and 0 <= y < self.world_height):
            return -1
        if self.perception.traversability[y][x] != TraversabilityLevel.TRAVERSIBLE:
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
    # Intention: wander / seek other victims (altruist / selfless)
    # ------------------------------------------------------------------

    def _wander_or_seek(self) -> AgentAction:
        victims = self.perception.victims.matrix
        positions = np.argwhere(victims == VictimPresence.PRESENT)
        positions = positions[(positions[:, 0] != self.y) | (positions[:, 1] != self.x)]

        if len(positions) > 0:
            distances = np.linalg.norm(positions - np.array([self.y, self.x]), axis=1)
            target = positions[np.argmin(distances)]
            action = self._step_towards(int(target[1]), int(target[0]))
            if action is not None:
                return action

        safe_moves = [
            a for a in AgentAction
            if a != AgentAction.WAIT and self._cell_safety_score(*self._target_cell(a)) >= 3
        ]
        if safe_moves:
            return np.random.choice(safe_moves)

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
