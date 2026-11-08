from .mpc import MpcAgent
from ..constants import AgentAction, FireLevel

class TmpcAgent(MpcAgent):
    def __init__(self, name: str, x: int, y: int, width: int, height: int, 
                 decay: float, scan_accuracy: float, scan_radius: int, scan_falloff: bool):
        super().__init__(name, x, y, width, height, decay, scan_accuracy, scan_radius, scan_falloff)
        self.fire_spread_rate = 1.0

    def _is_feasible(self, target_cell: tuple[int, int]) -> bool:
        """
        Decides whether a given action will result in a feasible position
        For TMPC, this now also checks whether tripping after this move results in an infeasible state

        Parameters:
        action: The action to check

        Returns:
        Whether the action is feasible
        """

        def cell_feasible(x: int, y: int) -> bool:
            out_of_bounds = not(0 <= x < self.world_width and 0 <= y < self.world_height)
            if out_of_bounds: return False
            wall = self.perception.traversability[y][x] == 1
            victim = self.perception.victims[y][x] == 1
            fire = self.perception.fire[y][x] == FireLevel.BURNING
            return not(wall or victim or fire)

        target_cell_x, target_cell_y = target_cell
        if not cell_feasible(target_cell_x, target_cell_y):
            return False

        # Check if tripping after this move results in an infeasible state
        # [up, down, left, right]
        dy = [-1, 1, 0, 0]
        dx = [0, 0, -1, 1]
        for i in range(4):
            new_x, new_y = target_cell_x + dx[i], target_cell_y + dy[i]
            if not cell_feasible(new_x, new_y):
                return False
            
        return True