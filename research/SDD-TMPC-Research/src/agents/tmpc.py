from .mpc import MpcAgent
from ..constants import AgentAction, FireLevel

class TmpcAgent(MpcAgent):
    def __init__(self, name: str, x: int, y: int, width: int, height: int, 
                 decay: float, scan_accuracy: float, scan_radius: int, scan_falloff: bool):
        super().__init__(name, x, y, width, height, decay, scan_accuracy, scan_radius, scan_falloff)
        self.fire_spread_rate = 1.0
    
    def get_action(self) -> AgentAction:
        self.tube_size += 1
        return super().get_action()

    def _is_feasible(self) -> bool:
        """
        Checks whether a given state violates constraints (e.g. not walking into a wall)
        Assumes given state (centre of the tube) is in the admissible set X
        For TMPC, this now also checks whether tripping from this position results violates constraints

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