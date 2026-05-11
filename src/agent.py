import numpy as np

from .state import State
from .constants import AgentAction, Scan
import numpy as np

from numpy.typing import NDArray

from tcod.map import compute_fov
from tcod.libtcodpy import FOV_RESTRICTIVE

# Threshold for what value of traversability causes a tile to become 'opaque' to the LoS check
LOS_THRESHOLD = 1

class Agent:
    def __init__(self, name: str, x: int, y: int, width: int, height: int, 
                 decay: float, scan_accuracy: float, scan_radius: int, scan_falloff: bool):
        self.name = name
        self.perception = State(width, height)
        self.x = x
        self.y = y
        self.perception.agents[y][x] = 1
        self.world_width = width
        self.world_height = height

        self.decay = decay # Certainty decay per time step [0,1]
        self.scan_accuracy = scan_accuracy # Scan accuracy [0,1]
        self.scan_radius = scan_radius # How far the agent can see when it scans
        self.scan_falloff = scan_falloff # Whether scan accuracy should decrease with distance

        self.move_history: list[tuple[int, int]] = [(x, y)] # Maintain a log of what positions the agent has been in for stats

        self.illegal_moves = 0

        self.explored: np.ndarray = np.zeros((height, width), dtype=bool) # Cells the agent has ever scanned

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