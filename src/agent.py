import numpy as np

from .state import State
from .constants import *

class Agent:
    def __init__(self, name: str, x: int, y: int, width: int, height: int, 
                 sigma: float, scan_accuracy: float, scan_radius: int):
        self.name = name
        self.perception = State(width, height)
        self.x = x
        self.y = y
        self.perception.agents[y][x] = 1
        self.world_width = width
        self.world_height = height

        self.sigma = sigma # Certainty decay per time step [0,1]
        self.scan_accuracy = scan_accuracy # Scan accuracy [0,1]
        self.scan_radius = scan_radius # How far the agent can see when it scans

        self.move_history: list[tuple[int, int]] = [(x, y)] # Maintain a log of what positions the agent has been in for stats

        self.illegal_moves = 0

    def scan(self, state: State) -> list[Tile]:
        """
        Retrieves information from the state passed in (with some noise/uncertainties?)

        Parameters:
        state: The true world to scan from
        """
        self.perception.confidence.matrix = np.maximum(self.perception.confidence.matrix - self.sigma, 0) # vectorized confidence decay

        min_x = max(0, self.x - self.scan_radius)
        max_x = min(self.world_width, self.x + self.scan_radius + 1)
        min_y = max(0, self.y - self.scan_radius)
        max_y = min(self.world_height, self.y + self.scan_radius + 1)

        scanned_tiles = []

        for i in range(min_y, max_y):
            for j in range(min_x, max_x):
                if self._tile_scanned(i, j):
                    self.perception.confidence[i][j] = max(self.perception.confidence[i][j] - self.sigma, self.scan_accuracy)
                    self.perception.traversability[i][j] = state.traversability[i][j]
                    self.perception.vulnerability[i][j] = state.vulnerability[i][j]
                    self.perception.fire[i][j] = state.fire[i][j]
                    self.perception.victims[i][j] = state.victims[i][j]
                    self.perception.agents[i][j] = state.agents[i][j]
                    scanned_tiles.append(Tile(j, i, 
                                              state.traversability[i][j],
                                              state.vulnerability[i][j],
                                              state.fire[i][j],
                                              state.victims[i][j],
                                              state.agents[i][j],
                                              self.perception.confidence[i][j]))
        
        return scanned_tiles


    def move(self, direction: AgentAction) -> None:
        """
        Updates an agent's internal position based on an action

        Parameters:
        direction: The action to perform
        """

        target_x, target_y = self.x, self.y

        match direction:
            case AgentAction.MOVE_UP:
                target_y -= 1
            case AgentAction.MOVE_DOWN:
                target_y += 1
            case AgentAction.MOVE_LEFT:
                target_x -= 1
            case AgentAction.MOVE_RIGHT:
                target_x += 1
            case _:
                return

        if not (0 <= target_x < self.world_width and 0 <= target_y < self.world_height):
            self.illegal_moves += 1
            return

        self.perception.agents[self.y][self.x] = 0
        self.x, self.y = target_x, target_y
        self.move_history.append((self.x, self.y))
        self.perception.agents[self.y][self.x] = 1

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

    def _tile_scanned(self, row: int, col: int) -> bool:
        """
        Checks if a tile is within the scan radius of the agent
        """
        # return abs(row - self.y) + abs(col - self.x) <= self.scan_radius # manhattan distance
        return abs(row - self.y) ** 2 + abs(col - self.x) ** 2 <= self.scan_radius ** 2 # euclidean distance
