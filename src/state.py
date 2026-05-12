from .grid import Grid

class State:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.traversability = Grid(width=width, height=height)
        self.vulnerability = Grid(width=width, height=height)
        self.fire = Grid(width=width, height=height)
        self.victims = Grid(width=width, height=height)
        self.agents = Grid(width=width, height=height)
        self.confidence = Grid(width=width, height=height)

    def copy(self) -> "State":
        copy = State.__new__(State)

        copy.width = self.width
        copy.height = self.height
        copy.traversability = self.traversability.copy()
        copy.vulnerability = self.vulnerability.copy()
        copy.victims = self.victims.copy()
        copy.agents = self.agents.copy()
        copy.fire = self.fire.copy()
        copy.confidence = self.confidence.copy()
         
        return copy