import numpy as np

class Grid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.matrix = np.zeros((height, width))

    def __getitem__(self, key):
        return self.matrix[key]
    
    def __setitem__(self, key, value):
        self.matrix[key] = value

    def __delitem__(self, key):
        self.matrix[key] = 0

    def copy(self) -> "Grid":
        copy = Grid.__new__(Grid)

        copy.height = self.height
        copy.width = self.width
        copy.matrix = np.copy(self.matrix)

        return copy