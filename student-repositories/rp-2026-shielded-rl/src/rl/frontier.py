"""
Utility function to compute the BFS distance from the agent to the nearest
unexplored region. 
This reuses sub-project 1's MPC idea, but reimplemented to adapt to the agents belief
state so that the RL policy can read it as an observation channel.
"""
from collections import deque

import numpy as np


def nearest_frontier_distance(
    explored: np.ndarray, traversable: np.ndarray, y: int, x: int
) -> int:
    """Hops from (y, x) to the nearest explored cell that borders an unexplored cell.
    Uses belief state traversability to only hop through cells the agent believes are traversable.
    """
    h, w = explored.shape
    visited = np.zeros((h, w), dtype=bool)
    visited[y, x] = True
    q: deque = deque(((y, x, 0),))
    while q:
        cy, cx, d = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if not explored[ny, nx]: # checks if any of the neighbors of (cy, cx) are unexplored, if so we have found the frontier and can return the distance.
                return d
            if not visited[ny, nx] and explored[ny, nx] and traversable[ny, nx]: # else if the neighbor is explored and traversable, we can add it to the queue to continue the search.
                visited[ny, nx] = True
                q.append((ny, nx, d + 1))
    return 0


def frontier_distance_field(explored: np.ndarray, traversable: np.ndarray) -> np.ndarray:
    """Normalised [0,1] BFS distance to nearest unexplored field from belief."""
    h, w = explored.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    q: deque = deque()
    ys, xs = np.nonzero(~explored)              # unexplored cells are the BFS sources
    for y, x in zip(ys.tolist(), xs.tolist()):
        dist[y, x] = 0
        q.append((y, x))
    while q:                                    # expand through explored, traversable cells
        cy, cx = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if (0 <= ny < h and 0 <= nx < w and dist[ny, nx] == -1
                    and explored[ny, nx] and traversable[ny, nx]):
                dist[ny, nx] = dist[cy, cx] + 1
                q.append((ny, nx))
    field = np.where(dist < 0, h + w, dist).astype(np.float32) / (h + w)
    return np.clip(field, 0.0, 1.0)
