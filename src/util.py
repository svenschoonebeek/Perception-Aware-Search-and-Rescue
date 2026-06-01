from matplotlib import ticker
import numpy as np
import logging
from numba import njit
import time
import matplotlib.pyplot as plt

from src.metric import MetricInTime

logger = logging.getLogger(__name__)

def setup_logging(is_debug: bool):
    """Configures the global logging level and format."""
    log_level = logging.DEBUG if is_debug else logging.INFO

    logging.basicConfig(
        level=log_level,
        format='%(levelname)s - %(message)s',
    )
    # prevent debug spew
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

@njit(debug=False, cache=True)
def fast_bfs_array(start_x, start_y, target_x, target_y, traversability, victims, fire, untraversable_val, victim_present_val, fire_burning_val):
    height, width = traversability.shape
    
    if start_x == target_x and start_y == target_y:
        return np.array([[start_x, start_y]], dtype=np.int16)

    max_queue_size = height * width
    queue_x = np.zeros(max_queue_size, dtype=np.int16)
    queue_y = np.zeros(max_queue_size, dtype=np.int16)
    
    visited = np.zeros((height, width), dtype=np.bool_)
    parent_x = np.full((height, width), -1, dtype=np.int16)
    parent_y = np.full((height, width), -1, dtype=np.int16)
    
    head, tail = 0, 0
    queue_x[tail] = start_x
    queue_y[tail] = start_y
    tail += 1
    visited[start_y, start_x] = True
    
    dx = np.array([0, 0, -1, 1], dtype=np.int16)
    dy = np.array([-1, 1, 0, 0], dtype=np.int16)
    
    found = False
    while head < tail:
        cx, cy = queue_x[head], queue_y[head]
        head += 1
        
        if cx == target_x and cy == target_y:
            found = True
            break
            
        for i in range(4):
            nx, ny = cx + dx[i], cy + dy[i]
            if 0 <= nx < width and 0 <= ny < height:
                if not visited[ny, nx]:
                    if (traversability[ny, nx] != untraversable_val and 
                        victims[ny, nx] != victim_present_val and 
                        fire[ny, nx] != fire_burning_val):
                        
                        visited[ny, nx] = True
                        parent_x[ny, nx] = cx
                        parent_y[ny, nx] = cy
                        queue_x[tail] = nx
                        queue_y[tail] = ny
                        tail += 1
                        
    if not found:
        return np.zeros((0, 2), dtype=np.int16)
        
    path_len = 0
    curr_x, curr_y = target_x, target_y
    while curr_x != -1 and curr_y != -1:
        path_len += 1
        curr_x, curr_y = parent_x[curr_y, curr_x], parent_y[curr_y, curr_x]
        
    path = np.zeros((path_len, 2), dtype=np.int16)
    curr_x, curr_y = target_x, target_y
    for i in range(path_len - 1, -1, -1):
        path[i, 0] = curr_x
        path[i, 1] = curr_y
        curr_x, curr_y = parent_x[curr_y, curr_x], parent_y[curr_y, curr_x]
        
    return path

def warmup_numba_compilers():
    """
    Forces Numba to compile JIT functions before the simulation starts.
    """
    logger.info("Warming up Numba JIT compilers...")
    start_time = time.perf_counter()
    dummy_matrix = np.zeros((2, 2), dtype=np.int8)
    
    fast_bfs_array(
        start_x=0, start_y=0, 
        target_x=1, target_y=1, 
        traversability=dummy_matrix, 
        victims=dummy_matrix, 
        fire=dummy_matrix, 
        untraversable_val=1,  # Dummy ints
        victim_present_val=1,
        fire_burning_val=1   
    )
    
    elapsed = time.perf_counter() - start_time
    logger.info(f"Numba warmup complete. Compilation took {elapsed:.2f}s.")

def plot_total_exploration(metric_history: list[MetricInTime]):
    if not metric_history:
        print("No metric history to plot.")
        return

    time_steps = [m.step for m in metric_history]
    total_explored = [m.total_explored for m in metric_history]

    plt.figure(figsize=(10, 6))

    agents = list(metric_history[0].area_explored.keys())
    for agent in agents:
        agent_explored = [m.area_explored.get(agent, 0.0) for m in metric_history]
        agent_name = getattr(agent, 'name', f"Agent {id(agent)}") 
        plt.plot(time_steps, agent_explored, linestyle='--', alpha=0.4, label=f'{agent_name}')

    plt.plot(time_steps, total_explored, color='blue', linewidth=2.5, label='Total Swarm Exploration')
    plt.fill_between(time_steps, total_explored, color='blue', alpha=0.05)

    full_exploration_step = next((m.step for m in metric_history if m.total_explored >= 0.999), None)
    if full_exploration_step is not None:
        plt.axvline(x=full_exploration_step, color='green', linestyle=':', linewidth=2)
        plt.text(full_exploration_step, 0.05, f' 100% Explored\n (Step {full_exploration_step})', 
                 color='green', verticalalignment='bottom')

    final_metric = metric_history[-1]
    
    if final_metric.time_to_all_found is not None:
        t_all = final_metric.time_to_all_found        
        plt.axvline(x=t_all, color='red', linestyle='-.', linewidth=1.5, alpha=0.7)
        plt.text(t_all, 0.05, f' All Victims Found\n (Step {final_metric.time_to_all_found})', color='red')

    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Area Explored", fontsize=12)
    plt.title("Swarm Exploration Progress & Milestones", fontsize=14, fontweight='bold')
    
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    plt.ylim(0, 1.05) 
    plt.xlim(0, max(time_steps) if time_steps else 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    
    plt.tight_layout()

    plt.show()