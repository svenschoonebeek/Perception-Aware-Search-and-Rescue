from .simulator import Simulator, MapConfig, visualize_grid_gen
from .agents.mpc import MpcAgent
from .agents.random import RandAgent
from .agent import Agent
from .visualization import Visualizer

WIDTH = 40
HEIGHT = 30

def main():
    sim = Simulator(WIDTH, HEIGHT)
    config = MapConfig(num_rooms=6, num_victims=5, min_room_length=4, min_room_width=4, max_room_length=7, max_room_width=7, max_tunnel_thickness=1)
    sim.generate_ground_truth(config)

    # seed: 197357699 
    # Saved seeds: 182840517 (181 steps); 

    agent1 = MpcAgent("mpc", 0, 0, WIDTH, HEIGHT, 0.01, 0.9, 3, True)
    # agent1 = RandAgent("randy", 0, 0, WIDTH, HEIGHT, 0.01, 0.9, 7, False)
    # agent1 = Agent("mpc", 0, 1, WIDTH, HEIGHT, 0.05, 0.9, 4, False)
    sim.add_agent(agent1)

    visualize_grid_gen(sim.ground_truth.traversability, sim.ground_truth.agents, 
                       sim.ground_truth.victims, sim.ground_truth.vulnerability,
                       sim.ground_truth.fire)

    print("Running Simulation steps...")
    history = sim.run(700) 

    print("Launching Visualizer...")
    viz = Visualizer(history, 1000, 1000)
    viz.run()

if __name__ == "__main__":
    main()