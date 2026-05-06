from .simulator import Simulator, MapConfig, visualize_grid_gen
from .agent import Agent
from .visualization import Visualizer

WIDTH = 40
HEIGHT = 25

def main():
    sim = Simulator(WIDTH, HEIGHT)
    config = MapConfig(num_rooms=5, num_victims=5)
    sim.generate_ground_truth(config)

   #  visualize_grid_gen(sim.ground_truth.traversability, sim.ground_truth.agents, sim.ground_truth.victims)

    bob = Agent("Bob", 0, 0, WIDTH, HEIGHT, 0.01, 0.9, 4)
    sim.add_agent(bob)
    bob1 = Agent("Bob1", 0, 1, WIDTH, HEIGHT, 0.01, 0.9, 4)
    sim.add_agent(bob1)
    bob2 = Agent("Bob2", 0, 2, WIDTH, HEIGHT, 0.01, 0.9, 4)
    sim.add_agent(bob2)
    bob3 = Agent("Bob3", 0, 3, WIDTH, HEIGHT, 0.01, 0.9, 4)
    sim.add_agent(bob3)
    bob4 = Agent("Bob4", 0, 4, WIDTH, HEIGHT, 0.01, 0.9, 4)
    sim.add_agent(bob4)
    bob5 = Agent("Bob5", 0, 5, WIDTH, HEIGHT, 0.01, 0.9, 4)
    sim.add_agent(bob5)
    bob6 = Agent("Bob6", 0, 6, WIDTH, HEIGHT, 0.01, 0.9, 4)
    sim.add_agent(bob6)

    print("Running Simulation steps...")
    history = sim.run(30) 

    print("Launching Visualizer...")
    viz = Visualizer(history, cell_size=25)
    viz.run()

if __name__ == "__main__":
    main()