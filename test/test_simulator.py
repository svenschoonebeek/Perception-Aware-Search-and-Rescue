import unittest

from src.simulator import *

class SimulatorTest(unittest.TestCase):
    def test_visualize_grid_gen(self):
        simulator = Simulator(100, 100)
        config = MapConfig(num_rooms=7,
                           unconnected_probability=0.0,
                           room_vulnerability_probability=0.0,
                           tunnel_vulnerability_probability=0.0,
                           initial_fire_points=3,
                           fire_spread_rate=0.3,
                           fire_duration=-1,
                           start_room_width=5,
                           start_room_length=5,
                           min_room_width=10,
                           max_room_width=20,
                           min_room_length=10,
                           max_room_length=20,
                           min_tunnel_thickness=4,
                           max_tunnel_thickness=6,
                           num_victims=5)
        simulator.generate_ground_truth(config)
        state_after_running = simulator.run(10)[0][-1]
        # visualize_grid_gen(simulator.ground_truth.traversability, simulator.ground_truth.agents, 
        #                    simulator.ground_truth.victims, simulator.ground_truth.vulnerability,
        #                    simulator.ground_truth.fire)
        visualize_grid_gen(state_after_running.traversability, state_after_running.agents, 
                           state_after_running.victims, state_after_running.vulnerability,
                           state_after_running.fire)
    
    def test_add_agent(self):
        sim = Simulator(5, 5)
        bob = Agent("Bob", 0, 0, 5, 5, sigma=0.05, scan_accuracy=0.9, scan_radius=0)
        sim.add_agent(bob)

        self.assertEqual(len(sim.agents), 1)
        self.assertEqual(sim.agents[0], bob)
        self.assertEqual(sim.ground_truth.agents[0][0], 1)

    def test_step(self):
        sim = Simulator(4, 3)
        bob = Agent("Bob", 0, 0, 4, 3, sigma=0.05, scan_accuracy=0.9, scan_radius=0)
        alice = Agent("Alice", 1, 1, 4, 3, sigma=0.05, scan_accuracy=0.9, scan_radius=0)
        sim.add_agent(bob)
        sim.add_agent(alice)

        new_state = sim.step()
        self.assertEqual(len(new_state), 3) # Ground truth + bob + alice

        new_ground = new_state[0]
        new_bob = new_state[1]
        new_alice = new_state[2]

        # Ground truth test
        self.assertEqual(new_ground.agents[0][0], 0)
        self.assertEqual(new_ground.agents[0][1], 1)
        self.assertEqual(new_ground.agents[1][1], 0)
        self.assertEqual(new_ground.agents[1][2], 1)
        # Bob test
        self.assertEqual(new_bob.agents[0][0], 0)
        self.assertEqual(new_bob.agents[0][1], 1)
        # Alice test
        self.assertEqual(new_alice.agents[1][1], 0)
        self.assertEqual(new_alice.agents[1][2], 1)

    def test_run(self):
        sim = Simulator(4, 3)
        bob = Agent("Bob", 0, 0, 4, 3, sigma=0.05, scan_accuracy=0.9, scan_radius=0)
        sim.add_agent(bob)

        steps = sim.run(3)
        # Steps should have 2 lists, [ground_truth, bob]
        self.assertEqual(len(steps), 2)
        # Each list should record 4 states, [initial (t_0), t_1, t_2, t_3]
        self.assertEqual(len(steps[0]), 4)
        self.assertEqual(len(steps[1]), 4)

        # Ground truth states properly track bob
        self.assertEqual(steps[0][0].agents[0, 0], 1)
        self.assertEqual(steps[0][1].agents[0, 0], 0)
        self.assertEqual(steps[0][1].agents[0, 1], 1)
        self.assertEqual(steps[0][2].agents[0, 1], 0)
        self.assertEqual(steps[0][2].agents[0, 2], 1)
        self.assertEqual(steps[0][3].agents[0, 2], 0)
        self.assertEqual(steps[0][3].agents[0, 3], 1)

        # Bob states properly track himself
        self.assertEqual(steps[1][0].agents[0, 0], 1)
        self.assertEqual(steps[1][1].agents[0, 0], 0)
        self.assertEqual(steps[1][1].agents[0, 1], 1)
        self.assertEqual(steps[1][2].agents[0, 1], 0)
        self.assertEqual(steps[1][2].agents[0, 2], 1)
        self.assertEqual(steps[1][3].agents[0, 2], 0)
        self.assertEqual(steps[1][3].agents[0, 3], 1)