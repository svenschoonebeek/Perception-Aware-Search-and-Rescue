import unittest

from src.simulator import Simulator
from src.constants import *
from src.agent import Agent, AgentAction
from src.metric import RunOutcome


def _make_agent(name, x, y, w, h, action, scan_radius=0):
    """Returns an Agent subclass that always picks `action`."""
    class FixedAgent(Agent):
        def get_action(self):
            return action
    return FixedAgent(name, x, y, w, h, decay=0.05, scan_accuracy=0.9, scan_radius=scan_radius, scan_falloff=False)


class TestWaitMetric(unittest.TestCase):
    def test_wait_records_no_collision(self):
        sim = Simulator(3, 3)
        a = _make_agent("a", 1, 1, 3, 3, AgentAction.WAIT)
        sim.add_agent(a)
        sim.run(2)

        self.assertEqual(sim.metrics.wait_actions[a], 2)
        self.assertEqual(sim.metrics.terrain_collisions[a], 0)
        self.assertEqual(sim.metrics.damage[a], 0)
        self.assertEqual((a.x, a.y), (1, 1))


class TestTerrainCollision(unittest.TestCase):
    def test_out_of_bounds_counts_as_terrain(self):
        sim = Simulator(3, 3)
        a = _make_agent("a", 0, 0, 3, 3, AgentAction.MOVE_LEFT)
        sim.add_agent(a)
        sim.run(3)

        self.assertEqual(sim.metrics.terrain_collisions[a], 3)
        self.assertEqual(sim.metrics.damage[a], 3)
        self.assertEqual((a.x, a.y), (0, 0))

    def test_wall_counts_as_terrain(self):
        sim = Simulator(3, 3)
        sim.ground_truth.traversability[0][1] = TraversabilityLevel.UNTRAVERSIBLE
        a = _make_agent("a", 0, 0, 3, 3, AgentAction.MOVE_RIGHT)
        sim.add_agent(a)
        sim.run(2)

        self.assertEqual(sim.metrics.terrain_collisions[a], 2)
        self.assertEqual((a.x, a.y), (0, 0))


class TestVictimCollision(unittest.TestCase):
    def test_victim_acts_as_obstacle(self):
        sim = Simulator(3, 3)
        sim.ground_truth.victims[0][1] = VictimPresence.PRESENT
        a = _make_agent("a", 0, 0, 3, 3, AgentAction.MOVE_RIGHT)
        sim.add_agent(a)
        sim.run(2)

        self.assertEqual(sim.metrics.victim_collisions[a], 2)
        self.assertEqual(sim.metrics.damage[a], 2)
        self.assertEqual((a.x, a.y), (0, 0))
        # Victim still on the map — not killed.
        self.assertEqual(sim.ground_truth.victims[0][1], VictimPresence.PRESENT)


class TestInterAgentCollision(unittest.TestCase):
    def test_two_movers_same_target(self):
        # A at (0,0) wants right -> (1,0). B at (2,0) wants left -> (1,0).
        sim = Simulator(3, 1)
        a = _make_agent("a", 0, 0, 3, 1, AgentAction.MOVE_RIGHT)
        b = _make_agent("b", 2, 0, 3, 1, AgentAction.MOVE_LEFT)
        sim.add_agent(a)
        sim.add_agent(b)
        sim.step()

        self.assertEqual(sim.metrics.inter_agent_collisions[a], 1)
        self.assertEqual(sim.metrics.inter_agent_collisions[b], 1)
        self.assertEqual((a.x, a.y), (0, 0))
        self.assertEqual((b.x, b.y), (2, 0))

    def test_three_movers_same_target_each_collides_once(self):
        # A,B,C all want (1,1).
        sim = Simulator(3, 3)
        a = _make_agent("a", 0, 1, 3, 3, AgentAction.MOVE_RIGHT)
        b = _make_agent("b", 2, 1, 3, 3, AgentAction.MOVE_LEFT)
        c = _make_agent("c", 1, 0, 3, 3, AgentAction.MOVE_DOWN)
        sim.add_agent(a)
        sim.add_agent(b)
        sim.add_agent(c)
        sim.step()

        self.assertEqual(sim.metrics.inter_agent_collisions[a], 1)
        self.assertEqual(sim.metrics.inter_agent_collisions[b], 1)
        self.assertEqual(sim.metrics.inter_agent_collisions[c], 1)

    def test_swap_pair(self):
        sim = Simulator(3, 1)
        a = _make_agent("a", 0, 0, 3, 1, AgentAction.MOVE_RIGHT)
        b = _make_agent("b", 1, 0, 3, 1, AgentAction.MOVE_LEFT)
        sim.add_agent(a)
        sim.add_agent(b)
        sim.step()

        self.assertEqual(sim.metrics.inter_agent_collisions[a], 1)
        self.assertEqual(sim.metrics.inter_agent_collisions[b], 1)
        self.assertEqual((a.x, a.y), (0, 0))
        self.assertEqual((b.x, b.y), (1, 0))

    def test_mover_into_stayer_both_collide(self):
        sim = Simulator(3, 1)
        a = _make_agent("a", 0, 0, 3, 1, AgentAction.MOVE_RIGHT)
        b = _make_agent("b", 1, 0, 3, 1, AgentAction.WAIT)
        sim.add_agent(a)
        sim.add_agent(b)
        sim.step()

        self.assertEqual(sim.metrics.inter_agent_collisions[a], 1)
        self.assertEqual(sim.metrics.inter_agent_collisions[b], 1)
        self.assertEqual(sim.metrics.wait_actions[b], 1)
        self.assertEqual((a.x, a.y), (0, 0))

    def test_chase_succeeds_when_leader_moves_away(self):
        # A at (0,0) wants right -> (1,0). B at (1,0) wants right -> (2,0).
        # B leaves (1,0), so A can take it. No collisions.
        sim = Simulator(3, 1)
        a = _make_agent("a", 0, 0, 3, 1, AgentAction.MOVE_RIGHT)
        b = _make_agent("b", 1, 0, 3, 1, AgentAction.MOVE_RIGHT)
        sim.add_agent(a)
        sim.add_agent(b)
        sim.step()

        self.assertEqual(sim.metrics.inter_agent_collisions[a], 0)
        self.assertEqual(sim.metrics.inter_agent_collisions[b], 0)
        self.assertEqual((a.x, a.y), (1, 0))
        self.assertEqual((b.x, b.y), (2, 0))


class TestVictimsFound(unittest.TestCase):
    def test_victim_found_via_scan_triggers_success(self):
        sim = Simulator(3, 3)
        sim.ground_truth.victims[1][2] = VictimPresence.PRESENT
        # Scan radius 2 covers (1,2) from (1,1).
        a = _make_agent("a", 1, 1, 3, 3, AgentAction.WAIT, scan_radius=2)
        sim.add_agent(a)
        sim.run(5)

        self.assertEqual(sim.metrics.total_victims, 1)
        self.assertEqual(sim.metrics.victims_found, 1)
        self.assertEqual(sim.metrics.time_to_first_found, 1)
        self.assertEqual(sim.metrics.time_to_all_found, 1)
        self.assertEqual(sim.metrics.outcome, RunOutcome.SUCCESS)
        # Run stopped early: initial state + step 1 = 2 entries.
        self.assertEqual(sim.metrics.steps_taken, 1)

    def test_no_victims_results_in_timeout(self):
        sim = Simulator(3, 3)
        a = _make_agent("a", 1, 1, 3, 3, AgentAction.WAIT)
        sim.add_agent(a)
        sim.run(3)

        self.assertEqual(sim.metrics.total_victims, 0)
        self.assertEqual(sim.metrics.outcome, RunOutcome.TIMEOUT)
        self.assertIsNone(sim.metrics.time_to_first_found)
        self.assertIsNone(sim.metrics.time_to_all_found)

    def test_partial_find_does_not_trigger_success(self):
        sim = Simulator(5, 5)
        sim.ground_truth.victims[0][0] = VictimPresence.PRESENT  # found
        sim.ground_truth.victims[4][4] = VictimPresence.PRESENT  # never seen
        a = _make_agent("a", 0, 0, 5, 5, AgentAction.WAIT, scan_radius=1)
        sim.add_agent(a)
        sim.run(3)

        self.assertEqual(sim.metrics.total_victims, 2)
        self.assertEqual(sim.metrics.victims_found, 1)
        self.assertEqual(sim.metrics.time_to_first_found, 1)
        self.assertIsNone(sim.metrics.time_to_all_found)
        self.assertEqual(sim.metrics.outcome, RunOutcome.TIMEOUT)


class TestAreaExplored(unittest.TestCase):
    def test_static_agent_explores_self_cell_only(self):
        sim = Simulator(3, 3)
        a = _make_agent("a", 1, 1, 3, 3, AgentAction.WAIT, scan_radius=0)
        sim.add_agent(a)
        sim.run(1)

        self.assertEqual(sim.metrics.total_traversable, 9)
        self.assertAlmostEqual(sim.metrics.area_explored[a], 1 / 9)

    def test_walking_increases_coverage(self):
        # Movement happens before scan, so the starting cell is never observed.
        # After 2 right-steps from (0,0): scans at (1,0) then (2,0). 2/3 covered.
        sim = Simulator(3, 1)
        a = _make_agent("a", 0, 0, 3, 1, AgentAction.MOVE_RIGHT, scan_radius=0)
        sim.add_agent(a)
        sim.run(2)

        self.assertAlmostEqual(sim.metrics.area_explored[a], 2 / 3)

    def test_walls_excluded_from_denominator(self):
        sim = Simulator(3, 3)
        sim.ground_truth.traversability[2][2] = TraversabilityLevel.UNTRAVERSIBLE
        a = _make_agent("a", 1, 1, 3, 3, AgentAction.WAIT, scan_radius=10)
        sim.add_agent(a)
        sim.run(1)

        self.assertEqual(sim.metrics.total_traversable, 8)
        self.assertAlmostEqual(sim.metrics.area_explored[a], 1.0)

    def test_per_agent_independent(self):
        sim = Simulator(3, 1)
        a = _make_agent("a", 0, 0, 3, 1, AgentAction.WAIT, scan_radius=0)
        b = _make_agent("b", 2, 0, 3, 1, AgentAction.WAIT, scan_radius=0)
        sim.add_agent(a)
        sim.add_agent(b)
        sim.run(1)

        # Each agent only scans its own cell — no shared union.
        self.assertAlmostEqual(sim.metrics.area_explored[a], 1 / 3)
        self.assertAlmostEqual(sim.metrics.area_explored[b], 1 / 3)


class TestMetricHistory(unittest.TestCase):
    def test_history_length_initial_plus_each_step(self):
        sim = Simulator(3, 3)
        a = _make_agent("a", 1, 1, 3, 3, AgentAction.WAIT)
        sim.add_agent(a)
        sim.run(3)

        # No victims -> runs full 3 steps then TIMEOUT. Initial + 3 = 4 snapshots.
        self.assertEqual(len(sim.metrics.history), 4)
        self.assertEqual(sim.metrics.history[0].step, 0)
        self.assertEqual(sim.metrics.history[3].step, 3)

    def test_initial_snapshot_has_zero_counters(self):
        sim = Simulator(3, 3)
        a = _make_agent("a", 1, 1, 3, 3, AgentAction.WAIT)
        sim.add_agent(a)
        sim.step()

        snap0 = sim.metrics.history[0]
        self.assertEqual(snap0.step, 0)
        self.assertEqual(snap0.victims_found, 0)
        self.assertEqual(snap0.wait_actions[a], 0)
        self.assertEqual(snap0.area_explored[a], 0.0)
        self.assertEqual(snap0.outcome, RunOutcome.IN_PROGRESS)

    def test_history_records_per_step_progression(self):
        sim = Simulator(3, 3)
        a = _make_agent("a", 1, 1, 3, 3, AgentAction.WAIT)
        sim.add_agent(a)
        sim.run(2)

        self.assertEqual(sim.metrics.history[0].wait_actions[a], 0)
        self.assertEqual(sim.metrics.history[1].wait_actions[a], 1)
        self.assertEqual(sim.metrics.history[2].wait_actions[a], 2)

    def test_snapshot_isolated_from_subsequent_mutations(self):
        sim = Simulator(3, 3)
        a = _make_agent("a", 1, 1, 3, 3, AgentAction.WAIT)
        sim.add_agent(a)
        sim.step()

        snap0 = sim.metrics.history[0]
        sim.metrics.wait_actions[a] = 999
        self.assertEqual(snap0.wait_actions[a], 0)

    def test_timeout_reflected_in_last_snapshot(self):
        sim = Simulator(3, 3)
        a = _make_agent("a", 1, 1, 3, 3, AgentAction.WAIT)
        sim.add_agent(a)
        sim.run(2)

        self.assertEqual(sim.metrics.outcome, RunOutcome.TIMEOUT)
        self.assertEqual(sim.metrics.history[-1].outcome, RunOutcome.TIMEOUT)

    def test_success_reflected_in_last_snapshot(self):
        sim = Simulator(3, 3)
        sim.ground_truth.victims[1][2] = VictimPresence.PRESENT
        a = _make_agent("a", 1, 1, 3, 3, AgentAction.WAIT, scan_radius=2)
        sim.add_agent(a)
        sim.run(5)

        # Found in step 1, run breaks. Snapshots: [initial, after-step-1].
        self.assertEqual(len(sim.metrics.history), 2)
        self.assertEqual(sim.metrics.history[0].outcome, RunOutcome.IN_PROGRESS)
        self.assertEqual(sim.metrics.history[1].outcome, RunOutcome.SUCCESS)


if __name__ == "__main__":
    unittest.main()
