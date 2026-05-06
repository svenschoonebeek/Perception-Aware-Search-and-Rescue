import unittest

from src.simulator import Simulator
from src.supervisor import Supervisor

class SupervisorTest(unittest.TestCase):
    def test_setup(self) -> None:
        self.simulator = Simulator(10, 10)
        self.supervisor = Supervisor(10, 10, 0.05)
        self.simulator.add_supervisor(self.supervisor)