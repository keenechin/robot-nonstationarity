from actuonix_driver import LinearDriver
from dynamixel_driver import DynamixelDriver
import numpy as np
import time


class FiveBar():
    def __init__(self):
        self.servos = DynamixelDriver(num_servos=2)
        self.linear = LinearDriver()
        self.linear.reset()
        self.servos.servo_move([2048, 2048])
        time.sleep(0.5)
        self.linear_min = 0.0012
        self.linear_max = 0.05
        self.theta1_min = 0.375
        self.theta1_max = 0.625
        self.theta2_min = 0.375
        self.theta2_max = 0.625

    def drift(self, pos):
        assert pos >= self.linear_min
        assert pos <= self.linear_max
        p = np.ones((1, 12))*0.0012
        p[0, 3:7] = pos
        self.linear.move_joint_position(p, 1.0)
        time.sleep(0.01)
        # self.linear.wait_until_done_moving()

    def move(self, pos1, pos2):
        assert pos1 >= self.theta1_min
        assert pos2 >= self.theta2_min
        assert pos1 <= self.theta1_max
        assert pos2 <= self.theta2_max
        self.servos.servo_move([pos1, pos2])

    def primitive(self, idx):
        pass


if __name__ == "__main__":
    robot = FiveBar()
    robot.drift(0.03)
    robot.move(1024, 2560)
    robot.drift(0.01)
    robot.move(2560, 1024)
    robot.drift(robot.linear_max)
    robot.move(2048, 2048)





