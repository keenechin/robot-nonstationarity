from actuonix_driver import LinearDriver
from dynamixel_driver import DynamixelDriver
import numpy as np
import time


class FiveBar():
    def __init__(self):
        self.servos = DynamixelDriver(num_servos=2)
        self.linear = LinearDriver()
        self.linear_min = 0.01
        self.linear_zeros = np.ones((1, 12))*self.linear_min
        self.linear_max = 0.0375
        self.theta1_min = 0.375
        self.theta1_max = 0.625
        self.theta2_min = 0.375
        self.theta2_max = 0.625
        self.reset()
        time.sleep(0.5)
    
    def reset(self):
        self.linear.move_joint_position(self.linear_zeros, 1.0)
        self.move(0.5, 0.5)

    def drift(self, pos):
        assert pos >= self.linear_min
        assert pos <= self.linear_max
        p = self.linear_zeros
        p[0, 3:7] = pos
        self.linear.move_joint_position(p, 1.0)
        time.sleep(0.01)
        # self.linear.wait_until_done_moving()

    def move(self, pos1, pos2):
        assert pos1 >= self.theta1_min
        assert pos2 >= self.theta2_min
        assert pos1 <= self.theta1_max
        assert pos2 <= self.theta2_max
        self.servos.servo_move([int(pos1 * self.servos.DXL_RANGE + self.servos.DXL_MINIMUM_POSITION_VALUE),
                                int(pos2 * self.servos.DXL_RANGE + self.servos.DXL_MINIMUM_POSITION_VALUE)])

    def primitive(self, idx):
        if idx == 1:
            robot.move(0.55, 0.55)
            robot.move(0.45, 0.55)
            robot.move(0.45, 0.45)
            robot.move(0.55, 0.45)
            robot.move(0.55, 0.55)
        if idx == 2:
            robot.move(0.6, 0.6)
            robot.move(0.4, 0.6)
            robot.move(0.4, 0.4)
            robot.move(0.6, 0.4)

if __name__ == "__main__":
    robot = FiveBar()
    robot.primitive(1)
    robot.drift(robot.linear_max)
    robot.primitive(2)
    robot.drift(robot.linear_min)
    robot.primitive(1)
    robot.primitive(2)
    robot.reset()