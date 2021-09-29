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
        time.sleep(0.1)
    
    def reset(self):
        self.linear.move_joint_position(self.linear_zeros, 1.0)
        self.move_abs(0.5, 0.5)

    def drift(self, pos):
        assert pos >= self.linear_min
        assert pos <= self.linear_max
        p = self.linear_zeros
        p[0, 3:7] = pos
        self.linear.move_joint_position(p, 1.0)
        time.sleep(0.01)
        # self.linear.wait_until_done_moving()

    def move_abs(self, pos1, pos2):
        print(f"{pos1},{pos2}")
        pos1 = max(pos1, self.theta1_min)
        pos1 = min(pos1, self.theta1_max)
        pos2 = max(pos2, self.theta2_min)
        pos2 = min(pos2, self.theta2_max)
        print(f"{pos1},{pos2}")
        assert pos1 >= self.theta1_min
        assert pos2 >= self.theta2_min
        assert pos1 <= self.theta1_max
        assert pos2 <= self.theta2_max
        self.servos.servo_move([int(pos1 * self.servos.DXL_RANGE + self.servos.DXL_MINIMUM_POSITION_VALUE),
                                int(pos2 * self.servos.DXL_RANGE + self.servos.DXL_MINIMUM_POSITION_VALUE)])

    def move_delta(self, delta1, delta2):
        curr1, curr2 = self.get_pose()
        new_pos1 = curr1 + delta1
        new_pos2 = curr2 + delta2
        self.move_abs(new_pos1, new_pos2)

    def get_pose(self):
        _, _, curr1 = self.servos.get_pos(self.servos.DXL_IDS[0])
        _, _, curr2 = self.servos.get_pos(self.servos.DXL_IDS[1])
        curr1 = (curr1 - self.servos.DXL_MINIMUM_POSITION_VALUE)/self.servos.DXL_RANGE
        curr2 = (curr2 - self.servos.DXL_MINIMUM_POSITION_VALUE)/self.servos.DXL_RANGE
        return curr1, curr2

    def primitive(self, idx):
        if idx == 1:
            self.move_abs(0.45, 0.55)
            self.move_abs(0.45, 0.45)
            self.move_abs(0.55, 0.45)
            self.move_abs(0.55, 0.55)
        if idx == 2:
            self.move_abs(0.6, 0.6)
            self.move_abs(0.4, 0.6)
            self.move_abs(0.4, 0.4)
            self.move_abs(0.6, 0.4)
        
        if idx == 3:
            self.move_delta(-0.1, 0)
            self.move_delta(0.0, -0.1)
            self.move_delta(0.1, 0.0)
            self.move_delta(0, 0.1)
            
        if idx == 4:
            self.move_delta(0.1, 0.1)
            self.move_delta(-0.1, 0.0)
            self.move_delta(0.0, -0.1)
            self.move_delta(0.1, 0.0)


if __name__ == "__main__":
    robot = FiveBar()
    # robot.primitive(1)
    # robot.drift(robot.linear_max)
    # robot.primitive(2)
    # robot.drift(robot.linear_min)
    # robot.primitive(1)
    # robot.primitive(2)
    # robot.reset()
    robot.primitive(3)
    robot.drift(robot.linear_max)
    robot.primitive(4)
    robot.drift(robot.linear_min)
    robot.primitive(3)
    robot.primitive(4)
    robot.reset()
