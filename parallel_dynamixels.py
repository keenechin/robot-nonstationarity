import os
import sys, tty, termios
import dynamixel_sdk as dmx

# Designed to control 2 XC-430-150-T servos from linux using U2D2

class DynamixelDriver():
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)

        self.ADDR_TORQUE_ENABLE          = 64
        self.ADDR_GOAL_POSITION          = 116
        self.ADDR_PRESENT_POSITION       = 132
        self.DXL_MINIMUM_POSITION_VALUE  = 0         # Refer to the Minimum Position Limit of product eManual
        self.DXL_MAXIMUM_POSITION_VALUE  = 4095      # Refer to the Maximum Position Limit of product eManual
        self.BAUDRATE                    = 57600

        # DYNAMIXEL Protocol Version (1.0 / 2.0)
        # https://emanual.robotis.com/docs/en/dxl/protocol2/
        PROTOCOL_VERSION            = 2.0

        DEVICENAME                  = '/dev/ttyUSB0'
        self.DXL_IDS                     = [1, 2]
        self.TORQUE_ENABLE               = 1
        self.TORQUE_DISABLE              = 0
        self.DXL_MOVING_STATUS_THRESHOLD = 20
        self.portHandler = dmx.PortHandler(DEVICENAME)
        self.packetHandler = dmx.PacketHandler(PROTOCOL_VERSION)

        self.connect()

        # Enable Dynamixel Torque
        for dxl_id in self.DXL_IDS:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE)
            if dxl_comm_result != dmx.COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            else:
                print(f"Dynamixel ID {dxl_id} has been successfully connected")

    def getch(self):
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        return ch

    def connect(self):
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            print("Press any key to terminate...")
            self.getch()
            quit()

        # Set port baudrate
        if self.portHandler.setBaudRate(self.BAUDRATE):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            print("Press any key to terminate...")
            getch()
            quit()

    def servo_move(self, pos, v=True):
        # Write goal position
        for dxl_id in self.DXL_IDS:
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, dxl_id, self.ADDR_GOAL_POSITION, pos)
            if dxl_comm_result != dmx.COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        if v:
            # Read present position
            while 1:
                for dxl_id in self.DXL_IDS:
                    dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, dxl_id, self.ADDR_PRESENT_POSITION)
                    if dxl_comm_result != dmx.COMM_SUCCESS:
                        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
                    elif dxl_error != 0:
                        print("%s" % packetHandler.getRxPacketError(dxl_error))

                    print("[ID:%03d] GoalPos:%03d  PresPos:%03d" % (dxl_id, dxl_goal_position[index], dxl_present_position))

                if abs(dxl_goal_position[index] - dxl_present_position) < self.DXL_MOVING_STATUS_THRESHOLD:
                    break


    def __del__(self):
        # Disable Dynamixel Torque
        for dxl_id in self.DXL_IDS:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE)
            if dxl_comm_result != dmx.COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))

        # Close port
        self.portHandler.closePort()



if __name__ == "__main__":
    index = 0
    robot = DynamixelDriver()
    dxl_goal_position = [robot.DXL_MINIMUM_POSITION_VALUE, robot.DXL_MAXIMUM_POSITION_VALUE]         # Goal position


    # Test Motor Movements
    while 1:
        print("Press any key to continue! (or press ESC to quit!)")
        if robot.getch() == chr(0x1b):
            break

        robot.servo_move(dxl_goal_position[index])
        # Change goal position
        if index == 0:
            index = 1
        else:
            index = 0





