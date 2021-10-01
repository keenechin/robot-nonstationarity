from five_bar import FiveBar
import pyrealsense2 as rs
import matplotlib.pyplot as plt


class FiveBarEnv():
    def __init__(self):
        self.hardware = FiveBar()
        # self.camera = rs.pipeline()
        # self.camera.start()

    def show_feed(self):
        camera = rs.pipeline()
        camera.start()
        for i in range(100):
            frames = camera.wait_for_frames()
            frame = frames.get_depth_frame()
            plt.imshow(frame)
        plt.show()
             
            


    def __del__(self):
        self.camera.stop()


if __name__ == "__main__":
    env = FiveBarEnv()
    env.show_feed()
    env.hardware.test_motion()
