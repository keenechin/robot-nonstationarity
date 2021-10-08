import pyrealsense2 as rs
import cv2
from multiprocessing import Process, Queue
import os
import signal
import time
import numpy as np
from queue import Empty


class RealsenseCamera():
    def __init__(self, mode="Detection"):
        camera = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.camera = camera
        self.feed = Queue()
        if mode == "Tracking":
            self.process = Process(target=self.track_points, args=(self.feed,))
        if mode == "Detection":
            self.process = Process(
                target=self.detect_points, args=(self.feed,))

    def start(self):
        self.process.start()

    def create_tracker(self, tracker_name):
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }
        return OPENCV_OBJECT_TRACKERS[tracker_name]()

    def initialize_tracking(self, trackerType="mosse"):  # medianflow
        self.camera.start()
        bboxes = []
        multi_tracker = cv2.MultiTracker_create()
        for i in range(100):  # give camera time to autoadjust
            frame = self.get_frame()

        while True:
            init_window = 'MultiTracker selections (MAKE BIG BOXES)'
            bbox = cv2.selectROI(init_window, frame,
                                 fromCenter=False, showCrosshair=True)
            bboxes.append(bbox)
            print("Press q to quit selecting boxes and start tracking")
            print("Press any other key to select next object")
            k = cv2.waitKey(0) & 0xFF
            if (k == ord('q')):  # q is pressed
                break
        cv2.destroyWindow(init_window)

        bboxes = bboxes[:2]
        print(bboxes)
        for bbox in bboxes:
            multi_tracker.add(self.create_tracker(trackerType), frame, bbox)
        time.sleep(2)
        return bboxes, multi_tracker

    def get_frame(self):
        frames = self.camera.wait_for_frames()
        color_frame = frames.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())
        return frame

    def initialize_detection(self):
        init_window = "Draw box around relevant area"
        for i in range(100):
            frame = self.get_frame()
        cv2.imshow(init_window, frame)
        bbox = cv2.selectROI(init_window, frame,
                             fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(init_window)
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), 255, -1)
        return mask

    def detect_points(self, queue, visualize=True):
        self.camera.start()
        mask = self.initialize_detection()
        while True:
            try:
                frame = self.get_frame()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_blurred = cv2.blur(gray, (3, 3))
                masked = cv2.bitwise_and(gray_blurred, gray_blurred, mask=mask)
                detected_circles = cv2.HoughCircles(masked,
                                                    cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50,
                                                    param2=0.95, minRadius=2, maxRadius=7)

                if detected_circles is not None:
                    # make integer, flatten
                    circles = np.uint16(np.around(detected_circles[0, :, :]))

                    queue.put(circles)

                    if visualize:
                        for circle in circles:
                            cv2.circle(img=frame, center=(
                                circle[0], circle[1]), radius=circle[2], color=(0, 200, 0), thickness=1)
                            cv2.imshow('Live Feed', frame)
                            k = cv2.waitKey(1)
                            if (k == ord('q')):  # q is pressed
                                break
                else:
                    print("Detection Fail.")
            except KeyboardInterrupt:
                print("User exit during detection")
                break

    def track_points(self, queue, visualize=False):
        bboxes, multi_tracker = self.initialize_tracking()
        while True:
            frame = self.get_frame()
            success, bboxes = multi_tracker.update(frame)
            if success:
                if not queue.empty():
                    try:
                        queue.get_nowait()
                    except Empty:
                        pass

                states = []
                for bbox in bboxes:
                    state = [int(bbox[0] + bbox[2]/2),
                             int(bbox[1] + bbox[3]/2)]
                    states = states + state

                queue.put(states)
            else:
                print("Tracking Fail.")
            if visualize:
                cv2.circle(img=frame, center=state, radius=5,
                           color=(0, 0, 200), thickness=-1)
                cv2.imshow('Live Feed', frame)
                k = cv2.waitKey(1)
                if (k == ord('q')):  # q is pressed
                    break

    def __del__(self):
        os.kill(self.process.pid, signal.SIGKILL)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    camera = RealsenseCamera()
    camera.start()
    while True:
        try:
            print(camera.feed.get())
        except KeyboardInterrupt:
            print("\nUser exit main.")
            break
    
