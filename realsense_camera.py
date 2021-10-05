import pyrealsense2 as rs
import cv2
from multiprocessing import Process, Queue
import numpy as np
from queue import Empty



class RealsenseCamera():
    def __init__(self):
        self.camera = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    def get_rgb(self):
        frames = self.camera.wait_for_frames()
        color_frame = frames.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())
        return frame

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

    def initialize_tracking(self, trackerType="csrt"):
        bboxes = []
        self.camera.start(self.config)
        multi_tracker = cv2.MultiTracker_create()
        for i in range(100): # give camera time to autoadjust
            frame = self.get_rgb()

        while True:
            bbox = cv2.selectROI('MultiTracker', frame, fromCenter=False, showCrosshair=True)
            bboxes.append(bbox)
            print("Press q to quit selecting boxes and start tracking")
            print("Press any other key to select next object")
            k = cv2.waitKey(0) & 0xFF
            if (k == ord('q')):  # q is pressed
                break
        cv2.destroyWindow('MultiTracker')

        bboxes = bboxes[:2]
        for bbox in bboxes:
            multi_tracker.add(self.create_tracker(trackerType), frame, bbox)
        return bboxes, multi_tracker

    def get_feed(self):
        self.feed = Queue()
        bboxes, multi_tracker = self.initialize_tracking()
        # self.track_points(self.feed, bboxes, multi_tracker, True)
        self.process = Process(target=self.track_points, args=(self.feed, bboxes, multi_tracker, True))
        return self.feed

    def track_points(self, queue, bboxes, multi_tracker, visualize=True):
        while True:
            frame = self.get_rgb()
            success, bboxes = multi_tracker.update(frame)
            if success:
                if not queue.empty():
                    try:
                        queue.get_nowait()
                    except Empty:
                        pass

                states = []
                for bbox in bboxes:
                    state = [int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2)]
                    states = states + state
                    print(state)
                    cv2.circle(img=frame, center=state, radius=5, color=(0, 0, 200), thickness=-1)

                queue.put(states)
            if visualize:
                cv2.imshow('Live Feed', frame)
                k = cv2.waitKey(1)
                if (k == ord('q')):  # q is pressed
                    break

    def __del__(self):
        cv2.destroyAllWindows()
        self.process.terminate()

