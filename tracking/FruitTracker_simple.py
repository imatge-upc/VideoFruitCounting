import os
import sys
import numpy as np

from tracking.bytetrack import byte_tracker


def create_tracker():
    """
    Create a tracker 
    :return: the tracker
    """
    tracker = byte_tracker.BYTETracker()

    return tracker


class FruitTracker:
    def __init__(self, img_sz, min_area):
        self.img_sz = img_sz
        self.min_area = min_area
        self.frame_num = 0
        self.all_detections = []
        self.actual_frame_dets = []
        self.det_centers = []
        self.det_ids = []
        self.predictions = []
        self.tracker = create_tracker()
        self.frame = []
        self.new_frame = []
        self.stretch = 0
        self.all_apples = {}
        self.distance = 0
        self.new_frame_dets = []

    def get_detections(self, detections, frame_num, frame, stretch):
        self.frame = frame
        self.frame_num = frame_num
        self.actual_frame_dets = detections
        self.all_detections.append(detections)
        self.stretch = stretch
        self.new_frame_dets = []

    def filter_detections_by_size(self):

        for detection in self.actual_frame_dets:
            bbox = detection[:4]
            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) >= self.min_area:
                self.new_frame_dets.append(detection)
            elif (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < self.min_area:
                self.actual_frame_dets.remove(detection)

        return self.new_frame_dets

    def track_detections_frame(self, detections, img_size):

        results = {
            'ids': [],
            'bboxes': [],
            'appears': [],
            'temporal_stretch': [],
            'final_stretch': [],
            'scores': [],
            'depth': []
        }

        # If there are no detections in the frame, skip it (all zeros)
        if detections is None or len(detections) == 0:
            trackers = self.tracker.update(np.empty((0, 5)), img_info=img_size, img_size=img_size)
            
        # If there are detections in the frame, track them
        else:
            trackers = self.tracker.update(np.array(detections), img_info=img_size, img_size=img_size)

            for t in trackers:
                score = t.score
                t_id = t.track_id
                t = t.tlbr
                t = np.append(t, t_id)

                self.det_centers.append((int((t[0] + t[2]) / 2), int((t[1] + t[3]) / 2)))
                self.det_ids.append(int(t[4]))
                results['bboxes'].append([int(t[0]), int(t[1]), int(t[2]), int(t[3])])
                results['ids'].append(int(t[4]))
                results['appears'].append(1)
                results['temporal_stretch'].append(0)
                results['final_stretch'].append(0)
                results['scores'].append(score)
                results['depth'].append(0)

        self.predictions.append(results)
        return self.det_centers, self.det_ids, self.predictions

    def get_all_apples(self, all_apples):
        self.all_apples = all_apples

    def track_yolo_results(self):
        # Filter detections by size
        self.actual_frame_dets = self.filter_detections_by_size()

        # perform the tracking of the detections
        self.det_centers, self.det_ids, self.all_tracking_predictions = self.track_detections_frame(
            detections=self.actual_frame_dets,
            img_size=self.img_sz)

        return self.all_tracking_predictions
