import sys
import os
import numpy as np
import cv2
import argparse
from pathlib import Path
import pandas as pd
from bisect import bisect


from tracking.FruitTracker_simple import FruitTracker
from functions.functions import get_all_detections, perform_tracking, write_results_simple

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate detection')

    parser.add_argument('--video', dest='video', help='Video file path.')
    parser.add_argument('--segments-file', dest='segments_file',
                        help='File containing the limits of the segments. For each video, the segment limits'
                             ' are defined at the frame where the segment change line is centered on the frame.')
    parser.add_argument('--weights', dest='weights', help='Weights file for the YOLOv5 detection model')
    parser.add_argument('--offset', dest='offset', default=0, help='Number of frames to skip at the beginning.')
    parser.add_argument('--conf', dest='confidence', default=0.25,
                        help='Minimum detection confidence to be taken into account.')
    parser.add_argument('--min_area', dest='min_area', default='-1',
                        help='Min area (in pixels) of the apples to be tracked. If -1, the value will be determined'
                        'from the seventh field of the video file name (distance).')

    parser.add_argument('--data', type=str, default='data/apple_segmentation.yaml', help='dataset.yaml path')    
    parser.add_argument('--camera', dest='camera', default='Agnostic', help='Camera type (ZED or KA). Only necessary for the provided videos.')

    parser.add_argument('--rotate', dest='rotate', default=False, help='Whether to rotate each frame 90º clockwise. Useful when camera has been situated vertically')
    parser.add_argument('--results_file', dest='predictions_filename', default="all_predictions.csv", help='Filename for the output tracking results')
    
    args = parser.parse_args()

    return args



def configure_min_area_filter_from_filename (camera_name, distance):
    '''
    Return the minimum area size for the configured presets:
      - Camera: ZED of KA
      - Recording distances: 125, 175, 225
    '''

    if camera_name =='Agnostic':
        print ('Error: This funcion is only intended for Kinect or ZED cameras')
        sys.exit()
        
    preset_areas = {'KA':{225: 100, 175:221, 125: 624}, 'ZED':{225:77, 175:500, 125:120}}
    if distance not in [125,175,225]:
        print ('ERROR: valid distances for the provided data are 125, 175 and 225')
        sys.exit(1)
    return preset_areas[camera_name][distance]
    

def get_stretch_changes_simple(video_path, segments_file):
    """
    Get a list that contains all the stretch changes for the current video

    Parameters:
        - video_path: Path to the video being processed
         - segment_file: File with the stretch locations (frames) in video

    Returns:
        - stretch_changes: list with all the stretch changes
    """

    df_stretch = pd.read_csv(segments_file, header=None)

    # Trim any trailing spaces in the names column                                                                                                   
    df_stretch = df_stretch.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Get the row corresponding to the current video                                                                                                 
    mask = df_stretch[0].isin([video_path.stem])

    # Return a list with the stretch_changes for the current video                                                                                   
    return df_stretch[mask].values.flatten()[1:]    


import torch
import torch.backends.cudnn as cudnn

# pip install yolov5
from yolov5.utils.torch_utils import select_device
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes

class yolo_detector:
    def __init__(self,
                 weights='data/last.pt', # use trained PT file
                 imgsz=(640, 640), # Mod
                 data='data/apple_segmentation.yaml',
                 conf_thres=0.25,
                 iou_thres=0.45,
                 half=False,
                 dnn=False,
                 max_det=1000,
                 ):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

        self.device = select_device('') # Mod
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data, fp16=half) # load the model
        self.stride, pt = self.model.stride, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        half &= pt and self.device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        self.half = half
        if pt:
            self.model.model.half() if half else self.model.model.float()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model.warmup(imgsz=(1, 3, *self.imgsz))



        
    def run(self, frame):
        print (frame.shape)
        
        img = letterbox(frame, self.imgsz, stride=self.stride, auto=True)[0]

        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)

        im = torch.from_numpy(img).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0

        print (im.shape)
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, max_det=self.max_det)

        to_return_bbox = []
        to_return_scores = []
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in reversed(det):
                    bbox = (torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    bbox[2], bbox[3] = w, h
                    to_return_bbox.append(bbox)
                    to_return_scores.append(conf.item())
        return to_return_bbox, to_return_scores


def main():
    args = parse_args()

    video_path = Path(args.video)
    segments_file = args.segments_file
    confidence = float(args.confidence)
    offset = int(args.offset)
    weights = args.weights
    min_area = int(args.min_area)
    camera = args.camera
    rotate = args.rotate
    data = args.data
    csv_all_predictions_filename = args.predictions_filename


    print ('Parameters: ---------------------------')
    print (f'video path: {video_path}')
    print (f'segments file: {segments_file}')
    print (f'confidence: {confidence}')
    print (f'offset: {offset}')
    print (f'weights: {weights}')
    print (f'min area: {min_area}')
    print (f'camera: {camera}')
    print (f'rotate: {rotate}')
    print (f'data: {data}')
    print (f'predictions filename: {csv_all_predictions_filename}')
    
    # Initialize variables
    ii = offset

    if camera not in ['Agnostic', 'ZED', 'KA']:
        print ('Error: Incorrect camera type. If unsure, use Agnostic')
        sys.exit()
        
    if camera == 'Agnostic' and min_area == -1:
        print ('ERROR!: When using agnostic camera, the min_area must be splicitly provided')
        sys.exit()
        
    if min_area == -1:
        # The distance from camera to the trees is encoded in the video name
        distance = int(os.path.split(video_path.stem)[1].split('_')[6])
        min_area = configure_min_area_filter_from_filename(camera, distance)

    # ???????
    if offset != 0:
        ii = offset + 1

    
    # get the information of the segments change
    stretch_changes = get_stretch_changes_simple(video_path, segments_file)

    
    # Parameters for detector configuration
    imgsz = [640]
    imgsz *= 2 if len(imgsz) == 1 else 1
    conf_thres = confidence
    iou_thres = 0.45
    half = False
    dnn = False
    max_det = 1000

    detector = yolo_detector(weights, imgsz, data, conf_thres, iou_thres, half, dnn, max_det)

    fruit_tracker = FruitTracker(img_sz=imgsz, min_area=min_area)

    # List used to find the value of each stretch  ???????
    list_to_compare = list(range(1, 22))
    reverse_list = list(range(21, 0, -1))

    # List with the coordinates of each frame
    frame_coordinates_list = []

    # List with all the results of tracking, Saved on CSV
    # all_tracking_predictions = [[Number of Frame, ID1, x_min, y_min, w, h, Score, Stretch],
    #                            [Number of Frame, ID2, x_min, y_min, w, h, Score, Stretch]], ...]
    all_tracking_predictions = []

    # List with the coordinates of each frame
    frame_coordinates_list = []

    prev_stretch    = 0
    frames_interval = ii


    cap = cv2.VideoCapture(args.video)
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Skip the first frames. Note that while cap.set(cv2.CAP_PROP_POS_FRAMES, offset) can also be used for that, 
        # actually reading and discarting the frames is safer and probably faster as well.
        if frame_counter < ii:
            continue

        stretch = bisect(stretch_changes, ii)

        
        frame_counter = frame_counter + 1
                
        print (f'Processing frame f{frame_counter}')

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if (rotate):
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        dets_conf = get_all_detections(detector, frame)


        # Pass the detections, number of frame and frame(np.array) to the tracker, also pass camera name
        fruit_tracker.get_detections(dets_conf, ii, frame, stretch)
        
        # Perform the tracking and save the results, new_predictions = last predictions returned by tracker
        tracking_predictions = fruit_tracker.track_yolo_results()

        new_predictions = tracking_predictions[-1]
        
        if camera != 'Agnostic':
            # If the video is shot from west, apply a transformation in order to invert the stretch number, so the stretch
            # 1 will be 21, 2 -> 20, etc...
            
            # Extract the orientation (west or east) from which the video was shot. For the provided videos this is encoded
            # in the filename
            video_orientation = video_path.stem.split('_')[4]
            
            if video_orientation == 'w':
                stretch_index = bisect(list_to_compare, stretch)
                if stretch_index == 0:
                    stretch = 21
                else:
                    stretch = reverse_list.index(stretch_index)

                #if str(video_path)[-17: -16] == 'w':  # JRMR: was in perform_tracking()
                stretch += 1

        all_tracking_predictions = perform_tracking(new_predictions=new_predictions,
                                                    all_tracking_predictions=all_tracking_predictions,
                                                    stretch=stretch, i=ii)


        ii += 1

    write_results_simple(csv_all_predictions_filename=csv_all_predictions_filename,
                         all_tracking_predictions=all_tracking_predictions)

        
    cap.release()

if __name__ == "__main__":
    main()

    
    
            
