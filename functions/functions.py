import os
import numpy as np
from bisect import bisect
import csv
import itertools
from typing import Optional, Tuple
import cv2
import pandas as pd # JRMR

from functions.bbox_utils import augment_size_of_bboxes_in_crops


def filter_by_depth(new_predictions, depth, user_depth):
    """
    Filter detections by depth

    Parameters:
        - new_predictions: detections to filter
        - depth: Depth images as np
        - user_depth: Max depth of detections

    Returns:
        - new_predictions: with the input detections filtered by depth
    """
    for id in new_predictions['ids']:
        apple_id_index = new_predictions['ids'].index(id)

        bbox = new_predictions['bboxes'][apple_id_index]
        bbox_depth = depth[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        if len(bbox_depth) > 0:
            mean_depth = np.mean(bbox_depth)

            if mean_depth > user_depth:
                del new_predictions['ids'][apple_id_index]
                del new_predictions['bboxes'][apple_id_index]
                del new_predictions['appears'][apple_id_index]
                del new_predictions['temporal_stretch'][apple_id_index]
                del new_predictions['final_stretch'][apple_id_index]
                del new_predictions['scores'][apple_id_index]
                del new_predictions['depth'][apple_id_index]
            else:
                new_predictions['depth'][apple_id_index] = round(float(mean_depth), 2)

    return new_predictions


def xywh2abcd(xywh):
    """
    Transform the bounding boxxes from yxwh format to abcd format

    Parameters:
        - xywh: bbox (X center, Y center, Width, Height)

    Returns:
        - abcd: bbox (point A, point B, point C, point D)
    """
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = xywh[0]
    x_max = xywh[0] + xywh[2]
    y_min = xywh[1]
    y_max = xywh[1] + xywh[3]

    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_min
    output[2][1] = y_max

    output[3][0] = x_max
    output[3][1] = y_max

    return output


def get_scan_stretch(Prow_transformed, transferred_coordinates):
    """
    Get index of the array of points to a given point, so that the index is equal to the stretch number.

    Parameters:
        - Prow_transformed: coordinate of the row translated to (0, 0) and rotated
        - transferred_coordinates: actuall camera coordiantes translated and rotated

    Returns:
        - index: stretch where the camera is located
    """
    actual_point = transferred_coordinates[0]

    x_axis_value_list = []
    for i in range(len(Prow_transformed)):
        x_axis_value_list.append(Prow_transformed[i][0])

    # Search in which index of the list should be placed the X coordinate given
    index = bisect(x_axis_value_list, actual_point)
    return index


def get_all_detections(detector, frame):
    """
    Run the detector in each iteration

    Parameters:
        - detector: Yolo Detector
        - frame: Frame to pass to the detector

    Returns:
        - dets_conf: with all the detections on the frame
    """
    # Get the detections and confidences for each frame
    detections, out_scores = detector.run(frame)

    # x1,y1,x2,y2
    for detection in detections:
        detection[0] = detection[0]
        detection[1] = detection[1]
        detection[2] = detection[0] + (detection[2])
        detection[3] = detection[1] + (detection[3])

    # Augment the bbox size by 30%
    dets_conf = []
    for k in range(len(out_scores)):
        a = detections[k]
        augment_size_of_bboxes_in_crops(a)
        b = out_scores[k]
        a.append(b)

        dets_conf.append(a)
    return dets_conf


# JRMR: Simplification of function get_stretch_changes.
# The construction of path names is removed from this function and moved to main() function
# Reading of the segments_file (previously named 'exec_mode'??) is now performed using pandas.
def get_stretch_changes_sframes(video_path, segments_file):
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


def get_stretch_changes(video_path, current, exec_mode):
    """
    Get a list that contains all the stretch changes basing into given csv ground truth information

    Parameters:
        - video_path: Path to the video being processed
        - current: Current path of the file inside the project
        - exec_mode: File with the stretch locations (frames) in video

    Returns:
        - stretch_changes: list with all the stretch changes
        - file_results: path to the final video folder
        - csv_path: path to the final video results '.csv' file
        - csv_frame_coordinates: path to the final video coordinates '.csv' file - Only in case of GPS
    """
    stretch_changes = []

    csv_frame_coordinates = str
    file_results = str
    final_dir = str
    if str(video_path)[-3:] == 'mkv':
        file_results = os.path.join(current, 'results', 'results_GT', 'KA')
        final_dir = os.path.join(file_results, video_path.stem)
        os.makedirs(os.path.dirname(final_dir), exist_ok=True)

        with open(exec_mode, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                row_name = row[0]
                if row_name[-1].isspace():
                    row_name = row_name[:-1]
                if row_name == video_path.stem:
                    stretch_changes.append(row)

    elif str(video_path)[-3:] == 'svo':
        file_results = os.path.join(current, 'results', 'results_GT', 'ZED')
        final_dir = os.path.join(file_results, video_path.stem)
        os.makedirs(os.path.dirname(final_dir), exist_ok=True)

        with open(exec_mode, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                row_name = row[0]
                if row_name[-1].isspace():
                    row_name = row_name[:-1]
                if row_name == video_path.stem:
                    stretch_changes.append(row)

    stretch_changes = list(itertools.chain.from_iterable(stretch_changes))
    stretch_changes.pop(0)
    stretch_changes = list(map(int, stretch_changes))

    # Paths to output files
    csv_path = os.path.join(final_dir, 'all_predictions.csv')

    # JRMR: Removed because this is never executed. This function is only called in SFRAME mode, never in the SGNSS (.txt) mode
    '''
    if str(video_path)[-3:] == 'svo' and str(exec_mode)[-3:] == 'txt':
        csv_frame_coordinates = os.path.join(file_results, video_path.stem, '/frame_coordinates.csv')
    '''
    
    return stretch_changes, file_results, csv_path, csv_frame_coordinates


def perform_tracking(new_predictions, all_tracking_predictions, stretch, i):
    """
    Store date of each detection and each apple

    Parameters:
        - new_predictions: predictions on the last frame
        - all_tracking_predictions: all the predictions to track
        - stretch: stretch in which the camera is located
        - i: frame number

    Returns:
        - all_tracking_predictions: Actualized predictions to track
    """
    # JRMR - Moved to main function to avoid dependence of this functions with the filename
    #if str(video_path)[-17: -16] == 'w':
    #    stretch += 1

    for id in new_predictions['ids']:
        apple_id_index = new_predictions['ids'].index(id)
        score = new_predictions['scores'][apple_id_index]

        # Actualize the actual apple prediction
        new_id = new_predictions['ids'][apple_id_index]
        x_min = new_predictions['bboxes'][apple_id_index][0]
        y_min = new_predictions['bboxes'][apple_id_index][1]
        w = new_predictions['bboxes'][apple_id_index][2] - \
            new_predictions['bboxes'][apple_id_index][0]
        h = new_predictions['bboxes'][apple_id_index][3] - \
            new_predictions['bboxes'][apple_id_index][1]

        # actual_apple_pred = [frame_num, ID, x_min, y_min, w, h, score (confidence of the detection), stretch]
        actual_apple_pred = []

        actual_apple_pred.clear()

        actual_apple_pred.append(i)
        actual_apple_pred.append(new_id)
        actual_apple_pred.append(x_min)
        actual_apple_pred.append(y_min)
        actual_apple_pred.append(w)
        actual_apple_pred.append(h)
        actual_apple_pred.append(round(score, 5))
        actual_apple_pred.append(stretch)

        all_tracking_predictions.append(actual_apple_pred)

    return all_tracking_predictions

def write_results(csv_all_predictions_filename, csv_frame_coordinates, all_tracking_predictions, video_extension,
                  gps_file, frame_coordinates_list):
    """
    Write the final results

    Parameters:
        - csv_all_predictions_filename: path to all predictions file
        - csv_frame_coordinates: path to frame coordinates in case of GPS
        - all_tracking_predictions: results of all the detections in each frame
        - video_extension: video extension 'mkv' or 'svo' to check if executed with GPS
        - gps_file: gps or gt file to check if executed with GPS
        - frame_coordinates_list: list with the coordinates of each frame
    """
    # Write all predictions on CSV
    os.makedirs(os.path.dirname(csv_all_predictions_filename), exist_ok=True)
    f = open(csv_all_predictions_filename, 'w')
    all_tracking_predictions_header = ['NumFrame', 'ID', 'x_min', 'y_min', 'w', 'h', 'score', 'stretch']
    writer = csv.writer(f)
    writer.writerow(all_tracking_predictions_header)
    writer.writerows(all_tracking_predictions)
    f.close()

    if video_extension == 'svo' and str(gps_file)[-3:] == 'txt':
        # Write the CSV with frame coordinates
        os.makedirs(os.path.dirname(csv_frame_coordinates), exist_ok=True)
        f = open(csv_frame_coordinates, 'w')
        header = ['NumFrame', 'Latitude', 'Longitude']
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(frame_coordinates_list)
        f.close()

def write_results_simple (csv_all_predictions_filename, all_tracking_predictions):
    """
    Write the final results.

    Parameters:
        - csv_all_predictions_filename: path to all predictions file
        - all_tracking_predictions: results of all the detections in each frame
    """
    # Write all predictions on CSV
    os.makedirs(os.path.dirname(csv_all_predictions_filename), exist_ok=True)
    f = open(csv_all_predictions_filename, 'w')
    all_tracking_predictions_header = ['NumFrame', 'ID', 'x_min', 'y_min', 'w', 'h', 'score', 'stretch']
    writer = csv.writer(f)
    writer.writerow(all_tracking_predictions_header)
    writer.writerows(all_tracking_predictions)
    f.close()


        
