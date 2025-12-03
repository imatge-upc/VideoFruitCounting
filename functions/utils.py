from statistics import mean
import os
import csv
import shutil
from bisect import bisect
from datetime import datetime
import math
import numpy as np
import utm
import shapefile
import pyproj
import matplotlib.cm as cm



def read_csv(csv_path):
    """
    Read input csv with all the detections

    Parameters:
        - csv_path: path to the 'csv' file

    Returns:
        - all_apples: list with all detections stored in the input csv
    """

    all_apples = []
    # Read the csv into list, each row is a list inside all_apples[]
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                pass
            else:
                all_apples.append(row[0:8])
            line_count += 1

    return all_apples


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


def get_gt_list(csv_path, gt_file):
    """
    Read the ground truth csv

    Parameters:
        - csv_path: path to the 'csv' file
        - gt_file: path to ground truth file

    Returns:
        - gt_list: list with the ground truth of stretch change
    """
    # Get the name of the video, from the path of csv data
    video_name = csv_path.split(os.path.sep)[-2]

    # List where will be stored the stretch changing frames
    gt_list = []


    with open(gt_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            row_name = row[0]
            if row_name[-1].isspace():
                row_name = row_name[:-1]
            if row_name == video_name:
                gt_list.append(row)
    gt_list = gt_list.pop(0)
    del gt_list[0]

    return gt_list


def stretch_velocity(all_apples, gt_list):
    """
    In each stretch change, assign all the apples that appears to the stretch where they are, and calculate their mean
    velocity to assign that velocity to the object that indicates the change of the stretch.

    Parameters:
        - all_apples: list with all detections stored in the input csv
        - gt_list: list with the ground truth of stretch change

    Returns:
        - assigned_apples: list with all the apples that have been assigned to a stretch
        - total_velocities: list with the final velocities of the stretch objects
        - id_ass_apples: list with the ids of the assigned apples
    """
    # List to store all the apples that have been assigned to a stretch
    assigned_apples = []

    # List to store the final velocities of the stretch objects
    total_velocities = []

    # List to store all the ids of the assigned apples
    id_ass_apples = []

    image_width = 1080

    for idx, frame_num in enumerate(gt_list):
        stretch = idx + 1  # idx + 1, because idx starts at 0

        # List to store all apples that appears in the frame of the changing stretch
        changing_apples_ids = []

        for apple_data in all_apples:
            # Add all apples that are in the frame to the changing_list
            if apple_data[0] == frame_num:
                # Add the ID, to know that we have to calculate the velocity of this apple
                changing_apples_ids.append(apple_data[1])
                id_ass_apples.append(int(apple_data[1]))

                # If apples appears in the frame, calculate which stretch it is, and store the id and the assigned stretch
                x_min = float(apple_data[2])
                x_max = x_min + float(apple_data[4])
                apple_center = x_max - ((x_max - x_min) / 2)
                if apple_center <= image_width / 2:
                    # Assign the stretch to the apple
                    assigned_apples.append([int(apple_data[1]), stretch - 1])
                else:
                    # Assign the stretch to the apple
                    assigned_apples.append([int(apple_data[1]), stretch])

        apples_to_calculate = {}
        for id in changing_apples_ids:
            for apple in all_apples:
                # Get all the appears of the apples needed to calculate the velocity
                if apple[1] == id:
                    if id in apples_to_calculate:
                        apples_to_calculate[id].append(apple)
                    else:
                        apples_to_calculate[id] = []
                        apples_to_calculate[id].append(apple)

        # List to store all the velocities off the apples
        #   [Vapple1, Vapple2, Vapple3....,VappleN]
        total_object_velocities = []

        if len(apples_to_calculate) == 0:
            total_velocities.append(0)
            continue

        for key, detections in apples_to_calculate.items():
            # There must be at least 2 apparitions of the apple in order to avoid problems
            # Calculation the velocity due to division by 0 pixels moved
            if len(detections) < 2:
                continue

            # List to store all the frames where the apple appears
            #   [frameX1, frameX2, frameX3....,frameXN]
            apple_frames = []
            # List to store al the bboxes centers of the apple
            apple_positions = []

            for detection in detections:
                apple_frames.append(int(detection[0]))

                # Calculate the position of each apparition of the apple and store all the positions in a list
                x_min = float(detection[2])
                x_max = x_min + float(detection[4])
                apple_center = x_max - ((x_max - x_min) / 2)
                apple_positions.append(apple_center)

                # If it is the last appear of the apple, calculate its velocity
                if detection == detections[-1]:
                    # Calculate the moved pixels and the number of frames that have passed
                    moved_pixels = max(apple_positions) - min(apple_positions)
                    passed_frames = max(apple_frames) - min(apple_frames)

                    # Calculate the apple velocity and store it into the total_velocities list
                    velocity = moved_pixels / passed_frames
                    total_object_velocities.append(velocity)

        # Calculate the mean velocity of all the apples that appears on the stretch changing frames
        # in order tho assign that velocity to the 'object' that indicates the stretch, so we can predict
        # the position of the object along the time, so total_velocities has the velocity in pixels/frame
        total_velocities.append(mean(total_object_velocities))

    return assigned_apples, total_velocities, id_ass_apples


def get_change_frames(gt_list, stretch_velocities):
    """
    Gets the list, where for each frame change, get from which frame to which frame the change occurs.

    Parameters:
        - gt_list: list with the ground truth of stretch change
        - stretch_velocities: list with the final velocities of the stretch objects

    Returns:
        - frames_changing: list with, for each frame change, get from which frame to which frame the change occurs
    """
    image_width = 1080
    gt_list = [int(i) for i in gt_list]
    # List to store values indicating the start and end frame of each stretch change
    # [(frame1, frame2), (frame3, frame4)...., (frameN-1, frameN)]
    frames_changing = []
    for idx, value in enumerate(gt_list):
        if stretch_velocities[idx] == 0:
            frames_changing.append([value - 50, value + 50, idx + 1])
            continue

        stretch_changing_frames = int(image_width / stretch_velocities[idx])
        frame1 = int(value - (stretch_changing_frames / 2))
        frame2 = int(value + (stretch_changing_frames / 2))

        frames_changing.append([frame1, frame2, idx + 1])

    return frames_changing


def most_frequent(List):
    return max(set(List), key=List.count)


def assign_all_apples(frames_changing, assigned_apples, all_apples, id_ass_apples, stretch_velocities, half_image = 540):
    """
    Calculate in which stretch each apple is located, taking as a reference the 'mark' of the stretch change

    Parameters:
        - frames_changing: list with, for each frame change, get from which frame to which frame the change occurs
        - assigned_apples: list with all the apples that have been assigned to a stretch
        - all_apples: list with all detections stored in the input csv
        - id_ass_apples: list with the ids of the assigned apples
        - stretch_velocities: list with the final velocities of the stretch objects
        - half_image: half the width of the input video (in pixels) 

    Returns:
        - assigned_apples: list with all the apples that have been assigned to a stretch
    """

    len_frames_changing = len(frames_changing)
    
    for idx, pair in enumerate(frames_changing):
        print (f'Processing {idx} of {len_frames_changing}')
        
        apples_to_assign = {}

        # List to store all the ids of the seen apples
        seen_apples = []
        for apple in all_apples:
            # Add the seen apples
            if int(apple[0]) < pair[1]:
                if not apple[1] in seen_apples:
                    seen_apples.append(apple[1])

            # Mark the apples detected between the two frames ass apples to assign
            if int(apple[1]) not in id_ass_apples:
                if pair[1] > int(apple[0]) > pair[0]:
                    if apple[1] in apples_to_assign:
                        apples_to_assign[apple[1]].append(apple)
                    else:
                        apples_to_assign[apple[1]] = []
                        apples_to_assign[apple[1]].append(apple)

        id = str
        for key, detections in apples_to_assign.items():
            if stretch_velocities[idx] == 0:
                continue

            apple_stretch = []

            for detection in detections:
                id = detection[1]
                # See if apple is at the right or at the left of the object
                center = int(pair[1] - pair[0])
                final_center = pair[0] + (center / 2)
                if int(detection[0]) < final_center:
                    x = int((final_center - int(detection[0])) * stretch_velocities[idx])
                    final_x = half_image + x
                else:
                    x = int((int(detection[0]) - final_center) * stretch_velocities[idx])
                    final_x = half_image - x

                # If the apples appear in the frame, calculate which stretch it is, and store the id and the assigned stretch
                x_min = float(detection[2])
                x_max = x_min + float(detection[4])
                apple_center = x_max - ((x_max - x_min) / 2)
                if apple_center < final_x:
                    apple_stretch.append(int(frames_changing[idx][2]) - 1)
                else:
                    apple_stretch.append(frames_changing[idx][2])

            # Count the number of times that apple is seen in each stretch
            assigned = most_frequent(apple_stretch)
            if int(id) not in id_ass_apples:
                assigned_apples.append([id, assigned])
                id_ass_apples.append(int(id))

        # Check if the ids in seen_apples are assigned, if not, assign them
        for apple in seen_apples:
            if not int(apple) in id_ass_apples:
                assigned_apples.append([int(apple), frames_changing[idx][2] - 1])
                id_ass_apples.append(int(apple))

    return assigned_apples


def write_results(apples_stretch, path_to_all_apples, path_to_stretch_num, csv_path, path_to_all_predictions):
    """
    write the final results on a csv

    Parameters:
        - apples_stretch: list with all the apples that have been assigned to a stretch
        - path_to_all_apples: path to write apples_stretch.csv
        - path_to_stretch_num: path to write all_apples.csv
        - csv_path: source path of all_predictions.csv
        - path_to_all_predictions: path to write all_predictions.csv
    """
    # Count number of apples in each stretch
    counts = {}

    for sublist in apples_stretch:
        value = sublist[1]
        if value < 1 or value > 21:
            continue
        if value in counts:
            counts[value] += 1
        else:
            counts[value] = 1

    os.makedirs(os.path.dirname(path_to_all_apples), exist_ok=True)
    f = open(path_to_all_apples, 'w')
    header = ['Stretch', 'Num_apples']
    writer = csv.writer(f)
    writer.writerow(header)
    for value, count in counts.items():
        writer.writerow([value, count])
    f.close()

    os.makedirs(os.path.dirname(path_to_stretch_num), exist_ok=True)
    f = open(path_to_stretch_num, 'w')
    header = ['AppleID', 'Stretch']
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(apples_stretch)
    f.close()

    os.makedirs(os.path.dirname(path_to_all_predictions), exist_ok=True)
    shutil.copy(csv_path, path_to_all_predictions)



