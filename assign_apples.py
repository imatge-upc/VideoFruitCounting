from functions.utils import stretch_velocity, read_csv, get_change_frames, assign_all_apples, write_results, get_gt_list
import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Perform apple assignment to stretches')

    parser.add_argument('--data_path', dest='data_path',
                        help='Path where the subdirectories generated when executing the fruit_tracker_simple.py'
                             'script are located.')
    parser.add_argument('--sframe_file', dest='sframe_file',
                        help='Path to the stretch frames "csv" file.', required=False)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    data_path = args.data_path
    sframe_file = args.sframe_file


    gt_list = []

    print ('Configured options:')

    for video_name in os.listdir(data_path):
        print(video_name) # Folder with the name of the video. Contains the results for this video.

        csv_path = os.path.join(data_path, video_name, 'all_predictions.csv')
        
        gt_list = get_gt_list(csv_path, sframe_file)
        all_apples = read_csv(csv_path)

        
        # assigned_apples is a list with all the apples that have been assigned to a stretch
        # stretch_velocities contains the mean velocity of all the apples that appears in the changing frame
        # Then, the mean velocity for each frame can be known
        assigned_apples, stretch_velocities, id_ass_apples = stretch_velocity(all_apples=all_apples, gt_list=gt_list)

        # Get the list of tuples, where for each stretch change we obtain from which frame to which frame we are at the
        # moment of the change.
        frames_changing = get_change_frames(gt_list=gt_list, stretch_velocities=stretch_velocities)

        print ('Entering function assign_all_apples(). This may take a while ...')

        apples_stretch = assign_all_apples(frames_changing=frames_changing, assigned_apples=assigned_apples,
                                           all_apples=all_apples, id_ass_apples=id_ass_apples,
                                           stretch_velocities=stretch_velocities)

        video_orientation = video_name.split('_')[4] # In the formatted video name, fifth field is the orientation

        # Create folder with the video names to store the results
        path_to_results = os.path.join('results_stretch', video_name)
        os.makedirs(os.path.dirname(path_to_results), exist_ok=True)

        path_to_stretch_num = os.path.join(path_to_results, 'all_apples.csv')
        path_to_all_apples = os.path.join(path_to_results, 'apples_stretch.csv')
        path_to_all_predictions = os.path.join(path_to_results, 'all_predictions.csv')
        path_to_gps = os.path.join(path_to_results, 'frame_coordinates.csv')

        # If the video is shot from west, apply a transformation in order to invert the stretch number, so the stretch
        # 1 will be 21, 2 -> 20, etc...
        if video_orientation == 'w':
            assigned_apples = [[sublist[0], 22 - sublist[1]] for sublist in assigned_apples]
            write_results(assigned_apples, path_to_all_apples, path_to_stretch_num, csv_path, path_to_all_predictions)
        else:
            write_results(apples_stretch, path_to_all_apples, path_to_stretch_num, csv_path, path_to_all_predictions)


if __name__ == "__main__":
    main()
