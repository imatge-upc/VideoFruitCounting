import os
import re
import cv2
import numpy as np
import pandas as pd
import sys
import argparse

from pyk4a import PyK4APlayback
import pyzed.sl as sl




def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate detection')

    parser.add_argument('--video_path', dest='video_path', help='Path where the video is located')
    parser.add_argument('--data_path', dest='data_path', help='Path to the directory that contains all the csv with the'
                                                              'csv previously generated')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    video_path = args.video_path
    data_path = args.data_path

    all_predictions_file = os.path.join(data_path, 'all_predictions.csv')
    results_all_apples_file = os.path.join(data_path, 'all_apples.csv')

    all_predictions = pd.read_csv(all_predictions_file)
    results_all_apples = pd.read_csv(results_all_apples_file)
    all_predictions['stretch_mode'] = all_predictions['stretch']

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)
        , (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128)
        , (128, 128, 128)]

    if 'KA' in data_path:
        def capture_next_frame(playback, offset_frame):
            capture = playback.get_next_capture()
            offset_frame += 1
            print(offset_frame)
            return capture, offset_frame

        def capture_previous_frame(playback, offset_frame):
            capture = playback.get_previouse_capture()
            offset_frame -= 1
            print(offset_frame)
            return capture, offset_frame

        visualization_scale = 0.5

        for x in range(1, 22):
            print("Stretch no. " + str(x) + ": " + str(
                all_predictions[all_predictions["stretch_mode"] == x]["ID"].unique().shape[0]) + " apples.")

        playback = PyK4APlayback(video_path)
        playback.open()
        print(f"Record length: {playback.length / 1000000: 0.2f} sec")

        # Offset method 2
        offset_frame = all_predictions["NumFrame"][1]
        fps_rate = int(str(playback.configuration["camera_fps"])[8:])
        frame_duration = 1 / int(fps_rate)
        offset_seconds = offset_frame * frame_duration
        offset_microseconds = int(offset_seconds * 1e6)

        if offset_frame != 0.0:
            playback.seek(int(offset_microseconds) - 1)

        key = 115
        print("  Quit the video reading:     q")
        print("  Play de video forward:     d")
        print("  Play de video backward:     a")
        print("  Stop the video:    s")
        print("  Visualize the next frame:     w")
        print("  Visualize the frame before:     x")
        play_forward = 1
        play_backward = 0
        capture = playback.get_next_capture()
        print(capture.color_timestamp_usec)

        # Set up parameters for the video
        # codec = cv2.VideoWriter_fourcc(*"mp4v")
        # framerate = 30
        # resolution = (540,960)
        # filename = "VideoApples.mp4"
        # out = cv2.VideoWriter(filename, codec, framerate, resolution)

        while key != 113:
            try:
                if key == 119:
                    capture, offset_frame = capture_next_frame(playback, offset_frame)
                    play_forward = 0
                    play_backward = 0
                elif key == 120:
                    capture, offset_frame = capture_previous_frame(playback, offset_frame)
                    play_forward = 0
                    play_backward = 0
                elif key == 100:
                    play_forward = 1
                    play_backward = 0
                elif key == 97:
                    play_forward = 0
                    play_backward = 1
                elif key == 115:
                    play_forward = 0
                    play_backward = 0

                if play_forward:
                    capture, offset_frame = capture_next_frame(playback, offset_frame)
                elif play_backward:
                    capture, offset_frame = capture_previous_frame(playback, offset_frame)

                if capture.color is not None:
                    frame_raw = cv2.imdecode(capture.color, cv2.IMREAD_COLOR)
                    frame = cv2.rotate(frame_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    frame = frame.astype(np.uint8)
                    for det_idx in all_predictions[all_predictions["NumFrame"] == offset_frame].index:
                        # if all_predictions.iloc[det_idx]["ID"] in apple_ids:
                        #     continue
                        # det_idx = 0
                        x_min = int(all_predictions.iloc[det_idx]["x_min"])
                        y_min = int(all_predictions.iloc[det_idx]["y_min"])
                        x_max = int(all_predictions.iloc[det_idx]["x_min"] + all_predictions.iloc[det_idx]["h"])
                        y_max = int(all_predictions.iloc[det_idx]["y_min"] + all_predictions.iloc[det_idx]["w"])
                        appleID = int(all_predictions.iloc[det_idx]["ID"])
                        score = all_predictions.iloc[det_idx]["score"]
                        stretch = int(all_predictions.iloc[det_idx]["stretch"])
                        color = colors[appleID % len(colors)]
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
                        # cv2.putText(frame, str(appleID), (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                        #             2)  # Apple ID
                        # cv2.putText(frame, str(round(score, 2)), (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                        #             2)  # Confidence
                        # cv2.putText(frame, str(stretch), (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                        #             2)  # Actual stretch of the camera
                        cv2.putText(frame, str(results_all_apples[results_all_apples["AppleID"] == appleID].iloc[0, 1]),
                                    (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)  # Assigned stretch

                    resized_frame = cv2.resize(frame,
                                               tuple([int(x * visualization_scale) for x in reversed(frame.shape[:2])]),
                                               interpolation=cv2.INTER_AREA)
                    # out.write(resized_frame)
                    cv2.imshow("Color", resized_frame)
                    key = cv2.waitKey(10)
                    # print(capture.color_system_timestamp_nsec)

            except EOFError:
                break
        # out.release()
        cv2.destroyAllWindows()

    else:
        all_predictions = pd.read_csv(all_predictions_file)
        # results_all_apples = pd.read_csv(results_all_apples_file)
        # f = open(apples_json_file)
        # apples = json.load(f)
        all_predictions['stretch_mode'] = all_predictions['stretch']

        offset_frame = all_predictions["NumFrame"][1]

        # Specify SVO path parameter
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(str(video_path))
        init_params.svo_real_time_mode = False  # Don't convert in realtime
        init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use millimeter units (for depth measurements)

        # Create ZED object
        zed = sl.Camera()

        # Open the SVO file specified as a parameter
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            sys.stdout.write(repr(err))
            zed.close()
            exit()

        rt_param = sl.RuntimeParameters()
        rt_param.sensing_mode = sl.SENSING_MODE.FILL

        key = ''
        print("  Quit the video reading:     q")
        print("  Play de video forward:     d")
        print("  Play de video backward:     a")
        print("  Stop the video:    s")
        print("  Visualize the next frame:     w")
        print("  Visualize the frame before:     x")
        print("  Save the frame:     g")
        play_forward = 1
        play_backward = 0

        capture = sl.Mat()
        visualization_scale = 0.5

        offset = offset_frame
        zed.set_svo_position(offset)

        while key != 113:
            try:
                if key == 119:
                    if zed.grab() == sl.ERROR_CODE.SUCCESS:
                        zed.retrieve_image(capture, sl.VIEW.LEFT)
                        offset_frame += 1

                    play_forward = 0
                    play_backward = 0
                elif key == 120:
                    play_forward = 1
                    play_backward = 0
                elif key == 100:  # play forward
                    play_forward = 1
                    play_backward = 0
                elif key == 97:  # play backward
                    play_forward = 0
                    play_backward = 1
                elif key == 115:  # stop
                    play_forward = 0
                    play_backward = 0

                if play_forward:
                    if zed.grab() == sl.ERROR_CODE.SUCCESS:
                        zed.retrieve_image(capture, sl.VIEW.LEFT)
                        offset_frame += 1
                elif play_backward:
                    if zed.grab() == sl.ERROR_CODE.SUCCESS:
                        zed.retrieve_image(capture, sl.VIEW.LEFT, -1)
                        offset_frame -= 1

                if capture is not None:
                    frame_raw = capture.get_data()
                    frame = cv2.rotate(frame_raw, cv2.ROTATE_90_CLOCKWISE)
                    for det_idx in all_predictions[all_predictions["NumFrame"] == offset_frame].index:
                        x_min = int(all_predictions.iloc[det_idx]["x_min"])
                        y_min = int(all_predictions.iloc[det_idx]["y_min"])
                        x_max = int(all_predictions.iloc[det_idx]["x_min"] + all_predictions.iloc[det_idx]["h"])
                        y_max = int(all_predictions.iloc[det_idx]["y_min"] + all_predictions.iloc[det_idx]["w"])
                        appleID = int(all_predictions.iloc[det_idx]["ID"])
                        score = all_predictions.iloc[det_idx]["score"]
                        stretch = int(all_predictions.iloc[det_idx]["stretch"])
                        color = colors[appleID % len(colors)]
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
                        # cv2.putText(frame, str(appleID), (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                        #             2)  # Apple ID
                        # cv2.putText(frame, str(round(score,2)), (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                        #             2)  # Confidence
                        # cv2.putText(frame, str(stretch), (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                        #             2)  # Actual stretch of the camera
                        cv2.putText(frame, str(results_all_apples[results_all_apples["AppleID"] == appleID].iloc[0, 1]),
                                    (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)  # Assigned stretch

                    resized_frame = cv2.resize(frame,
                                               tuple([int(x * visualization_scale) for x in reversed(frame.shape[:2])]),
                                               interpolation=cv2.INTER_AREA)
                    # out.write(resized_frame)
                    cv2.imshow("Color", resized_frame)
                    key = cv2.waitKey(10)
                    # print(capture.color_system_timestamp_nsec)

            except EOFError:
                break
        # out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
