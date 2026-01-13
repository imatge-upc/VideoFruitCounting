# Video Fruit Counting

## Summary

This repository provides a computer vision-based method for accurately counting fruits (apples) in orchard videos. We leverage the power of YOLOv5 object detection and ByteTrack object tracking to develop a highly efficient and robust solution.

Our method enables continuous and precise tracking of multiple fruits under challenging conditions, including:

* Varying illumination
* Changes in fruit appearance
* Different scanning distances

This solution is invaluable for agricultural applications, providing accurate fruit identification and counting for improved yield estimation and orchard management.

We offer:

* Complete code implementation
* Example annotated videos
* Pre-trained models for easy adaptation to diverse scenarios

## Dataset Description

The provided dataset consists of videos recorded along rows of apple trees oriented North-South. Each side of the trees (East and West) was captured in separate, continuous video sequences.

* **Stretches:** Each tree row is divided into 21 smaller sections, called "stretches."
* **Stretch Markers:** The boundaries between stretches are marked by vertical ribbons positioned between the trees.
* **SFRAME Files:** We provide CSV files (SFRAME files) that list the exact frame numbers where these ribbons appear centered in the video frame, defining the start and end of each stretch.

**Example SFRAME.csv Content:**

```
210928_114406_k_r2_e_015_175_162,2199,2622,2997,3474,3806,4167,4538,4922,5275,5657,6042,6469,6915,7386,7781,8231,8636,9032,9498,9862,10267,10618
210906_112041_k_r2_e_015_175_162,880,1184,1412,1711,1963,2225,2497,2786,3070,3412,3731,4011,4312,4608,4853,5133,5395,5652,5920,6216,6498,6753
...
```

Videos were captured with the cameras in vertical setup to cover the complete height of the trees. To perform detection and tracking, frames must be rotated 90º
counterclokwise.



* Videos were captured using ZED and Azure Kinect cameras.
* Only the RGB (left view) information was utilized.
* Videos are named using the following convention:
```
date_time_cameratype_row_orientation_stretchnumber_distance_height.{svo,mkv,mp4}
```

Where:

* **date_time**: ```YYYYMMDD_HHMMSS```
* **cameratype**: ```z``` (ZED) or ```k``` (Kinect)
* **row**: ```r{1d}``` (row number)
* **orientation**: ```e``` (East) or ```w``` (West)
* **stretchnumber**: ```{%03d}``` (stretch ID, 000-020)
* **distance**: ```{%03d}``` (distance from camera to trees, 125, 175, or 225 cm)
* **height**: ```{%03d}``` (camera height in centimeters)


To improve accessibility, the original SVO and MKV videos were converted to the MP4 (H.264) format, removing the dependency on the ZED and Kinect SDKs. Consequently, results generated with this code may exhibit minor discrepancies compared to those published in the paper.

## Video Capture:

Example: ```210928_080625_z_r1_e_015_225_162.mp4```

## Getting Started
The workflow is divided into two main steps:

1. Object Detection and Tracking (Inference): Running the ```fruit_tracker_simple.py``` script to detect and track fruits in the video frames.
2. Fruit Count Assignment: Using the ```assign_apples.py``` script to parse the tracking results and count the fruits in each stretch.


### Step 1: Inference / Demo (Object Detection and Tracking)

Required Files:

- [Download example videos](https://gofile.me/73Gps/3VE9yy9aX)
- [Download pretrained YOLOv5 model](https://gofile.me/73Gps/clQj1gJhv)
- [Download SFRAME files](https://gofile.me/73Gps/DECp9orob)

Run the fruit_tracker_simple.py script:

```
$ python fruit_tracker_simple.py --help
usage: fruit_tracker_simple.py [-h] [--video VIDEO_NAME] [--segments-file SEGMENTS_FILE] [--weights WEIGHTS] [--offset OFFSET] [--conf CONFIDENCE]
                               [--min_area MIN_AREA] [--data DATA] [--camera CAMERA] [--rotate ROTATE] [--results_file PREDICTIONS_FILENAME]

Perform apple detection and tracking on the provided video

options:
  -h, --help            show this help message and exit
  --video VIDEO_NAME    Video file path.
  --segments-file SEGMENTS_FILE
                        File containing the limits of the segments. For each video, the segment limits are defined at the frame where the segment change line
                        is centered on the frame.
  --weights WEIGHTS     Weights file for the YOLOv5 detection model
  --offset OFFSET       Number of frames to skip at the beginning.
  --conf CONFIDENCE     Minimum detection confidence to be taken into account.
  --min_area MIN_AREA   Min area (in pixels) of the apples to be tracked. If -1, the value will be determinedfrom the seventh field of the video file name
                        (distance).
  --data DATA           dataset.yaml path
  --camera CAMERA       Camera type (ZED or KA). Only necessary for the provided videos.
  --rotate ROTATE       Whether to rotate each frame 90º clockwise. Useful when camera has been situated vertically
  --results_file PREDICTIONS_FILENAME
                        Filename for the output tracking results
```

Example Command:
```
python3 fruit_tracker_simple.py --video data/210928_100624_k_r1_e_015_225_162_rgb.avi --segments data/SFRAME/KA_stretch_sframe.csv --weights data/last.pt --min_area -1 --rotate True --camera KA --data data/apple_segmentation.yaml --results_file results/all_predictions.csv
```

**Output:**

The tracking results will be saved in ```results/VIDEO_NAME/all_predictions.csv```

### Step 2: Fruit Count Assignment
After running fruit_tracker_simple.py, use the assign_apples.py script to count the fruits in each stretch.

Run the ```assign_apples.py``` script:

```
python assign_apples.py --help
usage: assign_apples.py [-h] [--data_path DATA_PATH] [--sframe_file SFRAME_FILE]

Perform apple assignment to stretches

options:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path where the subdirectories generated when executing the fruit_tracker_simple.py file are located.
  --sframe_file SFRAME_FILE
                        Path to the stretch frames "csv" file.
```

Example Command:

```
python3 assign_apples.py  --data_path ./results --sframe_file ./data/SFRAME/KA_stretch_sframe.csv 
```

**Output:**

Two CSV files will be generated:

* all_apples.csv: Maps each fruit to its corresponding stretch.
* apples_stretch.csv: Provides the total fruit count for each stretch.

## Authorship

 
This project is contributed by [GRAP-UdL-AT](http://www.grap.udl.cat/en/index.html), [imatge-upc](https://imatge.upc.edu/) and ,  [UEA-Irta](https://github.com/UEA-Irta).

Please contact authors to report bugs @ marc.felip@udl.cat, @ ramon.morros@upc.edu, @ jordi.gene@irta.cat

## Citation

Please cite our research article or dataset when using our software and/or dataset:

```
@article{XXXXX2026,

   title   = {Video-Based Fruit Detection and Tracking: Effects of Scanning Conditions on Counting and Mapping Accuracy},
   author  = {Felip-Pom{\'e}s, Marc and Gen{\'e}-Mola, Jordi and Arn{\'o}, Jaume and Lordan, Jaume and Ruiz-Hidalgo, Javier and Morros, Josep-Ramon and Net Barn{\'e}s, Francesc and Gregorio, Eduard},
   journal = {Submitted},
   year    = {2026},
}
```

## References

[1] Felip-Pomés M, Gené-Mola J, Arnó J, Lordan J, Ruiz-Hidalgo J, Morros JR, Net-Barnés F, Gregorio E. 2026. Video-Based Fruit Detection and Tracking: Effects of Scanning Conditions on Counting and Mapping Accuracy. Submitted.

### Acknowledgements

This work was partly funded by the Department of Research and Universities of the Generalitat de Catalunya (grants 2017, SGR 646 and 2021 LLAV 00088), by the Spanish Ministry of Science and Innovation /AEI/10.13039/501100011033 / ERDF (grants RTI2018-094222-B-I00 [PAgFRUIT project], PID2021-126648OB-I00 [PAgPROTECT project], PID2024-161868OB-I00 [C3DRUM project], and PID2024-156495OR-C31 [YIELD4CAST]), by the Spanish Ministry of Science and Innovation / AEI/10.13039/501100011033 / European Union NextGeneration / PRTR (grant TED2021-131871B-I00 [DIGIFRUIT project]), and by the Agrotecnio-IRTA Joint Projects Call 2025 (PhotonFruit project). The work of Jordi Gené-Mola was partially supported by the Spanish Ministry of Universities through a Margarita Salas postdoctoral grant funded by the European Union - NextGenerationEU.