# KA_ZED_Video_Fruit_Counting

## Summary
We provide a method of fruit counting. To achieve this, YOLOv5 object detection and ByteTrack object tracking technologies are used to develop a highly efficient method.

By implementing this method, we are able to continuously and accurately track multiple fruits under different conditions, such as illumination changes, variations in fruit appearance, and scanning distance. This solution is able to effectively identify and count fruit, which is invaluable for agricultural applications.

This repository provides the code and detailed documentation on how to use this fruit counting solution. In addition, we provide examples of pre-trained datasets and trained models to facilitate implementation and adaptation to different scenarios.

The provided dataset has been recorded over lines of trees (oriented N/S). The lines are oriented N/S. Each side of the trees (E and W) is captured using a continuous video.

Each line is divided into 21 smaller sections (stretches) each. The separation between stretches is marked using a vertical ribbon situated between the trees. To mark each stretch in the video sequences,
two methods can be used:
 a) SGNSS mode: Using the GNSS coordinates of the ribbon. This requires precise geolocalization of the ribbons (using RTK equipement or similar). The location of the ribbons is given using a .shp file, generated using ????
    This file is used in combination with a tracking file, which contains the geolocation of the camera during the recording (sampled at xxx Hz). This info is stored in a .txt file, with the same base name as the video sequence.

Example:
'''
    $GNGGA,080625.60,4137.07186,N,00052.24083,E,4,12,0.57,246.6,M,49.7,M,0.6,0012*6D,2021-09-28 08:06:25.980797
    $GNGGA,080625.70,4137.07186,N,00052.24083,E,4,12,0.57,246.6,M,49.7,M,0.7,0012*6D,2021-09-28 08:06:26.047875
    ...
'''

    In our examples, this mode is only used for the ZED camera, as the .

 b) SFRAME mode: Using a .csv file containing the frame numbers where the ribbons appear exactly at the center of the image.  

Example
'''
    210928_114406_k_r2_e_015_175_162,2199,2622,2997,3474,3806,4167,4538,4922,5275,5657,6042,6469,6915,7386,7781,8231,8636,9032,9498,9862,10267,10618
    210906_112041_k_r2_e_015_175_162,880,1184,1412,1711,1963,2225,2497,2786,3070,3412,3731,4011,4312,4608,4853,5133,5395,5652,5920,6216,6498,6753
    ...
'''


The videos are named using the following convention:
'''
date_time_cameratype_row_orientation_stretchnumber_distance_???.{svo,mkv}
'''

distance: YYYMMDD_HHMMSS
camera type: z (ZED) or k (Kinect). p???
row number: r{1d}
orientation: e (east) or w (west)
stretch id: {%03d} (from 000 to 020)
distance: {%03d}. Distance from camera to line of trees. In our recordings, videos have been recorded at 125, 175 and 225 cm.
???: {%03d}

Example: 210928_080625_z_r1_e_015_225_162.svo





## Getting started
The code is divided into two parts. The first part involves running a file to perform inference on a convolutional neural network (CNN). This inference will allow the detection and tracking of fruits in each of the video frames. As a result, a CSV file will be generated containing detailed information about each tracked apple. The second step parses this file to count the total number of apples in each section of the plantation.


### Inference / Demo - 1st step
To run FruitTracker_inference.py in SFRAME mode you can download the following files:
- [Download KA (Azure Kinect) or ZED (ZED STEREOLABS) video](https://gofile.me/73Gps/3VE9yy9aX)
- [Download the YOLOv5 pretrained model](https://gofile.me/73Gps/clQj1gJhv)
- [Download SFRAME files](https://gofile.me/73Gps/DECp9orob)
<br/>

- [Download geographic data of the orchard](https://gofile.me/73Gps/Z1zOg5u27)
<br/>

Then run fruit_tracker_simple.py:
```
python3 fruit_tracker_simple.py \
  --video $VIDEO_PATH \
  --segments 
  --shp_file $SHP_PATH_IN_CASE_OF_SGNSS \
  --offset $FRAMES_TO_SKIP \
  --conf $MINIMUM_DETECTION_CONFIDENCE \
  --tracker $TRACKER_TYPE
```

Example:
```
python3 fruit_tracker_simple.py --video data/210928_100624_k_r1_e_015_225_162_rgb.avi --segments data/SFRAME/KA_stretch_sframe.csv --weights=data/last.pt --min_area -1 --rotate True --camera KA --data data/apple_segmentation.yaml --results_file results/all_predictions.csv
```

The results will be stored in the following path:<br/>
```
results/all_predictions.csv*
```

### Inference / Demo - 2nd step

At this point, the *fruit_tracker_simple.py* file has already been run, so we have the results of the video saved in *all_predictions.csv*. The script 'assign_apples.py' parses the tracking file from the first step to count the total number of apples in each section of the plantation.


```
python3 assign_apples.py \
  --data_path $PATH_TO_PREVIOUS_RESULTS_PARENT_DIR \
  --gt_file $PATH_TO_SFRAME_FILE
```

Two '.csv' files are generated:

- all_apples.csv: Indicates in which section of the farm each fruit is located.
- apples_stretch.csv: Indicates the total number of fruits in each section.

Example:
```
python3 assign_apples.py --data_path "../FruitTracking/results/results_SFRAME/ZED" --gt_file "ZED_stretch_SFRAME.csv"
```

### Final step

Finally, if desired, a yield map of the farm can be generated with the results. To do this you will need a georeferenced image ('tif' format). You can download the image of the orchard at [this link](https://gofile.me/73Gps/Ebj1i2A6i).

Then run assign_apples/generate_yield_map.py:
```
python3 generate_yield_map.py \
  --image $PATH_TO_TIF_IMAGE \
  --analyze_path $PATH_TO_DIRECORY_OF_ALL_VIDEOS_RESULTS \
  --shp_file $SHP_PATH_IN_CASE_OF_GNSS 
```

Example:
``` 
python3 generate_yield_map.py --image "ORCHARD_IMAGE.tif" --analyze_path "assign_apples/results_stretch/ZED/SGNSS" --shp_file "Trams escaneig_20220104_102114/2021-9 PomersStory/Trams escaneig.shp"
```

Then, in *assign_apples/yield_maps* the yield map image will be saved:
![Yield Maps](./demo/z_175.png)
<br/>

By executing the file *assign_apples/visualize_results.py* you can see the result of the detection and tracking, to do this you must execute it as follows:
```
python3 visualize_results.py \
  --video_path $PATH_TO_VIDEO \
  --data_path $PATH_TO_VIDEO_DIRECOTRY_RESULTS_AFTER_2ND_STEP
```

Example:
```
python3 visualize_results.py --video_path "210928_114406_k_r2_e_015_175_162.mkv" --data_path "assign_apples/results_stretch/KA/SFRAME/210928_114406_k_r2_e_015_175_162"
```


By executing this file, you will be able to visualize the results in this way:

![Detection/Tracking - Results](./demo/example_video.GIF)














