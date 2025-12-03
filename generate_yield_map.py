import rasterio
import cv2
import os
import csv
import numpy as np
import sys

from functions.utils import get_video_combinations, get_gnss_ref, adapt_prow, get_max_min_csv, assign_color
import argparse
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate detection')

    parser.add_argument('--image', dest='image', help='Georeferenced image of the farm.')
    parser.add_argument('--analyze_path', dest='analyze_path',
                        help='Path to the results previously generated with "assign_apples.py" that you want to '
                             'analyze. You must specify the type of execution. E.g. KA/GT, ZED/GT or ZED/GPS, do '
                             'not pass the path to all results.')
    parser.add_argument('--shp_file', dest='shp_file',
                        help='Path of the "shp" file corresponding to the terrain where the videos have been obtained.',
                        required=False)

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    image = args.image
    all_videos_results = args.analyze_path
    shp_file = args.shp_file

    combinations = get_video_combinations(all_videos_results)

    if len(combinations) == 0:
        print ('The provided path contains no valid results, Exiting ...')
        sys.exit()
        
    img = cv2.imread(image)
    for key, values in combinations.items():
        if len(values) < 8:
            continue
        for video, results in values.items():
            row = int(video[1])
            orientation = video[-1]

            stretch_results_path = os.path.join(all_videos_results, results[1], 'apples_stretch.csv')

            maximum, minimum = get_max_min_csv(stretch_results_path)

            # Color legend
            color_bar = np.zeros((100, 20, 3), dtype=np.uint8)
            for i in range(color_bar.shape[0]):
                value = i / (color_bar.shape[0] - 1) * (maximum - minimum) + minimum
                color = assign_color(value, maximum, minimum)
                color_bar[i] = color

            with open(stretch_results_path, 'r') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip the first row

                for csv_row in csv_reader:
                    actual_stretch = int(csv_row[0])
                    total_apples = int(csv_row[1])

                    Prow1_transformed, R, T, Prow = get_gnss_ref(shp_file, row)
                    final_prow = adapt_prow(Prow)

                    if orientation == "e":
                        initial_coord_x, initial_coord_y = final_prow[actual_stretch - 1][0], \
                        final_prow[actual_stretch - 1][1]
                        final_coord_x, final_coord_y = final_prow[actual_stretch][0], final_prow[actual_stretch][1]
                    else:
                        initial_coord_x, initial_coord_y = final_prow[actual_stretch][0], final_prow[actual_stretch][1]
                        final_coord_x, final_coord_y = final_prow[actual_stretch - 1][0], \
                        final_prow[actual_stretch - 1][1]

                    with rasterio.open(image) as dataset:
                        x1, y1 = dataset.index(initial_coord_y, initial_coord_x)
                        x2, y2 = dataset.index(final_coord_y, final_coord_x)

                    if orientation == "w":
                        x1, y1, x2, y2 = x1 + 2, y1 - 2, x2 + 2, y2 - 2
                    else:
                        x1, y1, x2, y2 = x1 - 2, y1 + 2, x2 - 2, y2 + 2

                    color = assign_color(total_apples, maximum, minimum)
                    img = cv2.line(img, (y1, x1), (y2, x2), color, 4)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')

        # Create a ScalarMappable with the same colormap and normalization
        sm = ScalarMappable(cmap='Reds', norm=plt.Normalize(minimum, maximum))

        # Add the color bar
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Total Apples')

        # Save the figure
        plt.savefig(os.path.join('yield_maps', key[0] + '_' + key[1] + '.png'))


if __name__ == "__main__":
    main()
