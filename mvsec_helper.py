import os
import cv2
import yaml
import numpy as np
from pathlib import Path

import torch

# Indices to select for each sequence depending on the split to remove take-off and landing parts
SEQUENCES_FRAMES = {'indoor_flying': {'split1': {'indoor_flying1': (140, 1201), # (140, 1200) VALIDATION, TEST  # 200 + 861
                                                 'indoor_flying2': (160, 1580), # (160, 1580) TRAIN             # |
                                                 'indoor_flying3': (125, 1815), # (125, 1815) TRAIN             # | 3110
                                                 },
                                      'split2': {'indoor_flying1': (80, 1260),  # (80, 1260) TRAIN              # |
                                                 'indoor_flying2': (120, 1421), # (120, 1420) VALIDATION, TEST  # 200 + 1101
                                                 'indoor_flying3': (125, 1815), # (125, 1815) TRAIN             # | 2870
                                                 },
                                      'split3': {'indoor_flying1': (80, 1260),  # (80, 1260) TRAIN              # |
                                                 'indoor_flying2': (160, 1580), # (160, 1580) TRAIN             # | 2600
                                                 'indoor_flying3': (73, 1616),  # (73, 1615) VALIDATION, TEST   # 200 + 1343
                                                 }
                                      }
                    }

DISPARITY_MULTIPLIER = 7.0
TIME_BETWEEN_EXAMPLES = 0.05  # sec == 50ms
# TIME_BETWEEN_EXAMPLES = 1
EXPERIMENTS = {
    'indoor_flying': [1, 2, 3],
    # 'indoor_flying': [1, 2, 3, 4],
    # 'outdoor_day': [1, 2],
    # 'outdoor_night': [1, 2, 3]
}
# Focal length multiplied by baseline [pix * meter].
FOCAL_LENGTH_X_BASELINE = {
    'indoor_flying': 19.941772,
    'outdoor_night': 19.651191,
    'outdoor_day': 19.635287
}
INVALID_DISPARITY = 0
DISPARITY_MAXIMUM = 255     # Assume that disparities > 255 are unknown disparities; depth < 0.5m = 50cm
IMAGE_WIDTH = 346
IMAGE_HEIGHT = 260
LOCATION = ['left', 'right']

def get_calibration_info(scenario: str, calib_dir: Path):
    assert scenario in ['indoor_flying', 'outdoor_day', 'outdoor_night'], f'Unknown scenario: {scenario}'
    assert os.path.exists(calib_dir), f'Calibration directory does not exist: {calib_dir}'
    
    # Load calibration data from .yaml
    yaml_path = calib_dir + '/camchain-imucam-' + scenario + '.yaml'
    with open(yaml_path, 'r') as yaml_file:
        calib = yaml.safe_load(yaml_file)
        
    # Load retification maps
    rect_map = {}
    for loc in ['left', 'right']:
        rect_map[loc] = {}
        for axis in ['x', 'y']:
            map_path = calib_dir + f'/{scenario}_{loc}_{axis}_map.txt'
            with open(map_path, 'r') as f:
                rect_map[loc][axis] = np.loadtxt(f)
                
    return calib, rect_map

def get_rectification_map(intrinsics_extrinsics):
    """Produces tables that map rectified coordinates to distorted coordinates.

       x_distorted = rectified_to_distorted_x[y_rectified, x_rectified]
       y_distorted = rectified_to_distorted_y[y_rectified, x_rectified]
    """
    dist_coeffs = intrinsics_extrinsics['distortion_coeffs']
    D = np.array(dist_coeffs)

    intrinsics = intrinsics_extrinsics['intrinsics']
    K = np.array([[intrinsics[0], 0., intrinsics[2]],
                  [0., intrinsics[1], intrinsics[3]], [0., 0., 1.]])
    K_new = np.array(intrinsics_extrinsics['projection_matrix'])[0:3, 0:3]

    R = np.array(intrinsics_extrinsics['rectification_matrix'])

    size = (intrinsics_extrinsics['resolution'][0],
            intrinsics_extrinsics['resolution'][1])

    rectified_to_distorted_x, rectified_to_distorted_y = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, K_new, size, cv2.CV_32FC1)

    return rectified_to_distorted_x, rectified_to_distorted_y

def rectify_events(events, distorted_to_rectified_x, distorted_to_rectified_y, image_size):
    rectified_events = []
    width, height = image_size
    for event in events:
        x, y, timestamp, polarity = event
        x = int(x)
        y = int(y)
        x_rectified = round(distorted_to_rectified_x[y, x])
        y_rectified = round(distorted_to_rectified_y[y, x])
        if (0 <= x_rectified < width) and (0 <= y_rectified < height):
            rectified_events.append(
                [timestamp, x_rectified, y_rectified, polarity])
            
    return rectified_events

def depth2disparity(depth_image, focal_length_x_baseline):
    disparity_image = np.round(DISPARITY_MULTIPLIER * np.abs(focal_length_x_baseline) / (depth_image + 1e-15))
    invalid = np.isnan(disparity_image) | (disparity_image == float('inf')) | (disparity_image >= DISPARITY_MAXIMUM)
    disparity_image[invalid] = INVALID_DISPARITY
    # output = (disparity_image * 256).astype(np.uint16)
    output = disparity_image.astype(np.uint8)
    return output

