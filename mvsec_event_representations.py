import numpy as np
from models import EffWNet
import torch
import h5py
import hdf5plugin
import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
from mvsec_helper import *
from event_representations import events_to_EvRep


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/root/data/MVSEC', type=str)
    parser.add_argument('--scenario', default='indoor_flying', type=str)
    args = parser.parse_args()
    
    # Get calibartion and retification informations
    calib_dir = args.data_dir + '/' + args.scenario + '/' + args.scenario + '_calib'
    calib, rect_map = get_calibration_info(args.scenario, calib_dir)
    focal_length_x_baseline = calib['cam1']['projection_matrix'][0][3]
    
    for exp in EXPERIMENTS[args.scenario]:
        
        data_path = args.data_dir + '/' + args.scenario + '/' + args.scenario + str(exp) + '_data.hdf5'
        gt_path = args.data_dir + '/' + args.scenario + '/' + args.scenario + str(exp) + '_gt.hdf5'
        save_dir = args.data_dir + '/' + args.scenario + '/' + args.scenario + str(exp)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        assert Path(data_path).exists()
        assert Path(gt_path).exists()
        assert Path(save_dir).is_dir()
        
        # Load the files
        data = h5py.File(data_path, 'r')
        gt = h5py.File(gt_path, 'r')
        
        events = {}
        images = {}
        event_timestamps = {}
        image_timestamps = {}
        for loc in LOCATION:
            # Get the events
            events[loc] = np.array(data['davis'][loc]['events'])        # EVENTS: X Y TIME POLARITY
            event_timestamps[loc] = events[loc][:,2].tolist()
            # Get the images
            if args.scenario == 'outdoor_day':
                # A hardware failure caused the grayscale images on the right DAVIS 
                # grayscale images for this scene to be corrupted. 
                # However, VI-Sensor grayscale images are available
                images[loc] = np.array(data['visensor'][loc]['image_raw'])
                image_timestamps[loc] = np.array(data['visensor'][loc]['image_raw_ts']).tolist()
            else:
                images[loc] = np.array(data['davis'][loc]['image_raw'])
                image_timestamps[loc] = np.array(data['davis'][loc]['image_raw_ts']).tolist()
        # Get the depth gt
        depth_gt = np.array(gt['davis']['left']['depth_image_rect'])
        sync_timestamps = np.array(gt['davis']['left']['depth_image_rect_ts']).tolist()
        sync_timestamps_right = np.array(gt['davis']['right']['depth_image_rect_ts']).tolist()
        assert np.all(np.array(sync_timestamps) == np.array(sync_timestamps_right))
        assert depth_gt.shape[0] == len(sync_timestamps)
        
        evreps = {}
        
        # Rectifying events and images
        # Iterate for left & right cameras
        for cam, loc in zip(['cam0', 'cam1'], LOCATION):
            # 'cam0': left camera, 'cam1': right camera
            if args.scenario == 'outdoor_day' and cam == 'cam0':
                cam = 'cam2'
            elif args.scenario == 'outdoor_day' and cam == 'cam1':
                cam = 'cam3'
            rectified_to_distorted_x, rectified_to_distorted_y = get_rectification_map(calib[cam])
            image_size = calib[cam]['resolution']
            
            for is_testing_on_exp in [True, False]:
                if exp == 1:
                    split = 1 if is_testing_on_exp else 2
                elif exp == 2:
                    split = 2 if is_testing_on_exp else 3
                elif exp == 3:
                    split = 3 if is_testing_on_exp else 1
                else:
                    raise ValueError(f'Unknown experiment number: {exp}')
                
                # Iterate for depth timestamps
                event_idx_t = 0
                image_idx_t = 0
                loc_evreps = []
                
                first_index, last_index = SEQUENCES_FRAMES[args.scenario][f'split{split}'][args.scenario + str(exp)]
                print(f'Processing frames from {first_index} to {last_index} for {loc} camera events.')
                
                for synchronization_index, end_timestamp in tqdm(enumerate(sync_timestamps), total=len(sync_timestamps), desc=(f'Split {split}: ' + args.scenario + str(exp) + f'/{loc}')):
                    if synchronization_index < first_index or synchronization_index > last_index:
                        continue
                    # else:
                        # print(f'Processing frame {synchronization_index} for {loc} camera events.')
                        # exit()
                    file_name = str(synchronization_index).zfill(6)    # e.g., 000002, 000004, ...
                    start_timestamp = end_timestamp - TIME_BETWEEN_EXAMPLES
                    
                    # Get synchronized events
                    for t in range(event_idx_t, len(event_timestamps[loc]), 1):
                        # Find event start index
                        event_timestamp = events[loc][t,2]
                        if event_timestamp >= start_timestamp:
                            event_start_index = t
                            event_idx_t = t
                            break
                    for t in range(event_idx_t, len(event_timestamps[loc]), 1):
                        # Find event end index
                        event_timestamp = events[loc][t,2]
                        if event_timestamp > end_timestamp:
                            event_end_index = t
                            event_idx_t = t
                            break
                    synchronized_events = events[loc][event_start_index: event_end_index]
                    # print(event_end_index - event_start_index)
                    # exit()
                    
                    # Events rectification
                    rectified_synchronized_events = np.array(rectify_events(synchronized_events, rect_map[loc]['x'], rect_map[loc]['y'], image_size))
                    
                    num_events = rectified_synchronized_events.shape[0]
                    timestamps = rectified_synchronized_events[:, 0].astype('float64')
                    assert np.min(timestamps) == timestamps[0]
                    assert np.max(timestamps) == timestamps[-1]
                    timestamps = timestamps - timestamps[0]
                    timestamps = timestamps / timestamps[-1]
                    x = rectified_synchronized_events[:, 1].astype('int32')
                    y = rectified_synchronized_events[:, 2].astype('int32')
                    pol = rectified_synchronized_events[:, 3]
                    # pol = (pol + 1.) / 2. # Convert polarity from {-1, 1} to {0, 1}
                    
                    # Get synchronized raw image
                    time_diff = np.finfo(np.float64).max
                    for t in range(image_idx_t, len(image_timestamps[loc]), 1):
                        if np.abs(end_timestamp - image_timestamps[loc][t]) <= time_diff:
                            time_diff = np.abs(end_timestamp - image_timestamps[loc][t])
                        else:
                            # Find synchronized image index
                            image_sync_index = t - 1
                            image_idx_t = t
                            break
                    synchronized_image = images[loc][image_sync_index]
                    
                    # Image rectification
                    rectified_synchronized_image = cv2.remap(synchronized_image, rectified_to_distorted_x, rectified_to_distorted_y, cv2.INTER_LINEAR)
                    
                    image_resolution = rectified_synchronized_image.shape[::-1]  # (width, height)
                    
                    # # Convert depth map to disparity map
                    # depth_gt[synchronization_index][np.isnan(depth_gt[synchronization_index])] = 0.
                    
                    # # Save depth map as .png
                    # save_depth_gt_dir = save_dir + '/depth_gt'
                    # min_depth_gt = np.min(depth_gt[synchronization_index])
                    # max_depth_gt = np.max(depth_gt[synchronization_index])
                    # depth_gt_normalized = (255 * (depth_gt[synchronization_index] - min_depth_gt) / (max_depth_gt - min_depth_gt)).astype(np.uint8)

                    # disp_gt = depth2disparity(depth_gt[synchronization_index], focal_length_x_baseline)
                    
                    # Generate event representation
                    evrep = events_to_EvRep(x, y, timestamps, pol, image_resolution, pol_between_0_and_1=False)
                    loc_evreps.append(evrep)
                
                # evreps[loc] = np.array(loc_evreps)
                evreps[loc] = torch.from_numpy(np.array(loc_evreps))
                # Save EvRep as .npy
                evrep_save_dir = save_dir + f'/evrep_{"test" if is_testing_on_exp else "train"}/'
                if not os.path.exists(evrep_save_dir):
                    os.makedirs(evrep_save_dir)
                # np.save(evrep_save_dir + f'/evrep_{loc}.npy', evreps[loc])
                torch.save(evreps[loc], evrep_save_dir + f'/evrep_{loc}.pt')
                
                # print(f'EvRep representation for {args.scenario + str(exp) + f"/{loc}"} generated and saved for Split {split}.')
            
        print(f'{args.scenario + str(exp)} done.')