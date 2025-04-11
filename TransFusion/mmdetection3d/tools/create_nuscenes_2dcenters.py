import os
import json
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points

def main():
    """
    Creates a JSON file with 2D bounding box info (and center coords) for each sample token.
    The format will be:
    
    {
      "<sample_token>": {
        "CAM_FRONT": [ {"cx": float, "cy": float, "bbox_2d": [...], "category_name": ...}, ... ],
        "CAM_BACK":  [ {...}, ... ],
        ...
      },
      "<another_sample_token>": { ... }
    }

    The code checks if each cam_name is in sample['data'] to avoid KeyError.
    Also skips boxes that are fully behind camera (z <= 0).
    """
    # 1) Choose your dataset version
    # For the full dataset => 'v1.0-trainval'
    # For mini => 'v1.0-mini'
    version = 'v1.0-trainval'

    # 2) Path to your nuScenes data root
    dataroot = 'data/nuscenes'

    # 3) Initialize NuScenes
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    # We'll store a dict: results_2d[sample_token][cam_name] = list of bounding boxes
    results_2d = {}

    # The standard 6 camera names in nuScenes
    cam_list = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
    ]

    # iterate over all samples in the dataset
    for sample in nusc.sample:
        sample_token = sample['token']
        # create sub-dict for each sample token
        results_2d[sample_token] = {}

        # for each camera in the standard camera list
        for cam_name in cam_list:
            # check if that camera is present in sample['data']
            if cam_name not in sample['data']:
                # skip if not in data
                continue
            
            sd_token = sample['data'][cam_name]  # camera sample_data token
            # get boxes in camera coords
            boxes = nusc.get_boxes(sd_token)

            # retrieve camera intrinsics
            sd_rec = nusc.get('sample_data', sd_token)
            cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
            camera_intrinsic = np.array(cs_rec['camera_intrinsic'], dtype=np.float32)

            ann_list = []
            for box in boxes:
                # each box has 8 corners in 3D
                corners_3d = box.corners()  # shape: [3, 8]

                # Project corners onto image
                # Might yield warnings about divide by zero if corners behind camera
                corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True)
                
                # filter out corners behind camera -> z <= 0
                # Or we check corners_3d[2,:] to see if some are negative
                # We'll skip the box if it is entirely behind camera
                # (Alternatively, you can do partial checks)
                z_vals = corners_3d[2, :]
                if (z_vals <= 0).all():
                    # all corners behind camera
                    continue

                x_coords = corners_2d[0, :]
                y_coords = corners_2d[1, :]

                # skip if it yields invalid or all corners behind camera in 2D sense
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                if x_max <= x_min or y_max <= y_min:
                    continue

                cx_2d = 0.5 * (x_min + x_max)
                cy_2_2d = 0.5 * (y_min + y_max)

                # store bounding box info
                ann = {
                    'cx': float(cx_2d),
                    'cy': float(cy_2_2d),
                    'bbox_2d': [float(x_min), float(y_min), float(x_max), float(y_max)],
                    'category_name': box.name
                }
                ann_list.append(ann)

            # store list in results_2d[sample_token][cam_name]
            results_2d[sample_token][cam_name] = ann_list

    # output file path
    out_path = os.path.join(dataroot, 'nuscenes_2dcenters.json')
    with open(out_path, 'w') as f:
        json.dump(results_2d, f)
    print(f"Saved 2D center data => {out_path}")

if __name__ == '__main__':
    main()
