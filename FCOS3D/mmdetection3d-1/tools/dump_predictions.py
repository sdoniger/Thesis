#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dump_predictions.py

A script that loads camera + LiDAR .pkl result files (produced by `tools/test.py --out`),
then prints the entire content for each frame. This helps you see exactly what bounding
boxes or dictionary structure is present. For large files, it can be very verbose.

Usage:
    python tools/dump_predictions.py \
      --camera-pkl /path/to/camera_results.pkl \
      --lidar-pkl /path/to/lidar_results.pkl \
      --max-frames 10

"""

import pickle
import argparse
import numpy as np

def short_array_repr(arr, limit=5):
    """
    Utility to show only first few rows of a NumPy array to avoid huge spam.
    """
    if arr.shape[0] > limit:
        head = arr[:limit]
        return f"{head}... (truncated, shape={arr.shape})"
    else:
        return str(arr)

def print_camera_preds(camera_preds, max_frames):
    """
    Dump the entire structure of camera predictions up to `max_frames`.
    """
    total = len(camera_preds)
    print(f"\n[INFO] Camera predictions total frames={total}")
    for i in range(min(total, max_frames)):
        print(f"\n=== CAMERA Frame {i} ===")
        frame_result = camera_preds[i]
        if isinstance(frame_result, dict):
            print("type(frame_result)=dict, keys=", frame_result.keys())
            for k,v in frame_result.items():
                if isinstance(v, list):
                    print(f"  key='{k}', list of length={len(v)}")
                    # possibly each sub-list is bounding boxes
                    for class_id, subarr in enumerate(v):
                        if hasattr(subarr, 'shape'):
                            short_repr = short_array_repr(subarr)
                            print(f"    class_id={class_id}, shape={subarr.shape}, data={short_repr}")
                        else:
                            print(f"    class_id={class_id}, not array, type={type(subarr)}, val={subarr}")
                elif hasattr(v, 'shape'):
                    # a single Nx(...) array
                    short_repr = short_array_repr(v)
                    print(f"  key='{k}', array shape={v.shape}, data={short_repr}")
                else:
                    print(f"  key='{k}', type={type(v)}, val={v}")
        elif isinstance(frame_result, list):
            print(f"type(frame_result)=list, length={len(frame_result)}")
            for class_id, subarr in enumerate(frame_result):
                if hasattr(subarr, 'shape'):
                    short_repr = short_array_repr(subarr)
                    print(f"  class_id={class_id}, shape={subarr.shape}, data={short_repr}")
                else:
                    print(f"  class_id={class_id}, type={type(subarr)}, val={subarr}")
        elif hasattr(frame_result, 'shape'):
            # Nx(...) array
            short_repr = short_array_repr(frame_result)
            print(f"single array shape={frame_result.shape}, data={short_repr}")
        else:
            print(f"Frame result type={type(frame_result)}, val={frame_result}")


def print_lidar_preds(lidar_preds, max_frames):
    """
    Similar approach for LiDAR predictions.
    """
    total = len(lidar_preds)
    print(f"\n[INFO] LiDAR predictions total frames={total}")
    for i in range(min(total, max_frames)):
        print(f"\n=== LIDAR Frame {i} ===")
        frame_result = lidar_preds[i]
        if isinstance(frame_result, dict):
            print("type(frame_result)=dict, keys=", frame_result.keys())
            for k,v in frame_result.items():
                if isinstance(v, list):
                    print(f"  key='{k}', list length={len(v)}")
                    for class_id, subarr in enumerate(v):
                        if hasattr(subarr, 'shape'):
                            short_repr = short_array_repr(subarr)
                            print(f"    class_id={class_id}, shape={subarr.shape}, data={short_repr}")
                        else:
                            print(f"    class_id={class_id}, type={type(subarr)}, val={subarr}")
                elif hasattr(v, 'shape'):
                    short_repr = short_array_repr(v)
                    print(f"  key='{k}', array shape={v.shape}, data={short_repr}")
                else:
                    print(f"  key='{k}', type={type(v)}, val={v}")
        elif isinstance(frame_result, list):
            print(f"type(frame_result)=list, length={len(frame_result)}")
            for class_id, subarr in enumerate(frame_result):
                if hasattr(subarr, 'shape'):
                    short_repr = short_array_repr(subarr)
                    print(f"  class_id={class_id}, shape={subarr.shape}, data={short_repr}")
                else:
                    print(f"  class_id={class_id}, type={type(subarr)}, val={subarr}")
        elif hasattr(frame_result, 'shape'):
            short_repr = short_array_repr(frame_result)
            print(f"single array shape={frame_result.shape}, data={short_repr}")
        else:
            print(f"Frame result type={type(frame_result)}, val={frame_result}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera-pkl', type=str, required=True,
                        help='Camera model .pkl results')
    parser.add_argument('--lidar-pkl', type=str, required=True,
                        help='LiDAR model .pkl results')
    parser.add_argument('--max-frames', type=int, default=10,
                        help='How many frames to print for each predictions file')
    args = parser.parse_args()

    # Load
    with open(args.camera_pkl, 'rb') as f:
        camera_preds = pickle.load(f)
    with open(args.lidar_pkl, 'rb') as f:
        lidar_preds = pickle.load(f)

    # Print camera predictions
    print_camera_preds(camera_preds, args.max_frames)

    # Print LiDAR predictions
    print_lidar_preds(lidar_preds, args.max_frames)

    print("\n[INFO] Done printing predictions. If you need more frames, increase --max-frames.\n")


if __name__ == '__main__':
    main()
