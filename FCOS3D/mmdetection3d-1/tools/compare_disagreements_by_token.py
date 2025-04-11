#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
compare_disagreements_by_token.py

This script compares class predictions between:
1) A camera-based FCOS3D config (with 'img_bbox' -> 'labels_3d')
2) A LiDAR-based TransFusion config (with 'pts_bbox' -> 'labels_3d')

We:
- Build each dataset from separate configs
- Load each .pkl result
- For each frame => get sample_token => parse `labels_3d`
- Collect sets of classes (using dataset.CLASSES)
- Compare if they differ => "disagreement"

Usage:
    python tools/compare_disagreements_by_token.py \
      --camera-config /path/to/fcos3d_nusc.py \
      --camera-pkl /path/to/fcos3d_results.pkl \
      --lidar-config /path/to/transfusion_nusc.py \
      --lidar-pkl /path/to/transfusion_results.pkl \
      --out disagreement_frames.json
"""

import os
import json
import pickle
import argparse
import numpy as np

# For older code (pre-mmengine), import config from mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset

def parse_fcos3d_classes(frame_result, dataset):
    """
    Extract predicted classes from a single FCOS3D camera frame.

    We expect:
      frame_result['img_bbox'] = {
        'boxes_3d': CameraInstance3DBoxes(...),
        'scores_3d': tensor(...),
        'labels_3d': tensor([...]),
        'attrs_3d': tensor([...])
      }

    We'll look for 'labels_3d' and map them to class names from dataset.CLASSES.
    """
    class_set = set()
    if 'img_bbox' in frame_result:
        camera_dict = frame_result['img_bbox']
        if 'labels_3d' in camera_dict:
            labels = camera_dict['labels_3d']  # a torch tensor
            # Convert to numpy or just iterate
            for lbl in labels:
                lbl = int(lbl.item())  # or int(lbl)
                class_name = dataset.CLASSES[lbl]
                class_set.add(class_name)
    return class_set

def parse_transfusion_lidar_classes(frame_result, dataset):
    """
    Extract predicted classes from a single TransFusion LiDAR frame.

    We expect:
      frame_result['pts_bbox'] = {
        'boxes_3d': LiDARInstance3DBoxes(...),
        'scores_3d': tensor(...),
        'labels_3d': tensor([...])
      }

    We'll look for 'labels_3d'.
    """
    class_set = set()
    if 'pts_bbox' in frame_result:
        lidar_dict = frame_result['pts_bbox']
        if 'labels_3d' in lidar_dict:
            labels = lidar_dict['labels_3d']
            for lbl in labels:
                lbl = int(lbl.item())
                class_name = dataset.CLASSES[lbl]
                class_set.add(class_name)
    return class_set

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera-config', type=str, required=True,
                        help='Config for FCOS3D dataset (NuScenesMonoDataset).')
    parser.add_argument('--camera-pkl', type=str, required=True,
                        help='Camera .pkl results from fcos3d.')
    parser.add_argument('--lidar-config', type=str, required=True,
                        help='Config for TransFusion dataset (NuScenesDataset).')
    parser.add_argument('--lidar-pkl', type=str, required=True,
                        help='LiDAR .pkl results from transfusion.')
    parser.add_argument('--out', type=str, default='disagreement_frames.json')
    args = parser.parse_args()

    # 1) Build the camera dataset
    cam_cfg = Config.fromfile(args.camera_config)
    camera_dataset = build_dataset(cam_cfg.data.val)
    with open(args.camera_pkl, 'rb') as f:
        camera_preds = pickle.load(f)
    print(f"[INFO] camera_dataset len={len(camera_dataset)}, camera_preds len={len(camera_preds)}")

    # We'll map sample_token -> set_of_classes
    camera_dict = {}
    for idx in range(len(camera_dataset)):
        frame_result = camera_preds[idx]
        class_set = parse_fcos3d_classes(frame_result, camera_dataset)

        # The sample token is stored in data_infos[idx]
        info = camera_dataset.data_infos[idx]
        token = info.get('token', str(idx))
        if token not in camera_dict:
            camera_dict[token] = set()
        camera_dict[token].update(class_set)

    print(f"[INFO] Camera => {len(camera_dict)} unique tokens")

    # 2) Build the LiDAR dataset
    lid_cfg = Config.fromfile(args.lidar_config)
    lidar_dataset = build_dataset(lid_cfg.data.val)
    with open(args.lidar_pkl, 'rb') as f:
        lidar_preds = pickle.load(f)
    print(f"[INFO] lidar_dataset len={len(lidar_dataset)}, lidar_preds len={len(lidar_preds)}")

    # We'll map sample_token -> set_of_classes
    lidar_dict = {}
    for idx in range(len(lidar_dataset)):
        frame_result = lidar_preds[idx]
        class_set = parse_transfusion_lidar_classes(frame_result, lidar_dataset)

        info = lidar_dataset.data_infos[idx]
        token = info.get('token', str(idx))
        lidar_dict[token] = class_set

    print(f"[INFO] LiDAR => {len(lidar_dict)} unique tokens\n")

    # 3) Compare by sample token
    common_tokens = set(camera_dict.keys()) & set(lidar_dict.keys())
    print(f"[INFO] Found {len(common_tokens)} tokens in common.\n")

    disagreements = []
    for token in sorted(common_tokens):
        cam_cls = camera_dict[token]
        lid_cls = lidar_dict[token]
        if cam_cls != lid_cls:
            disagreements.append({
                "sample_token": token,
                "camera_classes": sorted(list(cam_cls)),
                "lidar_classes": sorted(list(lid_cls))
            })

    print(f"[INFO] Found {len(disagreements)} disagreement tokens out of {len(common_tokens)}.")
    with open(args.out, 'w') as f:
        json.dump(disagreements, f, indent=2)
    print(f"[INFO] Disagreements saved to {args.out}")

if __name__ == '__main__':
    main()
