#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dump_predictions_by_token.py

Builds camera + LiDAR dataset dictionaries (token -> set_of_classes)
and prints each token's camera classes vs. LiDAR classes
in a simple console format.

Usage:
    python tools/dump_predictions_by_token.py \
      --camera-config /path/to/fcos3d_config.py \
      --camera-pkl camera_results.pkl \
      --lidar-config /path/to/transfusion_config.py \
      --lidar-pkl lidar_results.pkl
"""

import os
import json
import pickle
import argparse
import numpy as np

from mmcv import Config
from mmdet3d.datasets import build_dataset

def parse_camera_labels_3d(frame_result, dataset):
    """Extract 3D labels from camera result, after you've aligned class orders."""
    class_set = set()
    if 'img_bbox' in frame_result:
        sub_dict = frame_result['img_bbox']
        if 'labels_3d' in sub_dict:
            for lbl in sub_dict['labels_3d']:
                class_name = dataset.CLASSES[int(lbl.item())]
                class_set.add(class_name)
    return class_set

def parse_lidar_labels_3d(frame_result, dataset):
    """Extract 3D labels from LiDAR result, after you've aligned class orders."""
    class_set = set()
    if 'pts_bbox' in frame_result:
        sub_dict = frame_result['pts_bbox']
        if 'labels_3d' in sub_dict:
            for lbl in sub_dict['labels_3d']:
                class_name = dataset.CLASSES[int(lbl.item())]
                class_set.add(class_name)
    return class_set

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera-config', type=str, required=True)
    parser.add_argument('--camera-pkl', type=str, required=True)
    parser.add_argument('--lidar-config', type=str, required=True)
    parser.add_argument('--lidar-pkl', type=str, required=True)
    args = parser.parse_args()

    # Build camera dataset
    cam_cfg = Config.fromfile(args.camera_config)
    camera_dataset = build_dataset(cam_cfg.data.val)
    with open(args.camera_pkl, 'rb') as f:
        camera_preds = pickle.load(f)
    camera_dict = {}
    for idx in range(len(camera_dataset)):
        cset = parse_camera_labels_3d(camera_preds[idx], camera_dataset)
        token = camera_dataset.data_infos[idx].get('token', str(idx))
        if token not in camera_dict:
            camera_dict[token] = set()
        camera_dict[token].update(cset)

    # Build lidar dataset
    lid_cfg = Config.fromfile(args.lidar_config)
    lidar_dataset = build_dataset(lid_cfg.data.val)
    with open(args.lidar_pkl, 'rb') as f:
        lidar_preds = pickle.load(f)
    lidar_dict = {}
    for idx in range(len(lidar_dataset)):
        lset = parse_lidar_labels_3d(lidar_preds[idx], lidar_dataset)
        token = lidar_dataset.data_infos[idx].get('token', str(idx))
        lidar_dict[token] = lset

    # Now get all tokens in common
    common_tokens = sorted(list(set(camera_dict.keys()) & set(lidar_dict.keys())))

    # Print a table-like output:
    print("TOKEN                             | CAMERA CLASSES                        | LIDAR CLASSES")
    print("-----------------------------------------------------------------------------------------")
    for token in common_tokens:
        cam_cls_list = sorted(list(camera_dict[token]))
        lid_cls_list = sorted(list(lidar_dict[token]))
        print(f"{token:<32} | {', '.join(cam_cls_list):<35} | {', '.join(lid_cls_list)}")

if __name__ == '__main__':
    main()
