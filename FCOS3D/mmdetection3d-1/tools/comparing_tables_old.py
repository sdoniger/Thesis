#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
compare_three_models_tables.py (Modified)

Creates three tables:
1) Table A (per-token predictions): token, camera_classes, lidar_classes, fusion_classes
2) Table B (class-level breakdown): how many tokens each model predicts for each class
3) Table C (detailed disagreements + GT classes):
   - Instead of "token", we now place "gt_classes" as the first column,
     plus camera_only_classes, lidar_only_classes, common_classes, fusion_classes.
   - This shows how camera vs. LiDAR differ and how the fusion resolves it,
     with knowledge of what GT classes exist in that token.

Usage:
    python compare_three_models_tables.py \
      --camera-config configs/fcos3d/fcos3d_nusc.py \
      --camera-pkl camera.pkl \
      --lidar-config configs/transfusion/transfusion_nusc_voxel_L.py \
      --lidar-pkl lidar.pkl \
      --fusion-config configs/transfusion/transfusion_nusc_fusion.py \
      --fusion-pkl fusion.pkl

Note:
- This assumes your dataset has `info['gt_names']` 
  listing ground-truth classes for each token. If not, adapt `build_gt_dict()`.
"""

import argparse
import csv
import pickle
import sys

import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset


def parse_camera_labels_3d(frame_result, dataset):
    """Parse camera-only model classes from 'img_bbox' -> 'labels_3d'."""
    class_set = set()
    if isinstance(frame_result, dict) and 'img_bbox' in frame_result:
        sub_dict = frame_result['img_bbox']
        if 'labels_3d' in sub_dict:
            for lbl in sub_dict['labels_3d']:
                lbl = int(lbl.item()) if hasattr(lbl, 'item') else int(lbl)
                class_name = dataset.CLASSES[lbl]
                class_set.add(class_name)
    return class_set

def parse_lidar_labels_3d(frame_result, dataset):
    """Parse LiDAR-only model classes from 'pts_bbox' -> 'labels_3d'."""
    class_set = set()
    if isinstance(frame_result, dict) and 'pts_bbox' in frame_result:
        sub_dict = frame_result['pts_bbox']
        if 'labels_3d' in sub_dict:
            for lbl in sub_dict['labels_3d']:
                lbl = int(lbl.item()) if hasattr(lbl, 'item') else int(lbl)
                class_name = dataset.CLASSES[lbl]
                class_set.add(class_name)
    return class_set

def parse_fusion_labels_3d(frame_result, dataset):
    """Parse fusion model classes. 
       If fusion also uses 'pts_bbox', do the same parse as LiDAR."""
    class_set = set()
    if isinstance(frame_result, dict) and 'pts_bbox' in frame_result:
        sub_dict = frame_result['pts_bbox']
        if 'labels_3d' in sub_dict:
            for lbl in sub_dict['labels_3d']:
                lbl = int(lbl.item()) if hasattr(lbl, 'item') else int(lbl)
                class_name = dataset.CLASSES[lbl]
                class_set.add(class_name)
    return class_set


def build_dict(config_path, pkl_path, parse_func):
    """
    1) Build dataset from config_path
    2) Load the predictions from pkl_path
    3) For each index in dataset, parse -> set_of_classes
    4) Return dict: token -> set_of_classes, plus the dataset
    """
    cfg = Config.fromfile(config_path)
    dataset = build_dataset(cfg.data.val)  # or .test if results are for test
    with open(pkl_path, 'rb') as f:
        preds = pickle.load(f)

    if len(preds) != len(dataset):
        print(f"[WARN] preds len={len(preds)} != dataset len={len(dataset)}", file=sys.stderr)

    result_dict = {}
    for idx in range(len(dataset)):
        frame_res = preds[idx]
        class_set = parse_func(frame_res, dataset)
        info = dataset.data_infos[idx]
        token = info.get('token', str(idx))
        result_dict[token] = class_set
    return result_dict, dataset


def build_gt_dict(dataset):
    """
    Build a dictionary: token -> set_of_GT_classes
    using 'gt_labels_3d' from ann. This matches the keys found in dataset.get_ann_info(idx).

    Steps:
      1) For each index in dataset, get ann = dataset.get_ann_info(idx).
      2) ann['gt_labels_3d'] is a tensor of label indices for each 3D box.
      3) Convert each index to a string class name via dataset.CLASSES.
      4) Store that set in gt_dict[token].
    """
    gt_dict = {}
    for idx in range(len(dataset)):
        info = dataset.data_infos[idx]
        token = info.get('token', str(idx))

        ann = dataset.get_ann_info(idx)
        if 'gt_labels_3d' in ann:
            labels_3d = ann['gt_labels_3d']  # e.g. tensor([0,2,4,...]) shape (N,)
            class_list = []
            for lbl in labels_3d:
                lbl_int = int(lbl.item())
                class_name = dataset.CLASSES[lbl_int]
                class_list.append(class_name)
            gt_set = set(class_list)
        else:
            gt_set = set()  # If there's no gt_labels_3d, we store empty set

        gt_dict[token] = gt_set

    return gt_dict



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera-config', type=str, required=True)
    parser.add_argument('--camera-pkl', type=str, required=True)
    parser.add_argument('--lidar-config', type=str, required=True)
    parser.add_argument('--lidar-pkl', type=str, required=True)
    parser.add_argument('--fusion-config', type=str, required=True)
    parser.add_argument('--fusion-pkl', type=str, required=True)
    parser.add_argument('--out-prefix', type=str, default='three_models_compare',
                        help="Prefix for the CSV tables.")
    args = parser.parse_args()

    # 1) Build camera, lidar, fusion dictionaries
    camera_dict, camera_dataset = build_dict(
        args.camera_config, args.camera_pkl, parse_camera_labels_3d
    )
    lidar_dict, lidar_dataset   = build_dict(
        args.lidar_config, args.lidar_pkl, parse_lidar_labels_3d
    )
    fusion_dict, fusion_dataset = build_dict(
        args.fusion_config, args.fusion_pkl, parse_fusion_labels_3d
    )

    # We'll assume all share the same dataset -> same # classes
    all_classes = camera_dataset.CLASSES

    # 2) Build GT dict: token -> set_of_GT_classes
    gt_dict = build_gt_dict(camera_dataset)  
    # (We use camera_dataset but they should be the same as lidar/fusion anyway.)

    # 3) Table A: Per-token
    all_tokens = sorted(set(camera_dict.keys()) |
                        set(lidar_dict.keys()) |
                        set(fusion_dict.keys()))
    table_a_rows = []
    for token in all_tokens:
        cam_cls = sorted(list(camera_dict.get(token, set())))
        lid_cls = sorted(list(lidar_dict.get(token, set())))
        fus_cls = sorted(list(fusion_dict.get(token, set())))
        row = {
            'token': token,
            'camera_classes': ','.join(cam_cls),
            'lidar_classes': ','.join(lid_cls),
            'fusion_classes': ','.join(fus_cls),
        }
        table_a_rows.append(row)

    table_a_csv = args.out_prefix + "_tableA_per_token.csv"
    with open(table_a_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['token','camera_classes','lidar_classes','fusion_classes']
        )
        writer.writeheader()
        for row in table_a_rows:
            writer.writerow(row)
    print(f"[INFO] Wrote Table A (per-token predictions) to {table_a_csv}")

    # 4) Table B: Class-level breakdown
    class_breakdown = []
    for c in all_classes:
        cam_count = 0
        lid_count = 0
        fus_count = 0
        for token in all_tokens:
            if c in camera_dict.get(token, set()):
                cam_count += 1
            if c in lidar_dict.get(token, set()):
                lid_count += 1
            if c in fusion_dict.get(token, set()):
                fus_count += 1
        class_breakdown.append({
            'class_name': c,
            'camera_token_count': cam_count,
            'lidar_token_count': lid_count,
            'fusion_token_count': fus_count
        })

    table_b_csv = args.out_prefix + "_tableB_class_breakdown.csv"
    with open(table_b_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['class_name','camera_token_count','lidar_token_count','fusion_token_count']
        )
        writer.writeheader()
        for row in class_breakdown:
            writer.writerow(row)
    print(f"[INFO] Wrote Table B (class-level breakdown) to {table_b_csv}")

    # 5) Table C: Detailed Disagreement + GT classes
    #    Instead of 'token' as first column, we do 'gt_classes'.
    table_c_rows = []
    for token in all_tokens:
        cam_set = camera_dict.get(token, set())
        lid_set = lidar_dict.get(token, set())
        if cam_set != lid_set:
            fus_set = fusion_dict.get(token, set())

            # ground-truth classes
            gt_set = gt_dict.get(token, set())
            camera_only = sorted(list(cam_set - lid_set))
            lidar_only  = sorted(list(lid_set - cam_set))
            common      = sorted(list(cam_set & lid_set))
            fusion_list = sorted(list(fus_set))

            row = {
                # 'token': token,   # remove the old 'token' column
                'gt_classes': ','.join(sorted(gt_set)),  # new first column
                'camera_only_classes': ','.join(camera_only),
                'lidar_only_classes': ','.join(lidar_only),
                'common_classes': ','.join(common),
                'fusion_classes': ','.join(fusion_list)
            }
            table_c_rows.append(row)

    table_c_csv = args.out_prefix + "_tableC_disagreement.csv"
    with open(table_c_csv, 'w', newline='') as f:
        fieldnames = [
            # 'token', 
            'gt_classes',
            'camera_only_classes',
            'lidar_only_classes',
            'common_classes',
            'fusion_classes'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in table_c_rows:
            writer.writerow(row)
    print(f"[INFO] Wrote Table C (disagreement + GT classes) to {table_c_csv}")

    print("[INFO] Done. Generated three CSV tables with the new 'gt_classes' as first col in Table C.")


if __name__ == '__main__':
    main()
