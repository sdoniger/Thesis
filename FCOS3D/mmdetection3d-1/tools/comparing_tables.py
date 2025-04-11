#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
compare_three_models_tables.py (Modified)

Creates three tables:
1) **Table A (per-token predictions)**: token, camera_classes, lidar_classes, fusion_classes
2) **Table B (class-level breakdown)**: how many tokens each model predicts for each class
3) **Table C (detailed disagreements + GT classes)**:
   - Instead of "token", the first column is "gt_classes" (the full set of ground-truth classes),
   - Then camera_classes, lidar_classes, fusion_classes columns are the **entire sets** each predicted, 
     *not* only the "camera_only" or "lidar_only" subsets.
   - We still only include tokens where camera vs. LiDAR differ (cam_set != lid_set).

Usage:
    python compare_three_models_tables.py \
      --camera-config configs/fcos3d/fcos3d_nusc.py \
      --camera-pkl camera.pkl \
      --lidar-config configs/transfusion/transfusion_nusc_voxel_L.py \
      --lidar-pkl lidar.pkl \
      --fusion-config configs/transfusion/transfusion_nusc_fusion.py \
      --fusion-pkl fusion.pkl

Notes:
- This code uses `build_gt_dict()` that reads `gt_labels_3d` from `dataset.get_ann_info(idx)` 
  and maps them to dataset.CLASSES. If your dataset stores ground-truth differently, adapt that function.
- If your CSV viewer shows empty strings as "NaN", we write "none" or "NoGT" as fallback to avoid that.
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
    """Parse fusion model classes (often the same approach as LiDAR)."""
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
    using 'gt_labels_3d' from dataset.get_ann_info(idx).
    Mapping each label index to dataset.CLASSES[lbl].
    """
    gt_dict = {}
    for idx in range(len(dataset)):
        info = dataset.data_infos[idx]
        token = info.get('token', str(idx))

        ann = dataset.get_ann_info(idx)
        if 'gt_labels_3d' in ann:
            labels_3d = ann['gt_labels_3d']
            class_list = []
            for lbl in labels_3d:
                lbl_int = int(lbl.item())
                # map index -> string name
                class_name = dataset.CLASSES[lbl_int]
                class_list.append(class_name)
            gt_set = set(class_list)
        else:
            gt_set = set()

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

    # 2) Build GT dict: token -> set_of_GT_classes
    gt_dict = build_gt_dict(camera_dataset)

    # 3) Table A: Per-token predictions
    all_tokens = sorted(set(camera_dict.keys()) |
                        set(lidar_dict.keys()) |
                        set(fusion_dict.keys()))
    table_a_rows = []
    for token in all_tokens:
        cam_set = sorted(camera_dict.get(token, set()))
        lid_set = sorted(lidar_dict.get(token, set()))
        fus_set = sorted(fusion_dict.get(token, set()))

        cam_str = ','.join(cam_set) if cam_set else 'none'
        lid_str = ','.join(lid_set) if lid_set else 'none'
        fus_str = ','.join(fus_set) if fus_set else 'none'

        row = {
            'token': token,
            'camera_classes': cam_str,
            'lidar_classes': lid_str,
            'fusion_classes': fus_str,
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
    all_classes = camera_dataset.CLASSES
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
    #    We list tokens where camera vs. LiDAR differ, 
    #    but columns are: [gt_classes, camera_classes, lidar_classes, fusion_classes].
    table_c_rows = []
    for token in all_tokens:
        cam_set = camera_dict.get(token, set())
        lid_set = lidar_dict.get(token, set())
        if cam_set != lid_set:
            fus_set = fusion_dict.get(token, set())
            # ground-truth
            gt_set  = gt_dict.get(token, set())

            gt_list     = sorted(gt_set) if gt_set else []
            cam_list    = sorted(cam_set) if cam_set else []
            lid_list    = sorted(lid_set) if lid_set else []
            fus_list    = sorted(fus_set) if fus_set else []

            row = {
                # new first column: the entire GT set, not "NoGT" if empty
                'gt_classes': ','.join(gt_list) if gt_list else 'NoGT',
                'camera_classes': ','.join(cam_list) if cam_list else 'none',
                'lidar_classes': ','.join(lid_list) if lid_list else 'none',
                'fusion_classes': ','.join(fus_list) if fus_list else 'none'
            }
            table_c_rows.append(row)

    table_c_csv = args.out_prefix + "_tableC_disagreement.csv"
    with open(table_c_csv, 'w', newline='') as f:
        fieldnames = [
            'gt_classes',
            'camera_classes',
            'lidar_classes',
            'fusion_classes'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in table_c_rows:
            writer.writerow(row)
    print(f"[INFO] Wrote Table C (disagreement + GT, entire camera/lidar/fusion sets) to {table_c_csv}")

    print("[INFO] Done. Generated three CSV tables with 'gt_classes' as first column in Table C.")

if __name__ == '__main__':
    main()
