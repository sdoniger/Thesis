#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
compare_triplet.py

This script:
1) Builds three datasets from camera, LiDAR, and fusion configs, so we can
   retrieve each sample's "token".
2) Loads the .pkl results for each model (camera, lidar, fusion).
3) Parses each frame's predictions into a set of classes for that token.
4) Computes partial merges stats, class-level adoption rates, IoU, etc.
5) Prints out the results.

Usage:
  python compare_triplet.py \
    --camera-config /path/to/fcos3d_config.py --camera-pkl /path/to/camera_results.pkl \
    --lidar-config /path/to/transfusion_config.py --lidar-pkl /path/to/lidar_results.pkl \
    --fusion-config /path/to/transfusion_fusion_config.py --fusion-pkl /path/to/fusion_results.pkl

Ensure you have mmcv, mmdet3d, etc. installed in your 'oldpy' environment.
"""

import argparse
import pickle
import sys

import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset

def parse_camera_labels_3d(frame_result, dataset):
    """
    Example parse function for a camera model that stores 'img_bbox' with 'labels_3d'.
    If your .pkl is structured differently, adapt accordingly.
    """
    class_set = set()
    if isinstance(frame_result, dict) and 'img_bbox' in frame_result:
        sub_dict = frame_result['img_bbox']
        if 'labels_3d' in sub_dict:
            for lbl in sub_dict['labels_3d']:
                lbl_int = int(lbl.item()) if hasattr(lbl, 'item') else int(lbl)
                class_name = dataset.CLASSES[lbl_int]
                class_set.add(class_name)
    elif isinstance(frame_result, list):
        # some older results might be list-of-lists shape
        # adapt if needed
        for class_id, arr in enumerate(frame_result):
            if len(arr) > 0:
                class_name = dataset.CLASSES[class_id]
                class_set.add(class_name)
    return class_set

def parse_lidar_labels_3d(frame_result, dataset):
    """
    Example parse function for a LiDAR model that stores 'pts_bbox' with 'labels_3d'.
    If your .pkl is structured differently, adapt accordingly.
    """
    class_set = set()
    if isinstance(frame_result, dict) and 'pts_bbox' in frame_result:
        sub_dict = frame_result['pts_bbox']
        if 'labels_3d' in sub_dict:
            for lbl in sub_dict['labels_3d']:
                lbl_int = int(lbl.item()) if hasattr(lbl, 'item') else int(lbl)
                class_name = dataset.CLASSES[lbl_int]
                class_set.add(class_name)
    return class_set

def compute_fusion_stats(camera_dict, lidar_dict, fusion_dict):
    """
    Compute partial merges stats, class-level adoption, IoU, etc.
    Returns a dictionary summarizing them.
    """

    tokens = sorted(set(camera_dict.keys()) & set(lidar_dict.keys()) & set(fusion_dict.keys()))
    n_data = len(tokens)

    # partial merges counts
    count_fusion_includes_cam_only = 0
    count_fusion_includes_lid_only = 0
    count_fusion_includes_both_cam_lid_only = 0
    count_fusion_ignores_both_cam_lid_only = 0

    # jaccard
    jaccard_cam_list = []
    jaccard_lid_list = []

    # class adoption counters for camera vs. fusion, and lidar vs. fusion
    adoption_camera = {}
    adoption_lidar  = {}

    def ensure_class(dct, c):
        if c not in dct:
            dct[c] = {"model_has": 0, "fusion_has_too": 0}

    for token in tokens:
        cam_set = camera_dict[token]
        lid_set = lidar_dict[token]
        fus_set = fusion_dict[token]

        # camera-only, lidar-only
        cam_only = cam_set - lid_set
        lid_only = lid_set - cam_set

        # does fusion pick any camera-only or lidar-only classes?
        fusion_cam_only = len(fus_set & cam_only) > 0
        fusion_lid_only = len(fus_set & lid_only) > 0

        if fusion_cam_only and fusion_lid_only:
            count_fusion_includes_both_cam_lid_only += 1
        elif fusion_cam_only:
            count_fusion_includes_cam_only += 1
        elif fusion_lid_only:
            count_fusion_includes_lid_only += 1
        else:
            count_fusion_ignores_both_cam_lid_only += 1

        # jaccard/IoU with camera
        inter_cam = len(fus_set & cam_set)
        union_cam = len(fus_set | cam_set)
        j_cam = inter_cam / union_cam if union_cam else 0
        jaccard_cam_list.append(j_cam)

        # jaccard/IoU with lidar
        inter_lid = len(fus_set & lid_set)
        union_lid = len(fus_set | lid_set)
        j_lid = inter_lid / union_lid if union_lid else 0
        jaccard_lid_list.append(j_lid)

        # class-level adoption
        # camera
        for c in cam_set:
            ensure_class(adoption_camera, c)
            adoption_camera[c]["model_has"] += 1
            if c in fus_set:
                adoption_camera[c]["fusion_has_too"] += 1
        # lidar
        for c in lid_set:
            ensure_class(adoption_lidar, c)
            adoption_lidar[c]["model_has"] += 1
            if c in fus_set:
                adoption_lidar[c]["fusion_has_too"] += 1

    partial_stats = {
        "n_data": n_data,
        "fusion_includes_cam_only_count": count_fusion_includes_cam_only,
        "fusion_includes_lid_only_count": count_fusion_includes_lid_only,
        "fusion_includes_both_cam_lid_only_count": count_fusion_includes_both_cam_lid_only,
        "fusion_ignores_both_cam_lid_only_count": count_fusion_ignores_both_cam_lid_only
    }
    mean_j_cam = sum(jaccard_cam_list)/n_data if n_data>0 else 0
    mean_j_lid = sum(jaccard_lid_list)/n_data if n_data>0 else 0
    jaccard_stats = {
        "mean_jaccard_fusion_camera": mean_j_cam,
        "mean_jaccard_fusion_lidar":  mean_j_lid
    }

    # compute class-level adoption fraction
    class_adoption_camera = {}
    for c, vals in adoption_camera.items():
        model_has = vals["model_has"]
        fus_has   = vals["fusion_has_too"]
        frac = fus_has / model_has if model_has else 0
        class_adoption_camera[c] = frac

    class_adoption_lidar = {}
    for c, vals in adoption_lidar.items():
        model_has = vals["model_has"]
        fus_has   = vals["fusion_has_too"]
        frac = fus_has / model_has if model_has else 0
        class_adoption_lidar[c] = frac

    return {
        "partial_stats": partial_stats,
        "jaccard_stats": jaccard_stats,
        "camera_class_adoption": class_adoption_camera,
        "lidar_class_adoption": class_adoption_lidar
    }

def build_dict_from_results(cfg_path, pkl_path, parse_func):
    """
    1) Load config to build dataset
    2) Load .pkl predictions
    3) For i in range(len(dataset)), map dataset.data_infos[i]['token'] -> set_of_classes
    4) Return that dictionary
    """
    cfg = Config.fromfile(cfg_path)
    dataset = build_dataset(cfg.data.val)  # or .test if your results are for test
    with open(pkl_path,'rb') as f:
        preds = pickle.load(f)

    n_data = len(dataset)
    if len(preds) != n_data:
        print(f"WARNING: preds len={len(preds)} != dataset len={n_data}", file=sys.stderr)

    result_dict = {}
    for i in range(n_data):
        token = dataset.data_infos[i].get('token', f'frame_{i}')
        frame_result = preds[i]
        classes = parse_func(frame_result, dataset)
        result_dict[token] = classes
    return result_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera-config', type=str, required=True)
    parser.add_argument('--camera-pkl', type=str, required=True)
    parser.add_argument('--lidar-config', type=str, required=True)
    parser.add_argument('--lidar-pkl', type=str, required=True)
    parser.add_argument('--fusion-config', type=str, required=True)
    parser.add_argument('--fusion-pkl', type=str, required=True)
    args = parser.parse_args()

    # 1) Build camera dict
    print("[INFO] Building camera_dict ...")
    camera_dict = build_dict_from_results(args.camera_config, args.camera_pkl, parse_camera_labels_3d)

    # 2) Build lidar dict
    print("[INFO] Building lidar_dict ...")
    lidar_dict  = build_dict_from_results(args.lidar_config, args.lidar_pkl, parse_lidar_labels_3d)

    # 3) Build fusion dict
    # If your fusion model also uses 'pts_bbox', parse_lidar_labels_3d is correct.
    # If it uses 'img_bbox', you might do parse_camera_labels_3d. Adjust as needed:
    print("[INFO] Building fusion_dict ...")
    fusion_dict = build_dict_from_results(args.fusion_config, args.fusion_pkl, parse_lidar_labels_3d)

    # 4) Compute stats
    stats = compute_fusion_stats(camera_dict, lidar_dict, fusion_dict)

    # 5) Print results
    print("===== PARTIAL STATS =====")
    print(stats["partial_stats"])

    print("\n===== JACCARD STATS =====")
    print(stats["jaccard_stats"])

    print("\n===== CAMERA CLASS ADOPTION (P(fusion picks c | camera picks c)) =====")
    camera_adoption = stats["camera_class_adoption"]
    for c, frac in sorted(camera_adoption.items()):
        print(f"{c}: {frac:.2f}")

    print("\n===== LIDAR CLASS ADOPTION (P(fusion picks c | lidar picks c)) =====")
    lidar_adoption = stats["lidar_class_adoption"]
    for c, frac in sorted(lidar_adoption.items()):
        print(f"{c}: {frac:.2f}")

if __name__ == "__main__":
    main()
