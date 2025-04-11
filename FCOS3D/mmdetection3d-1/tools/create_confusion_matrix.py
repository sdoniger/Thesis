#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
create_confusion_matrix.py

Generates confusion matrices for camera-only, LiDAR-only, and fusion models
using the real 3D IoU from mmdet3d.core.bbox.iou_calculators.iou3d_calculator.

Usage:
  python create_confusion_matrix.py \
    --camera-config /path/to/fcos3d_config.py \
    --camera-pkl /path/to/camera_results.pkl \
    --lidar-config /path/to/transfusion_lidar_config.py \
    --lidar-pkl /path/to/lidar_results.pkl \
    --fusion-config /path/to/transfusion_fusion_config.py \
    --fusion-pkl /path/to/fusion_results.pkl \
    --iou-thresh 0.5 \
    --coordinate lidar

Requires:
- mmdet3d >= 0.17 or similar, with iou3d_calculator present
- A consistent bounding box format Nx7: [x,y,z,dx,dy,dz,rot] 
- Ground-truth stored in data_info['gt_boxes_3d'].tensor, data_info['gt_names']

Output:
- Prints three confusion matrices for camera, LiDAR, and fusion, each as CSV lines.
- Columns = predicted classes plus background
- Rows = GT classes plus background
"""

import argparse
import sys
import pickle
import numpy as np

import torch

import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset
# This is the official 3D IoU calculator
from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import BboxOverlaps3D


def parse_pred_boxes(frame_result, dataset, is_camera=False, is_lidar=False):
    """
    Extract predicted bounding boxes in Nx7 tensor:
    [x, y, z, dx, dy, dz, rot], plus a predicted class_name
    from the .pkl result structure.

    - is_camera => parse 'img_bbox' with 'boxes_3d' & 'labels_3d'
    - is_lidar/fusion => parse 'pts_bbox' with 'boxes_3d' & 'labels_3d'

    Returns a list of dict:
      {
        'bbox_3d': (torch.Tensor) shape (7,),
        'class_name': str
      }
    """
    results = []

    if is_camera:
        # Typically 'img_bbox' with 'boxes_3d', 'labels_3d'
        if isinstance(frame_result, dict) and ('img_bbox' in frame_result):
            sub_dict = frame_result['img_bbox']
            if 'boxes_3d' in sub_dict and 'labels_3d' in sub_dict:
                boxes_3d = sub_dict['boxes_3d']
                labels_3d = sub_dict['labels_3d']
                for i in range(len(labels_3d)):
                    lbl_int = int(labels_3d[i].item())
                    cls_name = dataset.CLASSES[lbl_int]
                    boxvals = boxes_3d.tensor[i]  # shape (7, ) if Nx7
                    results.append({
                        'bbox_3d': boxvals,  # a 7-dim tensor
                        'class_name': cls_name
                    })
    else:
        # LiDAR or fusion => 'pts_bbox'
        if isinstance(frame_result, dict) and ('pts_bbox' in frame_result):
            sub_dict = frame_result['pts_bbox']
            if 'boxes_3d' in sub_dict and 'labels_3d' in sub_dict:
                boxes_3d = sub_dict['boxes_3d']
                labels_3d = sub_dict['labels_3d']
                for i in range(len(labels_3d)):
                    lbl_int = int(labels_3d[i].item())
                    cls_name = dataset.CLASSES[lbl_int]
                    boxvals = boxes_3d.tensor[i]  # shape (7,) if Nx7
                    results.append({
                        'bbox_3d': boxvals,
                        'class_name': cls_name
                    })

    return results


def parse_gt_boxes(data_info, dataset):
    """
    Extract ground-truth boxes in Nx7 plus GT class names from data_info.
    We assume data_info['gt_boxes_3d'].tensor = Nx7,
              data_info['gt_names'] = list of strings
    Returns a list of dict:
      {
        'bbox_3d': (torch.Tensor) shape (7,),
        'class_name': str
      }
    If your dataset structure is different, adapt here.
    """
    if 'gt_boxes_3d' in data_info and 'gt_names' in data_info:
        gt_boxes_3d = data_info['gt_boxes_3d']
        gt_names    = data_info['gt_names']
        results = []
        for i in range(len(gt_names)):
            cls_name = gt_names[i]
            boxvals = gt_boxes_3d.tensor[i]  # shape(7,)
            results.append({
                'bbox_3d': boxvals,
                'class_name': cls_name
            })
        return results
    else:
        return []


def build_confmat_for_model(cfg_path, pkl_path, model_name,
                            coordinate='lidar', iou_thresh=0.5,
                            is_cam=False, is_lid=False):
    """
    1) Build dataset
    2) Load .pkl
    3) For each frame, parse predicted boxes -> Nx7
    4) Parse GT boxes -> Mx7
    5) Use BboxOverlaps3D to get IoU matrix
    6) Do naive 'best match' approach
    7) Fill confusion matrix
    Return: (cm, class_names)

    confusion matrix shape = (Nclasses+1, Nclasses+1),
    last row/col => 'background' for unmatched GT or unmatched pred.
    """
    print(f"[INFO] Building dataset for {model_name} from {cfg_path}")
    cfg = Config.fromfile(cfg_path)
    dataset = build_dataset(cfg.data.val)  # or test
    class_names = dataset.CLASSES
    n_class = len(class_names)
    
    # We'll create (n_class+1)x(n_class+1) confusion matrix
    cm = np.zeros((n_class+1, n_class+1), dtype=int)

    # Helper: map class -> index
    def class_to_idx(cls):
        if cls in class_names:
            return class_names.index(cls)
        return n_class  # background index if unknown

    # The official 3D IoU calculator from mmdet3d
    iou_calc = BboxOverlaps3D(coordinate=coordinate)

    print(f"[INFO] Loading predictions from {pkl_path}")
    with open(pkl_path, 'rb') as f:
        preds = pickle.load(f)
    if len(preds)!=len(dataset):
        print(f"[WARN] {model_name}: preds len={len(preds)} != dataset len={len(dataset)}")

    # For each frame
    for i in range(len(dataset)):
        if i>=len(preds):
            break

        frame_result = preds[i]
        info = dataset.data_infos[i]

        # parse predicted boxes
        pred_boxes_list = parse_pred_boxes(frame_result, dataset,
                                           is_camera=is_cam, is_lidar=is_lid)
        # parse GT
        gt_boxes_list   = parse_gt_boxes(info, dataset)

        # Convert them to torch Tensors of shape (N, 7)
        # plus store class indices
        if len(pred_boxes_list)>0:
            pred_bboxes_3d = []
            pred_cls_idx   = []
            for p in pred_boxes_list:
                pred_bboxes_3d.append(p['bbox_3d'].unsqueeze(0))  # shape(1,7)
                pred_cls_idx.append(class_to_idx(p['class_name']))
            pred_bboxes_3d = torch.cat(pred_bboxes_3d, dim=0)  # shape(N,7)
            pred_cls_idx   = np.array(pred_cls_idx, dtype=int)
        else:
            pred_bboxes_3d = torch.zeros((0,7))
            pred_cls_idx   = np.zeros((0,), dtype=int)

        if len(gt_boxes_list)>0:
            gt_bboxes_3d = []
            gt_cls_idx   = []
            for g in gt_boxes_list:
                gt_bboxes_3d.append(g['bbox_3d'].unsqueeze(0))
                gt_cls_idx.append(class_to_idx(g['class_name']))
            gt_bboxes_3d = torch.cat(gt_bboxes_3d, dim=0)  # shape(M,7)
            gt_cls_idx   = np.array(gt_cls_idx, dtype=int)
        else:
            gt_bboxes_3d = torch.zeros((0,7))
            gt_cls_idx   = np.zeros((0,), dtype=int)

        N = pred_bboxes_3d.shape[0]
        M = gt_bboxes_3d.shape[0]
        
        # If no preds or no GT, handle quickly
        if N==0 and M==0:
            continue
        elif N==0:
            # all GT unmatched => background col
            for g_idx in range(M):
                cm[gt_cls_idx[g_idx], n_class]+=1  # col=background
            continue
        elif M==0:
            # all preds unmatched => background row
            for p_idx in range(N):
                cm[n_class, pred_cls_idx[p_idx]]+=1
            continue

        # Compute IoU matrix: shape (Npred, Mgt)
        # BboxOverlaps3D expects shape (N,7) and (M,7), output shape (N,M)
        ious = iou_calc(pred_bboxes_3d, gt_bboxes_3d, mode='iou')  # shape(N,M)

        # For naive matching: for each pred, pick the best gt if iou>=threshold
        matched_gt = set()
        matched_pred = set()
        for p_idx in range(N):
            row_iou = ious[p_idx,:]  # shape(M,)
            best_val, best_g = torch.max(row_iou, dim=0)
            if best_val>=iou_thresh:
                # check if best_g not matched yet
                if best_g.item() not in matched_gt:
                    matched_gt.add(best_g.item())
                    matched_pred.add(p_idx)
                    # increment cm
                    g_cls_i = gt_cls_idx[best_g.item()]
                    p_cls_i = pred_cls_idx[p_idx]
                    cm[g_cls_i, p_cls_i]+=1
        
        # unmatched GT => predicted background
        for g_idx in range(M):
            if g_idx not in matched_gt:
                cm[gt_cls_idx[g_idx], n_class]+=1
        
        # unmatched preds => gt background
        for p_idx in range(N):
            if p_idx not in matched_pred:
                cm[n_class, pred_cls_idx[p_idx]]+=1

    return cm, class_names


def print_confmat_as_csv(cm, class_names, model_name):
    """
    Print confusion matrix as CSV lines:
    Row labels = GT classes + 'background'
    Column labels = predicted classes + 'background'
    """
    print(f"\n=== Confusion Matrix for {model_name} ===")
    row_labels = list(class_names) + ["background"]
    col_labels = list(class_names) + ["background"]

    # Print header row
    print("GT\\Pred," + ",".join(col_labels))

    # Each row
    for r_idx, r_lbl in enumerate(row_labels):
        row_vals = [str(cm[r_idx,c_idx]) for c_idx in range(len(col_labels))]
        print(f"{r_lbl}," + ",".join(row_vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera-config', required=True)
    parser.add_argument('--camera-pkl', required=True)
    parser.add_argument('--lidar-config', required=True)
    parser.add_argument('--lidar-pkl', required=True)
    parser.add_argument('--fusion-config', required=True)
    parser.add_argument('--fusion-pkl', required=True)
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                        help="IoU threshold for bounding box matching")
    parser.add_argument('--coordinate', type=str, default='lidar',
                        choices=['camera','lidar','depth'],
                        help="Coordinate system for 3D IoU. Usually 'lidar' or 'camera'.")
    args = parser.parse_args()

    # 1) Camera confusion matrix
    cm_cam, class_names = build_confmat_for_model(
        args.camera_config, args.camera_pkl, 
        model_name="Camera",
        coordinate=args.coordinate,
        iou_thresh=args.iou_thresh,
        is_cam=True, is_lid=False)
    print_confmat_as_csv(cm_cam, class_names, "Camera")

    # 2) LiDAR confusion matrix
    cm_lid, _ = build_confmat_for_model(
        args.lidar_config, args.lidar_pkl,
        model_name="LiDAR",
        coordinate=args.coordinate,
        iou_thresh=args.iou_thresh,
        is_cam=False, is_lid=True)
    print_confmat_as_csv(cm_lid, class_names, "LiDAR")

    # 3) Fusion confusion matrix
    cm_fus, _ = build_confmat_for_model(
        args.fusion_config, args.fusion_pkl,
        model_name="Fusion",
        coordinate=args.coordinate,
        iou_thresh=args.iou_thresh,
        is_cam=False, is_lid=False)
    print_confmat_as_csv(cm_fus, class_names, "Fusion")


if __name__ == "__main__":
    main()
