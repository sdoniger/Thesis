#!/usr/bin/env python
import torch

"""
Script: rename_dla_ckpt.py

Usage:
  1) python rename_dla_ckpt.py
  2) It loads 'ctdet_coco_dla_2x.pth', renames old DLA keys 'module.base.*',
     'module.dla_up.*', 'module.ida_up.*' to new 'img_backbone.*'.
  3) Skips 'hm.*', 'wh.*', 'reg.*', 'module.base.fc.*' (CenterNet detection heads).
  4) Saves the partial checkpoint to 'my_camera_dla_backbone.pth'.
"""

OLD_CHECKPOINT_PATH = 'ctdet_coco_dla_2x.pth'
NEW_CHECKPOINT_PATH = 'my_camera_dla_backbone.pth'

def main():
    # 1) Load old checkpoint
    old_ckpt = torch.load(OLD_CHECKPOINT_PATH, map_location='cpu')
    if 'state_dict' in old_ckpt:
        state_dict_old = old_ckpt['state_dict']
    else:
        state_dict_old = old_ckpt

    new_state_dict = {}

    # 2) Iterate keys in old checkpoint, rename or skip
    for k, v in state_dict_old.items():

        # Skip detection heads from CenterNet
        # typically 'module.hm.*', 'module.wh.*', 'module.reg.*', or no 'module.' prefix
        if k.startswith('module.hm.') or k.startswith('module.wh.') or k.startswith('module.reg.'):
            continue
        if k.startswith('hm.') or k.startswith('wh.') or k.startswith('reg.'):
            continue

        # Also skip any fully-connected classification in base FC if it exists
        if k.startswith('module.base.fc'):
            continue

        # If your old checkpoint has 'module.base.*', rename to 'img_backbone.base.*'
        if k.startswith('module.base.'):
            new_k = k.replace('module.base.', 'img_backbone.base.')

        # If your old checkpoint has 'module.dla_up.*', rename to 'img_backbone.dla_up.*'
        elif k.startswith('module.dla_up.'):
            new_k = k.replace('module.dla_up.', 'img_backbone.dla_up.')

        # If it has 'module.ida_up.*', rename to 'img_backbone.ida_up.*'
        #  (Only do this if your new code actually has 'ida_up' inside your DLA.)
        elif k.startswith('module.ida_up.'):
            new_k = k.replace('module.ida_up.', 'img_backbone.ida_up.')

        else:
            # skip all other leftover keys, e.g. module.ida_0, or something not needed
            continue

        new_state_dict[new_k] = v

    # 3) Overwrite old_ckpt['state_dict'] with our new dictionary
    old_ckpt['state_dict'] = new_state_dict

    # 4) Save partial checkpoint
    torch.save(old_ckpt, NEW_CHECKPOINT_PATH)
    print(f"Created partial checkpoint: {NEW_CHECKPOINT_PATH}")
    print(f"Number of keys in new state_dict: {len(new_state_dict)}")

if __name__ == '__main__':
    main()