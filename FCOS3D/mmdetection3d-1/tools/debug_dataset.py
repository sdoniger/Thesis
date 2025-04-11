import sys
from mmcv import Config
from mmdet3d.datasets import build_dataset

# Example snippet for an interactive environment or a short script
def debug_dataset_ann_info(cfg_path, index=0):
    # 1) Load config
    cfg = Config.fromfile(cfg_path)
    # 2) Build dataset (e.g., val set)
    dataset = build_dataset(cfg.data.val)

    # 3) Grab annotation info for a single index (default=0)
    ann = dataset.get_ann_info(index)

    # 4) Print the keys + entire annotation dict
    print("keys =", ann.keys())
    print("ann =", ann)

    for i in range(5):  # check first 5 samples
        ann = dataset.get_ann_info(i)
        print(f"Sample {i} ann keys:", ann.keys())
        if 'gt_labels_3d' in ann:
            print(f"Sample {i}, gt_labels_3d shape=", ann['gt_labels_3d'].shape)
            print("gt_labels_3d =", ann['gt_labels_3d'])
        else:
            print("No gt_labels_3d found at all!")

if __name__ == "__main__":
    # Example usage if you put this in a file named debug_dataset.py:
    #
    #   python debug_dataset.py /path/to/your_config.py 0
    #
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str,
                        help="Path to your model/dataset config.")
    parser.add_argument("index", type=int, nargs="?", default=0,
                        help="Which data index to inspect. Default=0.")
    args = parser.parse_args()

    debug_dataset_ann_info(args.config_path, args.index)
