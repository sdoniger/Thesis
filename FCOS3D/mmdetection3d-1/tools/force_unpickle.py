# force_unpickle.py
import sys
import numpy as np
import pickle

# Monkey patch so python thinks `numpy._core` is `numpy.core`
sys.modules['numpy._core'] = np.core

input_path = "/content/drive/MyDrive/Thesis/FCOS3D/mmdetection3d-1/data/nuscenes-mini/nuscenes_infos_val.pkl"
output_path= "/content/drive/MyDrive/Thesis/FCOS3D/mmdetection3d-1/data/nuscenes-mini/nuscenes_infos_val_fixed.pkl"

with open(input_path, 'rb') as f:
    data = pickle.load(f)

print("Successfully loaded data from", input_path)
with open(output_path, 'wb') as f:
    pickle.dump(data, f, protocol=4)

print("Saved fixed file to", output_path)
