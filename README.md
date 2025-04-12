# Multimodal Sensor Fusion: Transformer-Based 3D Object Detection Research

This repository contains experimental work building on two prominent 3D object detection frameworks: [TransFusion](https://github.com/XuyangBai/TransFusion) and [FCOS3D](https://github.com/open-mmlab/mmdetection3d). The goal is to explore multimodal fusion and custom configurations for LiDAR-only, camera-only, and fusion-based object detection in autonomous driving scenarios.

## 🔧 Repository Structure

```
Thesis/
├── FCOS3D/           # Modified version of mmdetection3d (based on FCOS3D)
├── TransFusion/      # Modified version of TransFusion (camera-only and fusion variants)
├── experiments/      # Custom scripts, configs, and logs
├── README.md
└── .gitignore
```

## 🧪 Experimental Branches

- \`main\`: Base repo with original subtree integration
- \`camera-only\`: TransFusion modified for monocular 3D detection
- \`lidar-fusion\`: Custom multimodal fusion modifications and evaluations

## 📄 Setup Instructions

1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/sdoniger/Thesis.git
   cd Thesis
   \`\`\`

2. Install dependencies using Conda: 
3. Train or test using a modified branch:
   \`\`\`bash
   git checkout camera-only
   bash experiments/train_camera.sh
   \`\`\`

## 📊 Results

*Images.* Results compare performance across sensor modalities, including:
- LiDAR-only
- Camera-only
- Sensor fusion

## 🤖 Acknowledgments

Some of the debugging strategies, Git integration workflows, and custom optimization scripts were assisted by [ChatGPT](https://chat.openai.com), particularly for:
- Resolving subtree conflicts
- Branching strategies for experimental workflows
- Script and configuration cleanup across FCOS3D and TransFusion modules

This support was instrumental in streamlining the development process and ensuring reproducibility of results.

## 📚 References

- Xuyang Bai et al. "TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection." (CVPR 2022)
- MMDetection3D: https://github.com/open-mmlab/mmdetection3d
- FCOS3D: https://github.com/open-mmlab/mmdetection3d/tree/dev-1.x/projects/FCOS3D
