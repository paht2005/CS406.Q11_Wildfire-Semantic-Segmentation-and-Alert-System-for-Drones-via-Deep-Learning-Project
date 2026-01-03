<p align="center">
  <a href="https://www.uit.edu.vn/" title="University of Information Technology">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="University of Information Technology (UIT)">
  </a>
</p>

<h1 align="center"><b>CS406.Q11 – Image Processing and Applications</b></h1>

---
# **CS406 Course Project: Wildfire Semantic Segmentation and Alert System for UAVs/Drones via Deep Learning**

> This repository contains the implementation of a **Wildfire Detection and Alert System** based on **semantic segmentation using deep learning**, developed for the course **CS406.Q11 – Image Processing and Applications** at the **University of Information Technology (UIT – VNU-HCM)**.
>
> The project leverages **U-Net–based architectures** trained on the **FLAME Dataset** to detect wildfire regions from aerial imagery captured by drones. In addition to pixel-level fire segmentation, the system incorporates **temporal consistency analysis** and a **rule-based alert mechanism** to trigger wildfire warnings in video streams.

---

## **Team Information**
| No. | Student ID | Full Name | Role | Github | Email |
|----:|:----------:|-----------|------|--------|-------|
| 1 | 23520032 | Truong Hoang Thanh An | Member | [Awnpz](https://github.com/Awnpz) | 23520032@gm.uit.edu.vn  |
| 2 | 23520023 | Nguyen Xuan An | Member | [annx-uit](https://github.com/annx-uit) |  23520023@gm.uit.edu.vn  | 
| 3 | 23520213 | Vu Viet Cuong | Member | [Kun05-AI](https://github.com/Kun05-AI) |  23520213@gm.uit.edu.vn  | 
| 4 | 23521143 | Nguyen Cong Phat | Leader | [paht2005](https://github.com/paht2005) | 23521143@gm.uit.edu.vn |

---

## **Table of Contents**
- [Features](#features)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Demo Application](#demo-application)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

---

## **Features**
- **Wildfire Semantic Segmentation** using U-Net with ResNet encoders.
- **Pixel-level fire mask prediction** from aerial RGB images.
- **Processed FLAME Dataset** with train/validation/test splits.
- **Temporal consistency & persistence analysis** for robust detection.
- **Alert scoring mechanism** based on:
  - Fire area ratio
  - Model confidence
  - Detection persistence across frames
- **Real-time demo interface** using **Gradio** for video-based inference.

---

## **Dataset**
### FLAME Dataset
- **Source**: All evaluations are conducted using the FLAME Dataset, published on IEEE DataPort. Dataset access: [FLAME Dataset on IEEE DataPort](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs)
- **Original format**:
  - RGB images (`.jpg`)
  - Binary fire masks (`.png`)
- **Dataset structure in this project**:
  - `dataset-raw/`: Original FLAME directory structure.
  - `flame-dataset/`: Processed dataset split into `train / val / test`.
- Images → RGB aerial images
- Masks → Binary wildfire segmentation masks

---

## **Repository Structure**
```
CS406.Q11_Wildfire-Semantic-Segmentation-and-Alert-System-for-Drones-via-Deep-Learning-Project/
├── dataset-raw/ # Original FLAME dataset structure
│ ├── Images/ # .jpg RGB images
│ └── Masks/ # .png segmentation masks
│
├── flame-dataset/ # Processed dataset (train/val/test)
│ ├── train/
│ │ ├── images/
│ │ └── masks/
│ ├── val/
│ │ ├── images/
│ │ └── masks/
│ └── test/
│ │ ├── images/
│ │ └── masks/
│
├── src/ # Jupyter notebooks (training & evaluation)
│ ├── lightning_logs/ # PyTorch Lightning logs [ignored]
│ ├── split_flame_raw-dataset.ipynb
│ ├── train-models.ipynb
│ └── model-evaluation.ipynb
│
├── models/ # Trained model checkpoints (.ckpt) [ignored]
├── docs/
│ ├── CS406.Q11-Nhom9_slide.pdf
│ └── CS406.Q11-Nhom9_report.pdf
│
├── demo/
│ ├── demo_video.mp4
│ └── demo_gif.gif
│
├── gradio_app.py # Gradio demo application
├── requirements.txt
├── .gitignore
├── LICENSE
├── thumbnail.png
└── README.md

```

## **Methodology**

### 1. Semantic Segmentation Model
- Architecture: **U-Net**
- Encoder: **ResNet34 / ResNet50**
- Frameworks:
  - `segmentation-models-pytorch`
  - `PyTorch Lightning`

### 2. Training Pipeline
- Input size: `512 × 512`
- Loss functions:
  - Binary Cross-Entropy
  - Dice Loss (optional experiments)
- Mixed-precision training (FP16)
- Model checkpointing using `.ckpt` format

### 3. Temporal Fire Detection Logic
To reduce false positives in video streams:
- Fire regions must persist across multiple consecutive frames.
- Alert score is computed using: **Score = w_area × AreaRatio + w_prob × Confidence + w_persist × Persistence**

### 4. Alert Mechanism
- Hysteresis thresholds:
  - **EVENT_ON**: Trigger fire alert
  - **EVENT_OFF**: Reset detection state
- Periodic alert logging with timestamp and dummy GPS metadata.

---

## **Installation**

### 1. Clone repository
```bash
git clone https://github.com/paht2005/CS406.Q11_Wildfire-Semantic-Segmentation-and-Alert-System-for-Drones-via-Deep-Learning-Project.git.git
cd CS406.Q11_Wildfire-Semantic-Segmentation-and-Alert-System-for-Drones-via-Deep-Learning-Project
```

### 2. Create virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate      # Linux / Mac
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

--- 
## **Usage**

### Dataset preprocessing
```bash
jupyter notebook src/split_flame_raw-dataset.ipynb
```
### Model training
```bash
jupyter notebook src/train-models.ipynb
```
### Model evaluation
```bash
jupyter notebook src/model-evaluation.ipynb
```

---
## **Demo Application**
Run the Gradio demo for video-based wildfire detection:
```bash
python gradio_app.py
```
- Upload a drone video
- Visualize wildfire segmentation masks
- View alert metadata in real time

--- 
## **Results**
- Accurate pixel-level wildfire segmentation on the FLAME dataset.
- Stable alert triggering thanks to temporal persistence filtering.
- Demonstrates feasibility of UAV-based wildfire monitoring using deep learning.

> Detailed quantitative results will be reported in the final project report.

---
## **Conclusion**
This project demonstrates the effectiveness of** deep learning–based semantic segmentation** for early wildfire detection using aerial imagery. 
By combining pixel-level fire segmentation with temporal analysis and alert logic, the system provides a robust foundation for real-time wildfire monitoring using UAVs.

--- 
## **License**
This project is for academic use in the course **CS406.Q11 – Image Processing and Applications** at UIT – VNU-HCM.
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
