
# Self-Driving Car Simulation Using CNN (CVI620 Final Project – Winter 2025)

This repository contains the code, data, and documentation for our CVI620 final project, which involves building a deep learning model using Convolutional Neural Networks (CNNs) to control a simulated self-driving car in the Udacity environment.

## 🚘 Abstract

This project presents an autonomous vehicle prototype using the Udacity Self-Driving Car Simulator and a deep learning model based on the NVIDIA CNN architecture. The objective was to train a model that can predict steering angles based on center camera images. The CNN model was trained on recorded driving data and evaluated based on MSE and the car’s ability to drive autonomously in the simulator.

---

## 📁 Folder Structure

```
/SelfDrivingCarProject/
│
├── data/
│   ├── IMG/                    # Recorded driving images
│   └── driving_log.csv         # Steering, throttle, brake, speed data
│
├── model/
│   ├── model.py                # Model architecture (NVIDIA CNN)
│   ├── train.py                # Training script
│   ├── optimized_model.h5      # Final trained model
│   └── loss_curve.png          # Loss graph
│
├── test/
│   └── TestSimulation.py       # Loads model into simulator
│
├── utils/
│   └── preprocess_utils.py     # Preprocessing functions
│
├── README.md                   # This file
├── project_report.md           # Optional final report
├── demo_video.mp4              # Screen recording of model driving
└── requirements.txt            # Environment dependencies
```

---

## 🛠️ Technologies Used

- Python 3.8
- TensorFlow 2 / Keras
- NumPy, Pandas, OpenCV
- Matplotlib
- Flask, SocketIO (for integration)
- Udacity Self-Driving Car Simulator

---

## 📦 Setup Instructions

1. Clone this repo
2. Create and activate the environment:

```
conda create --name self_driving_env --file package_list.txt
conda activate self_driving_env
```

3. Run training:
```
python train.py
```

4. Test in simulator:
```
python TestSimulation.py
```

---

## 🧠 Model Architecture

Based on the [NVIDIA End-to-End Self-Driving CNN](https://arxiv.org/abs/1604.07316), our model consists of:
- 5 Convolutional layers
- ReLU activations
- Dropout layers to reduce overfitting
- Fully connected dense layers
- Final output: steering angle

---

## 📉 Results & Visualization

- Loss Function: Mean Squared Error (MSE)
- Evaluation: Validation loss and visual track performance

![Loss Curve](model/loss_curve.png)

---

## 🎥 Demo Video

If simulator worked, demo is available here: `demo_video.mp4`  
If not, simulation testing was skipped due to hardware limitations (documented in report).

---

## 👥 Team Contributions

- Artom: Data generation, driving and dataset creation
- [Teammate]: Built the CNN model and handled initial training
- **Rutarj**: Model tuning and optimization, documentation, loss visualization, and report preparation

---

## 📌 Challenges Faced

- Simulator crashes on non-NVIDIA GPU systems
- Data balancing and augmentation decisions to reduce bias
- Validation vs training loss convergence tuning

---
