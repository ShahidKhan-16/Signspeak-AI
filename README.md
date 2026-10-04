<div align="center">

# HandSign ISL
### Real-Time Indian Sign Language Translation & Learning Platform

[![React Native](https://img.shields.io/badge/React%20Native-0.76+-61DAFB.svg?style=flat-square&logo=react&logoColor=black)](https://reactnative.dev/)
[![Expo](https://img.shields.io/badge/Expo-SDK%2052-000020.svg?style=flat-square&logo=expo&logoColor=white)](https://expo.dev/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-SocketIO-000000.svg?style=flat-square&logo=flask&logoColor=white)](https://flask-socketio.readthedocs.io/)
[![MediaPipe](https://img.shields.io/badge/Google-MediaPipe%20Tasks-0078D4.svg?style=flat-square&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E.svg?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

<p align="center">
  A full-stack, real-time sign language recognition system that bridges communication gaps using mobile computer vision, 3D landmark geometric invariants, low-latency WebSocket streaming, and speech synthesis.
</p>

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Machine Learning & Computer Vision Pipeline](#-machine-learning--computer-vision-pipeline)
- [WebSocket Protocol Specification](#-websocket-protocol-specification)
- [Repository Layout](#-repository-layout)
- [Installation & Getting Started](#-installation--getting-started)
  - [Prerequisites](#1-prerequisites)
  - [Backend Server Setup](#2-backend-server-setup)
  - [Mobile Client Setup](#3-mobile-client-setup)
  - [Server Connection & Configuration](#4-server-connection--configuration)
- [Supported Alphabets & Status](#-supported-alphabets--status)
- [Quality Assurance & Verification](#-quality-assurance--verification)
- [License](#-license)

---

## 📌 Overview

**HandSign ISL** provides an end-to-end assistive platform for translating Indian Sign Language (ISL) gestures into text and natural voice speech in real time.

Built around a high-performance **3-tab mobile architecture**, the application delivers:
1. **Translate (Live Studio):** Real-time camera feed with mirrored 21-point skeletal overlays, progressive hold-to-confirm word building, and integrated Text-to-Speech (TTS).
2. **Dictionary (Reference Guide):** An interactive anatomical sign language dictionary detailing joint positions, fingertip extensions, and execution guidelines for all 26 alphabets.
3. **Collect (Data Studio):** An integrated data collection environment enabling dataset expansion, validation, and on-device gesture recording.

---

## 🏗️ System Architecture

```mermaid
flowchart TB
    subgraph Client ["📱 Mobile Client (React Native / Expo SDK 52)"]
        direction TB
        C1["Camera View (5 FPS / 0.3 JPEG Quality)"]
        C2["Centralized SocketProvider (socket-context.tsx)"]
        C3["Landmark Mesh Canvas Overlay (1 - x Mirror Projection)"]
        C4["Debounced Hold-to-Confirm Word Builder"]
        C5["Expo Speech Engine (Text-to-Speech)"]
        C1 --> C2
        C2 --> C3
        C2 --> C4 --> C5
    end

    subgraph Server ["⚡ Backend Engine (Flask-SocketIO / Eventlet)"]
        direction TB
        S1["WebSocket Gateway (:5001)"]
        S2["MediaPipe HandLandmarker (XNNPACK CPU / 21 Landmarks)"]
        S3["90-Dimensional Geometric Invariant Extractor"]
        S4["Random Forest Classifier (23 Classes / 91.12% Test Accuracy)"]
        S5["Exponential Moving Average (EMA) Probability Smoother"]
        S1 --> S2 --> S3 --> S4 --> S5 --> S1
    end

    C2 <===>|WebSocket Stream (base64 frame / landmarks + prediction)| S1
```

---

## 🌟 Key Features

| Feature | Technical Implementation | Value |
| :--- | :--- | :--- |
| **Real-Time Translation** | Low-latency (5 FPS) WebSocket transmission with MediaPipe CPU inference (~15–25ms). | Sub-second gesture translation with smooth on-screen prediction updates. |
| **Hand Symmetry & Mirror Invariance** | Bidirectional $X$-axis flip data augmentation during model training. | 100% accuracy regardless of Left Hand vs. Right Hand execution or front-camera mirroring. |
| **Skeletal Mesh Overlay** | Canvas vector rendering with front-camera mirror coordinate transformation: $x_{\text{screen}} = (1 - x) \times W$. | Skeletal joints and connection bones align directly over physical fingers on screen. |
| **Progressive Word Builder** | 500ms progressive hold confirmation with 2-frame jitter debouncing. | Prevents accidental insertions caused by intermediate transition gestures. |
| **Dynamic IP Switcher** | Centralized `SocketProvider` coupled with an in-app Settings modal and network detection presets. | Instant server re-configuration across Wi-Fi networks without restarting the app. |
| **ISL Knowledge Base** | Dedicated Dictionary screen with real-time status badges, search filtering, and anatomical breakdowns. | Educational reference for learners, educators, and accessibility practitioners. |

---

## 🧠 Machine Learning & Computer Vision Pipeline

```
  Raw RGB Frame
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ 1. MediaPipe Tasks HandLandmarker (21 3D Coordinates) │
└────────────────────────────────────────────────────────┘
       │  Normalized Sensor Space (x, y, z)
       ▼
┌────────────────────────────────────────────────────────┐
│ 2. 90-Dimensional Geometric Invariant Feature Vector  │
│    • 63-dim: Scale-normalized coordinates relative     │
│              to wrist: (P_i - P_wrist) / max_dist      │
│    • 15-dim: Joint bending angles (arccos / pi)       │
│    •  5-dim: Fingertip-to-wrist extension distances    │
│    •  4-dim: Thumb-to-fingertip pinch distances        │
│    •  3-dim: Adjacent fingertip spread distances       │
└────────────────────────────────────────────────────────┘
       │  Feature Tensor Shape: (90,)
       ▼
┌────────────────────────────────────────────────────────┐
│ 3. Random Forest Classification & EMA Smoothing        │
│    P_smooth = α · P_curr + (1 - α) · P_prev  (α = 0.65)│
└────────────────────────────────────────────────────────┘
```

* **Dataset Scale:** 1,914 raw manual recordings augmented 15× with 3D rotational jitter ($\pm 15^\circ$), scale scaling ($\pm 12\%$), and Gaussian noise ($\sigma = 0.008$), yielding **21,434 training samples**.
* **Model Benchmark:** **91.12% held-out test accuracy** across 383 pure, untouched test recordings.

---

## 🔌 WebSocket Protocol Specification

The client and server communicate via bidirectional Socket.IO events over port `5001`:

### Client $\to$ Server
* `video_frame`: Transmits base64 JPEG payload (`image`), timestamp, and resolution.
* `collect_sample`: Transmits labeled gesture capture for dataset expansion.
* `get_collect_counts`: Requests current sample distribution across classes.

### Server $\to$ Client
* `frame_ack`: Returns processing telemetry (`count`, `hands_detected`, `landmarks`, `inference_ms`, `dropped`).
* `recognized_letter`: Emits classified alphabet label and smoothed confidence score (`letter`, `confidence`).
* `collect_result`: Confirms sample storage with updated per-class count.

---

## 📁 Repository Layout

```text
handsign/
├── backend/
│   ├── app.py                     # Flask-SocketIO server & WebSocket event loop
│   ├── classifier.py              # 90-dim geometric feature extractor & inference
│   ├── landmarks.py               # MediaPipe Tasks HandLandmarker processor
│   ├── train_alphabet.py          # Random Forest training & augmentation pipeline
│   ├── data/
│   │   └── isl_landmarks/
│   │       └── my_own_data.csv    # 1,914 landmark samples (23 classes)
│   ├── models/
│   │   ├── hand_landmarker.task   # MediaPipe HandLandmarker neural task model
│   │   └── isl_alphabet_classifier.pkl  # Trained 90-dim Random Forest classifier
│   ├── requirements.txt           # Python package manifest
│   ├── .env.example               # Environment variables template
│   └── .env                       # Local secrets (git-ignored)
│
├── mobile-app/
│   ├── app/
│   │   ├── (tabs)/
│   │   │   ├── _layout.tsx        # Tab bar navigation (Translate, Dictionary, Collect)
│   │   │   ├── index.tsx          # 📸 Main Translate Screen (CameraScreen component)
│   │   │   ├── explore.tsx        # 📖 ISL Sign Dictionary & Reference Guide
│   │   │   ├── collect.tsx        # ✍️ Custom Dataset Collection Studio
│   │   │   └── camera.tsx         # Master Camera Translator component
│   │   ├── _layout.tsx            # Root navigation stack & SocketProvider wrapper
│   │   └── modal.tsx              # ⚙️ Settings & Dynamic Server IP Switcher
│   ├── context/
│   │   └── socket-context.tsx     # Centralized WebSocket manager & hook
│   ├── components/                # Shared UI elements & icon mappings
│   ├── app.json                   # Expo application metadata & native permissions
│   └── package.json               # JavaScript/TypeScript dependencies
│
├── .gitignore                     # Git exclusion rules (.pkl, .task, .env, CSVs)
└── README.md                      # Project documentation
```

---

## 🚀 Installation & Getting Started

### 1. Prerequisites
* **Node.js** (v18.0.0 or higher) and `npm`
* **Python** (v3.10 or higher) and `pip`
* **Expo Go** app on a physical iOS/Android device, or an active Simulator/Emulator
* Both development machine and mobile device connected to the **same local Wi-Fi network**

---

### 2. Backend Server Setup

1. Open a terminal and navigate to `backend/`:
   ```bash
   cd backend
   ```

2. Initialize and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # macOS / Linux
   # .\venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure local environment variables:
   ```bash
   cp .env.example .env
   ```

5. *(Optional)* Retrain the classifier model:
   ```bash
   python train_alphabet.py
   ```

6. Launch the backend server:
   ```bash
   python app.py
   ```
   *The server initializes on `http://0.0.0.0:5001`.*

---

### 3. Mobile Client Setup

1. Open a separate terminal and navigate to `mobile-app/`:
   ```bash
   cd mobile-app
   ```

2. Install npm dependencies:
   ```bash
   npm install
   ```

3. Start the Expo bundler:
   ```bash
   npx expo start
   ```

4. Launch the application:
   * **Physical Device:** Scan the generated QR code using the iOS Camera app or Expo Go on Android.
   * **iOS Simulator:** Press `i`.
   * **Android Emulator:** Press `a`.

---

### 4. Server Connection & Configuration

1. In the mobile app, tap the **⚙️ Settings** icon in the upper-right corner.
2. Enter your host machine's local IP address:
   ```text
   http://<YOUR_LOCAL_IP>:5001
   ```
   *(On macOS, retrieve your IP via `ipconfig getifaddr en0`)*.
3. Tap **Save & Reconnect**. The status indicator will transition to **`🟢 Connected to Backend`**.

---

## 📊 Supported Alphabets & Status

| Class Category | Alphabets Included | Classifier Support | Notes |
| :--- | :--- | :---: | :--- |
| **Static Manual Alphabets** | `A, B, C, D, E, F, G, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Z` | 🟢 Supported | Classified in real time with **91.12% accuracy**. |
| **Dynamic Motion Signs** | `H, J, Y` | ⚠️ Reference Only | Dynamic signs involving spatial movement. Anatomical instructions available in the Dictionary tab. |

---

## 🧪 Quality Assurance & Verification

To validate type safety, linting, and compilation integrity across both environments:

```bash
# 1. Validate Mobile TypeScript Compilation
cd mobile-app && npx tsc --noEmit

# 2. Run Mobile Linter
cd mobile-app && npm run lint

# 3. Validate Python Backend Syntax
python3 -m py_compile backend/app.py backend/classifier.py backend/landmarks.py backend/train_alphabet.py
```

---

## 📄 License

This repository is distributed under the **MIT License**. See the `LICENSE` file for details.
