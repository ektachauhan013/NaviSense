# NaviSense: AI-Powered Mobility Assistant for the Visually Impaired

NaviSense is a real-time, AI-powered spatial intelligence and navigation assistant designed to help visually impaired individuals navigate indoor and outdoor environments safely. Using a standard smartphone camera as an edge device, the system detects obstacles, estimates monocular depth, calculates safe walking corridors, and provides real-time voice-guided navigation through spatial audio.

## 🏗️ System Architecture
The system follows a hybrid AI architecture, combining Deep Learning (Neural Networks) for perception with Symbolic AI (Rule-Based Expert Systems) for deterministic safety.

**Pipeline Flow:**
`Camera Input` → `YOLOv8 (Object Detection)` → `Depth Anything V2 (Monocular Depth)` → `Spatial Intelligence (Zones & Free Space)` → `Safety Decision Engine` → `Voice Output (TTS)`

*   **Backend:** FastAPI (Python 3.12) with async frame-dropping to ensure real-time responsiveness.
*   **Frontend:** HTML5, JavaScript (Camera API, Canvas overlay, Spatial Audio TTS).
*   **Deployment:** Dockerized for cloud or edge deployment.

## ✨ Key Features & Upgrades (EY Workstream)
*The Depth Estimation & Spatial Intelligence workstream focused on making the system robust, safe, and perspective-aware.*

1.  **Vision Transformer Depth Estimation:** Utilizes **Depth Anything V2 Small** (Hugging Face) to generate relative depth maps.
2.  **Adaptive EMA Temporal Filtering:** Monocular depth models suffer from high-frequency noise. We implemented an Exponential Moving Average (EMA) to smooth depth maps across frames, preventing proximity alerts from flickering.
3.  **Perspective-Weighted Zones:** Unlike naive image slicing, our spatial zones dynamically expand and contract based on the Y-axis to model the trapezoidal shape of human walking paths.
4.  **Continuous 1D Clearance Profile:** The floor is divided into 15 vertical strips to find the widest navigable gap, rather than hard-coded binary thresholds.
5.  **Safety State Machine (`decision_engine.py`):** A deterministic symbolic AI layer that resolves conflicts. It includes a **Low-Texture Failsafe** (stops the user if facing a blank wall/glass door) and a **2.5-second Cooldown** to prevent audio spam.
6.  **Developer Telemetry HUD:** A real-time frontend overlay displaying FPS, latency, and the active AI state (CLEAR, CAUTION, STOP).

## 🧠 AI Syllabus Alignment
This project implements several foundational AI concepts:
*   **Intelligent Agents (Unit I):** A goal-based agent operating in a dynamic environment.
*   **Rule-Based Expert Systems (Unit II):** The `decision_engine.py` and `advice.py` modules use IF-THEN logic to translate probabilistic AI outputs into safe actions.
*   **Handling Uncertainty (Unit III):** Applied **Robust Statistics** (Median & Percentile Aggregation) to ignore noisy depth pixels, and **Temporal Smoothing** to handle time-series uncertainty.
*   **Neural Networks (Unit III):** Leverages Transfer Learning with a CNN (YOLOv8) and a Vision Transformer (Depth Anything V2).
*   **Ethics in AI (Unit IV):** The system is designed with a "Fail-Safe" philosophy (defaulting to STOP when uncertain). **Disclaimer:** This is a prototype and is not a replacement for a white cane or guide dog.

## 💻 Tech Stack
*   **Deep Learning:** `torch`, `torchvision`, `transformers`
*   **Computer Vision:** `ultralytics` (YOLOv8), `opencv-python-headless`
*   **Data & Math:** `numpy`, `Pillow`
*   **Backend:** `fastapi`, `uvicorn`, `python-multipart`
*   **Frontend:** HTML5, CSS3, Vanilla JavaScript (Web Speech API)

## 🚀 How to Run Locally

**1. Clone the repository and set up the environment:**
```bash
git clone https://github.com/ektachauhan013/NaviSense.git
cd NaviSense
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate