# Multimodal Transformer for Affect Analysis in Human-Virtual Agent Dyadic Interactions

## Project Overview
This project aims to **automate affect analysis** by predicting **Pleasure, Arousal, and Dominance (PAD)** values from **non-intrusive multimodal data** (video and audio). Traditional affect recognition methods **ignore Dominance** or rely on **intrusive sensor-based approaches** (EEG, HRV). Our solution uses a **transformer-based late fusion model** to integrate information from video and audio without requiring wearable sensors.

### 🔹 Key Features
- **PAD Prediction**: Predicts **Pleasure, Arousal, and Dominance** from video and audio.
- **Multimodal Approach**: Uses both **video (VideoViT)** and **audio (Wav2Vec2)** models.
- **Late Fusion Transformer Model**: Dynamically assigns importance to audio/video based on context.
- **Transfer Learning**: Pretrained models on DEAP dataset are fine-tuned on MITHOS dataset.
- **Real-time Emotion Analysis**: Enables real-time affect prediction without manual intervention.

---

## 📂 Project Directory Structure
To keep the repository well-organized, the following directory structure is used:

```
│── src/                     # Main source code
│   ├── audio/               # Audio processing pipeline
│   │   ├── audio_model.py   # Wav2Vec2-based model for audio PAD prediction
│   │   ├── audio_data_loader.py   # Loads the MITHOS dataset for audio
│   │   ├── audio_train.py   # Training script for audio model
│   ├── video/               # Video processing pipeline
│   │   ├── video_model.py   # ViViT-based model for video PAD prediction
│   │   ├── video_data_loader.py   # Loads the DEAP/MITHOS dataset for video
│   │   ├── video_train.py   # Training script for video model
│   ├── fusion/              # Fusion model pipeline
│   │   ├── fusion_model.py  # Transformer-based fusion model
│   │   ├── fusion_data_loader.py  # Loads preprocessed audio and video features
│   │   ├── fusion_train.py  # Training script for fusion model
│── configs/                 # Configuration files (YAML)
│── data/                    # Data storage (if applicable)
│── results/                 # Logs, results, and output files
│── requirements.txt         # List of dependencies
│── README.md                # Project documentation
│── LICENSE                  # License for open-source usage
```

---

## 🔧 Installation & Setup

### 🔹 Prerequisites
- **Docker (Podman GPU)**: Used for running the model in a containerized environment.
- **Python 3.8+** with **pip** and **virtual environments**.

### 🔹 Steps to Set Up
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/multimodal-affect-analysis.git
   cd multimodal-affect-analysis
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Docker Container** (Ensure GPU access if required):
   ```bash
   podman-gpu -it -v /home/rodricks/<someFolder>:/workspace:z -v /mnt/datasets/mithos/video_data:/video_data nvcr.io/nvidia/pytorch:22.05-py3
   ```

5. **Manually install OpenCV and dlib (inside Docker)**:
   ```bash
   pip install opencv-python==4.5.3.56 dlib
   ```

---

## ▶️ How to Run the Model

### 🔹 Train the Audio Model
```bash
python src/audio/audio_train.py
```

### 🔹 Train the Video Model (Pretrain on DEAP)
```bash
python src/video/video_train.py
```

### 🔹 Fine-tune Video Model on MITHOS
```bash
python src/video/train_mithos.py
```

### 🔹 Train the Fusion Model
```bash
python src/fusion/fusion_train.py
```

---

## 📊 Evaluation
After training, evaluate the models:

### 🔹 Evaluate the Audio Model
```bash
python src/audio/audio_evaluate.py
```

### 🔹 Evaluate the Video Model
```bash
python src/video/video_evaluate.py
```

### 🔹 Evaluate the Fusion Model
```bash
python src/fusion/fusion_evaluate.py
```

---

## 📈 Results & Performance
The performance of the models was evaluated using:
- **Mean Absolute Error (MAE)**
- **Range-2 Accuracy**
- **Pearson’s Correlation Coefficient (PCC)**
- **Mean Squared Error (MSE)**

### 🔹 Key Findings
- **Fusion Model performs better** than individual audio/video models.
- **Audio is more effective** in predicting **Pleasure and Arousal**.
- **Video is more effective** in predicting **Dominance**.

---

## 📝 Citation
If you use this work, please cite:
```bibtex
@thesis{Rodricks2024,
  author = {Senorita Rodricks},
  title = {Multimodal Transformer for Affect Analysis in Human-Virtual Agent Dyadic Interactions},
  school = {Saarland University},
  year = {2024}
}
```

---

### 🚀 **Next Steps**
1. **Copy and paste this README.md into your GitHub repository**.
2. **Test all commands** to ensure they run without issues.
3. **Let me know** if you'd like any modifications!
