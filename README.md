🎧 Speech Emotion Recognition (SER)

A PyTorch-based framework for classifying speech signals into emotion categories (e.g., happy, sad, angry, neutral).
It integrates both traditional acoustic features (MFCC, eGeMAPS/eSAC) and modern deep learning architectures (CNN–RNN, Conformer, MemoCMT, MTAF).
The system supports multiple public datasets and provides a unified pipeline for extending to custom data.

🚀 Key Features
🗂 Datasets

Pre-integrated: EMOVO, EMODB, German Emotional Speech

Easy to extend — add new datasets using a standardized format (labels.csv + audio folder)

🎵 Acoustic Features

MFCC, eGeMAPS/eSAC, Spectrograms

Augmentation: SpecAugment and its variants for robust training

🧠 Models

cnn_rnn: CNN–RNN hybrid

conformer: Convolution-augmented Transformer

memocmt: Memory-augmented Cross-Modal Transformer

mtaf: Multimodal Transformer Attention Fusion

📊 Evaluation & Analysis

Compare feature sets (e.g., eGeMAPS vs eSAC)

Cross-corpus generalization and transfer learning

Minority-class recall and emotion confusion analysis

🏗 Repository Structure
speech-emotion-recognition/
├── data/             # Raw and processed datasets
├── features/         # Feature extraction scripts
├── models/           # Model definitions and training modules
├── utils/            # Data loaders, metrics, augmentations
├── configs/          # YAML/JSON hyperparameter files
├── logs/             # TensorBoard training logs
├── checkpoints/      # Saved model weights
├── requirements.txt  # Python dependencies
├── train.py          # Training entry point
├── evaluate.py       # Evaluation & inference script
└── README.md         # This document

⚙️ Quick Start
1️⃣ Clone & Install
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition
conda create -n ser_env python=3.8 -y
conda activate ser_env
pip install -r requirements.txt

2️⃣ Prepare Data

Organize your dataset as:

data/{Dataset}/
├── wavs/              # Audio files (.wav)
└── labels.csv         # file_name,emotion_label


Preprocess to 16kHz mono:

python utils/data_loader.py --dataset EMOVO --output_dir data/EMOVO/processed/

3️⃣ Extract Features

MFCC example:

python features/extract_mfcc.py \
  --input_dir data/EMOVO/processed/wavs/ \
  --output_dir features/EMOVO/mfcc/ \
  --n_mfcc 40


eGeMAPS/eSAC example:

python features/extract_egemaps.py \
  --input_dir data/EMODB/processed/wavs/ \
  --output_dir features/EMODB/egemaps/

4️⃣ Train a Model
python train.py \
  --config configs/default_config.yaml \
  --dataset EMOVO \
  --feature mfcc \
  --model cnn_rnn


Checkpoints are saved under:

checkpoints/{dataset}/{model}/best_model.pt

5️⃣ Evaluate or Infer

Batch evaluation:

python evaluate.py \
  --config configs/default_config.yaml \
  --dataset EMOVO \
  --feature mfcc \
  --model cnn_rnn \
  --checkpoint checkpoints/EMOVO/cnn_rnn/best_model.pt


Single-file inference:

python evaluate.py \
  --infer \
  --wav_path path/to/audio.wav \
  --feature mfcc \
  --model cnn_rnn \
  --checkpoint checkpoints/EMOVO/cnn_rnn/best_model.pt

🧩 Configuration & Logging

Modify hyperparameters in configs/*.yaml

TensorBoard logs are saved under logs/{dataset}/{model}/

Launch TensorBoard:

tensorboard --logdir logs/

🧱 Add a New Dataset

Create a folder data/YourDataset/ with:

wavs/        # all audio files
labels.csv   # file_name, emotion_label


Update utils/data_loader.py to support loading your dataset

Train with:

python train.py --dataset YourDataset

🧮 Dependencies
torch>=1.8.0
torchaudio
librosa
numpy
scipy
pandas
scikit-learn
pyAudioAnalysis (optional)
tensorboard
PyYAML
tqdm


Install with:

pip install -r requirements.txt

🔬 Reproducibility & Extensibility

Deterministic training seeds and dataset splits

Modular design — easily swap models, features, or datasets

Compatible with transfer learning and multimodal emotion research

🙏 Acknowledgments

This repository builds upon prior work in speech emotion recognition research and open datasets including EMOVO, EMODB, and German Emotional Speech.
Contributions, improvements, and extensions are welcome via pull requests!

