# Speech Emotion Recognition

## Overview

A PyTorch-based system for classifying speech into emotion categories (e.g., happy, sad, angry, neutral). Combines traditional acoustic features (MFCC, eGeMAPS/eSAC) with modern deep learning models (CNN–RNN, Conformer, MemoCMT, MTAF). Supports multiple public datasets and easy extension to custom data.

## Features

* **Datasets**: EMOVO, EMODB, German Emotional Speech; add your own via a standardized format.
* **Features**: MFCC, eGeMAPS/eSAC, spectrograms; data augmentation with SpecAugment variants.
* **Models**:

  * CNN–RNN hybrid
  * Conformer (Conv-augmented Transformer)
  * Memory-augmented Cross-Modal Transformer (MemoCMT)
  * Multimodal Transformer Fusion (MTAF)
* **Evaluation**: Compare features (eGeMAPS vs. eSAC), cross-corpus transfer, minority-class recall.

## Repo Structure

speech-emotion-recognition/
├── data/                   # Raw and processed datasets
├── features/               # Feature extraction scripts
├── models/                 # Model definitions and training code
├── utils/                  # Data loaders, metrics, augmentations
├── configs/                # YAML/JSON hyperparameter files
├── logs/                   # TensorBoard logs
├── checkpoints/            # Saved model weights
├── requirements.txt        # Dependencies
├── train.py                # Training script
├── evaluate.py             # Evaluation/inference script
└── README.md               # This document

## Quick Start

1. **Clone & Install**
   git clone [https://github.com/yourusername/speech-emotion-recognition.git](https://github.com/yourusername/speech-emotion-recognition.git)
   cd speech-emotion-recognition
   conda create -n ser\_env python=3.8 -y
   conda activate ser\_env
   pip install -r requirements.txt

2. **Prepare Data**

   * Place audio in data/{Dataset}/wavs/ and labels in data/{Dataset}/labels.csv (file\_name,emotion\_label).
   * Preprocess to 16 kHz mono:
     python utils/data\_loader.py --dataset EMOVO --output\_dir data/EMOVO/processed/

3. **Extract Features**

   * MFCC example:
     python features/extract\_mfcc.py --input\_dir data/EMOVO/processed/wavs/ --output\_dir features/EMOVO/mfcc/ --n\_mfcc 40
   * eGeMAPS/eSAC example:
     python features/extract\_egemaps.py --input\_dir data/EMODB/processed/wavs/ --output\_dir features/EMODB/egemaps/

4. **Train**
   python train.py --config configs/default\_config.yaml --dataset EMOVO --feature mfcc --model cnn\_rnn
   Checkpoints saved under checkpoints/{dataset}/{model}/.

5. **Evaluate / Infer**

   * Batch evaluation:
     python evaluate.py --config configs/default\_config.yaml --dataset EMOVO --feature mfcc --model cnn\_rnn --checkpoint checkpoints/EMOVO/cnn\_rnn/best\_model.pt
   * Single-file inference:
     python evaluate.py --infer --wav\_path path/to/audio.wav --feature mfcc --model cnn\_rnn --checkpoint checkpoints/EMOVO/cnn\_rnn/best\_model.pt

## Configuration & Logging

* Edit hyperparameters in configs/\*.yaml.
* Training logs saved in logs/{dataset}/{model}/. Launch TensorBoard:
  tensorboard --logdir logs/

## Add a New Dataset

1. Add data/YourDataset/wavs/ and labels.csv.
2. Update utils/data\_loader.py to handle YourDataset.
3. Run training with --dataset YourDataset.

## Dependencies

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

## Acknowledgments

Thanks to the authors of EMOVO, EMODB, German Emotional Speech, and all referenced works for inspiration and resources. Contributions and feedback are welcome!

