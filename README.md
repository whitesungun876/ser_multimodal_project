# 🎧 Multimodal Speech Emotion Recognition (SER)

This repository contains the experimental framework for **"Multimodal Speech Emotion Recognition: Audio, Text, and Fusion Approaches,"** conducted at the **University of Copenhagen**. The project systematically investigates deep learning architectures, multimodal fusion strategies, and targeted data augmentation to overcome common challenges in SER, such as subjective variability and class imbalance.

---

## 🔍 Research Focus

This project addresses three core hypotheses:

**H1 (Multimodal Fusion):** Combining audio and text signals yields higher performance than unimodal baselines.


**H2 (Architecture):** Deep convolutional models (ResNet-18) can match or exceed Transformer-based architectures on moderate-scale datasets when trained from scratch.


**H3 (Augmentation):** Class-differentiated data augmentation substantially improves performance on imbalanced emotion classes.



---

## 🚀 Key Features

### 🧠 Model Architectures

**ResNet-18 (CNN-only):** Strong baseline for extracting local spectral patterns from Mel spectrograms.


**CNN + BiLSTM:** Captures both local spectral information and temporal dependencies.

 
**Audio Transformer:** Patch-based self-attention mechanism applied to spectrograms.


**Text-only:** Frozen **BERT** ([CLS] token) followed by an MLP.


**Early Fusion:** Feature-level concatenation of audio and text embeddings.


**Patch-based Cross-Attention Fusion:** Fine-grained alignment of audio patches and text tokens.



### 🎵 Audio & Text Processing


**Audio:** 16 kHz mono resampling, 64-band Mel spectrograms (25ms window, 10ms hop), and Z-score normalization.


**Text:** IEMOCAP manual transcripts processed via `bert-base-uncased` WordPiece tokenizer.


**SpecAugment:** Time and frequency masking applied during training to improve robustness.



### 🔁 Data Augmentation Strategies

We implement a **Class-Differentiated Pipeline** that applies tailored augmentations per emotion to address dataset imbalance:


**Angry:** Heavy pitch shifts (\pm3 semitones) and intense SpecAugment.


**Happy:** Environmental noise overlay (MUSAN) and mild pitch/time variations.


**Sad:** Room reverb simulation and pink noise addition.


**Neutral:** Gaussian noise and slight SpecAugment.



---

## 📊 Experimental Results (IEMOCAP)

Results are reported as **Macro-F1 (Mean ± Std)** across 5-fold speaker-independent cross-validation.

| Model Configuration | Macro-F1 Score |
| --- | --- |
| Audio Transformer (Speech-only) | <br>0.554 \pm 0.030 

| CNN + BiLSTM (Speech-only) | <br>0.603 \pm 0.022 

| **ResNet-18 (Speech-only)** | <br>**0.613 \pm 0.025** 

| Early Fusion (Audio + Text) | <br>0.672 \pm 0.018 

| Cross-Attention Fusion | <br>0.700 \pm 0.020 

| **Cross-Attention + Class-Diff. Augmentation** | <br>**0.927 \pm 0.012** 


---

## 🗂 Dataset Setup

The primary dataset used is the **IEMOCAP** corpus:

**Size:** 5,531 utterances (~10 hours).


**Classes:** Angry (1,103), Happy (1,200), Sad (1,665), Neutral (1,563).


**Expected Directory Structure:**

```text
data/IEMOCAP/
├── wavs/
│   ├── Session1/ ...
│   └── Session5/ ...
└── labels.csv       # Format: file_name, emotion_label

```

---

## 🏗 Repository Structure

```text
speech-emotion-recognition/
├── data/                # Raw and processed datasets
├── features/            # Audio/Text feature extraction scripts
├── models/              # ResNet, BiLSTM, Transformer, and Fusion definitions
├── utils/               # Data loaders, Class-Differentiated Augmentation, metrics
├── configs/             # YAML experiment configurations
├── logs/                # TensorBoard training logs
├── checkpoints/         # Saved model weights
├── train.py             # Main training entry point
└── evaluate.py          # Evaluation and inference script

```

---

## ⚙️ Quick Start

### 1. Installation

```bash
git clone https://github.com/whitesungun876/ser_multimodal_project.git
cd ser_multimodal_project

conda create -n ser_env python=3.8 -y
conda activate ser_env
pip install -r requirements.txt

```

### 2. Preprocessing

```bash
python utils/data_loader.py \
  --dataset IEMOCAP \
  --output_dir data/IEMOCAP/processed/

```

### 3. Training

```bash
python train.py \
  --config configs/resnet18_baseline.yaml \
  --dataset IEMOCAP \
  --model resnet18

```

---

## 🔬 Research & Reproducibility

 
**Ablation Ready:** The codebase supports toggling SpecAugment, basic audio augmentation, and class-specific policies.


**Explainability:** Support for generating **Class Activation Maps (CAMs)** and attention heatmaps to verify emotional salience.


**Speaker-Independent:** Evaluation strictly follows 5-fold CV where one session (2 actors) is held out for testing per fold.



---

## 📄 Reference

If you use this code or research in your work, please cite:

```bibtex
@article{lian2025multimodal,
  title={Multimodal Speech Emotion Recognition: Audio, Text, and Fusion Approaches},
  author={Lian, Jieyu},
  journal={University of Copenhagen},
  year={2025}
}

```

