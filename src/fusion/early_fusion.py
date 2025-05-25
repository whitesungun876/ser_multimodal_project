import os
import numpy as np
import pandas as pd
from tqdm import tqdm

LABEL_CSV = "/Users/whitesungun/Desktop/ser_multimodal_project/ser_multimodal_project/data/iemocap_labels.csv"
BERT_DIR = "/Users/whitesungun/Desktop/ser_multimodal_project/ser_multimodal_project/data/bert_cls_features"
WAV2VEC2_DIR = "/Users/whitesungun/Desktop/ser_multimodal_project/ser_multimodal_project/data/wav2vec2_embeddings"
SAVE_DIR = "/Users/whitesungun/Desktop/ser_multimodal_project/ser_multimodal_project/data/early_fused"
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    df = pd.read_csv(LABEL_CSV)
    fused_features = []
    labels = []
    missed = 0

    for i, row in tqdm(df.iterrows(), total=len(df)):
        utt_id = row["utt_id"]
        emotion = row["emotion"]

        bert_path = os.path.join(BERT_DIR, f"{utt_id}.npy")
        wav2vec2_path = os.path.join(WAV2VEC2_DIR, f"{utt_id}.npy")

        if not os.path.exists(bert_path) or not os.path.exists(wav2vec2_path):
            missed += 1
            continue

        try:
            bert_feat = np.load(bert_path)
            wav_feat = np.load(wav2vec2_path)
            fused = np.concatenate([bert_feat, wav_feat])  # early fusion
            fused_features.append(fused)
            labels.append(emotion)
        except Exception as e:
            print(f"⚠️ Error loading {utt_id}: {e}")
            continue

    # Save features and labels
    np.save(os.path.join(SAVE_DIR, "X.npy"), np.array(fused_features))
    np.save(os.path.join(SAVE_DIR, "y.npy"), np.array(labels))
    print(f"\n✅ Saved fused features to {SAVE_DIR} | Samples: {len(fused_features)}, Skipped: {missed}")

if __name__ == "__main__":
    main()
