import os
import numpy as np
import pandas as pd
from tqdm import tqdm

LABEL_CSV = "/Users/whitesungun/Desktop/ser_multimodal_project/ser_multimodal_project/data/iemocap_labels.csv"
BERT_DIR = "/Users/whitesungun/Desktop/ser_multimodal_project/ser_multimodal_project/data/bert_cls_features"
WAV2VEC2_DIR = "/Users/whitesungun/Desktop/ser_multimodal_project/ser_multimodal_project/data/wav2vec2_embeddings"
SAVE_DIR = "/Users/whitesungun/Desktop/ser_multimodal_project/ser_multimodal_project/data/early_fused_aligned"
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    df = pd.read_csv(LABEL_CSV)
    X, y, utt_ids = [], [], []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        utt_id = row["utt_id"]
        label = row["emotion"]
        bert_path = os.path.join(BERT_DIR, f"{utt_id}.npy")
        wav2vec2_path = os.path.join(WAV2VEC2_DIR, f"{utt_id}.npy")

        if not os.path.exists(bert_path) or not os.path.exists(wav2vec2_path):
            skipped += 1
            continue

        try:
            bert_feat = np.load(bert_path)
            wav_feat = np.load(wav2vec2_path)
            fused_feat = np.concatenate([bert_feat, wav_feat])
            X.append(fused_feat)
            y.append(label)
            utt_ids.append(utt_id)
        except Exception as e:
            print(f"⚠️ Error with {utt_id}: {e}")

    np.save(os.path.join(SAVE_DIR, "X.npy"), np.array(X))
    np.save(os.path.join(SAVE_DIR, "y.npy"), np.array(y))
    np.save(os.path.join(SAVE_DIR, "utt_ids.npy"), np.array(utt_ids))

    print(f"\n✅ Saved aligned data to {SAVE_DIR}")
    print(f"Total samples: {len(X)}, Skipped: {skipped}")

if __name__ == "__main__":
    main()
