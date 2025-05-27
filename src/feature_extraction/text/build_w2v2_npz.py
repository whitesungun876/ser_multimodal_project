import os
import numpy as np
import pandas as pd
from tqdm import tqdm

EMB_IDX   = "data/wav2vec2_embeddings/iemocap_wav2vec2_index.csv"
LABEL_CSV = "data/iemocap_labels.csv"

SAVE_DIR  = "data/wav2vec2_embeddings"
os.makedirs(SAVE_DIR, exist_ok=True)

LABEL2ID = {"ang": 0, "hap": 1, "sad": 2, "neu": 3}

def main():
    labels_df = pd.read_csv(LABEL_CSV)
    labels_df["label_id"] = labels_df["emotion"].map(LABEL2ID)

    emb_idx = pd.read_csv(EMB_IDX)
    merged = emb_idx.merge(labels_df[["utt_id","label_id"]],
                          on="utt_id", how="inner")

    X_list, y_list = [], []
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Building W2V2 array"):
        emb = np.load(row["embedding_path"])    
        X_list.append(emb)
        y_list.append(row["label_id"])

    X = np.stack(X_list).astype("float32")      
    y = np.array(y_list, dtype="int64")        

    np.save(os.path.join(SAVE_DIR, "X_w2v2.npy"), X)
    np.save(os.path.join(SAVE_DIR, "y_w2v2.npy"), y)
    print(f"saved X_w2v2.npy {X.shape}, y_w2v2.npy {y.shape}")

if __name__ == "__main__":
    main()
