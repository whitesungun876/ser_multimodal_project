# src/build_mel_npz.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

LABEL_CSV   = "data/iemocap_labels.csv"
MEL_ROOT    = "data/mel_spectrograms"
SAVE_DIR    = "data/mel_npz"
os.makedirs(SAVE_DIR, exist_ok=True)

LABEL2ID = {"ang":0,"hap":1,"sad":2,"neu":3}

def main():
    df = pd.read_csv(LABEL_CSV)
    feats, labels = [], []
    missing = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="📥 loading Mel .npy files"):
        npy_file = os.path.join(MEL_ROOT, f"{idx}.npy")   
        if not os.path.exists(npy_file):
            missing.append(idx)
            continue
        mel = np.load(npy_file)  
        feats.append(mel)
        labels.append(LABEL2ID[row['emotion']])


    print(f"⚠️ miss Mel ducument {len(missing)}/{len(df)} ，for example：{missing[:10]}")

    max_T = max(f.shape[1] for f in feats)
    padded = [np.pad(f,
                     ((0,0),(0, max_T - f.shape[1])),
                     mode="constant")
              for f in feats]

    X = np.stack(padded).astype("float32")   # (N, 64, max_T)
    y = np.array(labels).astype("int64")     # (N,)

    out_npz = os.path.join(SAVE_DIR, "mel_features.npz")
    np.savez(out_npz, X=X, y=y)
    print(f"✅ saved {X.shape} / {y.shape} -> {out_npz}")

    np.save(os.path.join(SAVE_DIR, "X.npy"), X)
    np.save(os.path.join(SAVE_DIR, "y.npy"), y)
    print(f"✅ also saved X.npy & y.npy in {SAVE_DIR}")

if __name__ == "__main__":
    main()

