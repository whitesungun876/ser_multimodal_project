# src/merge_bert.py

import os
import numpy as np
import pandas as pd

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname((here))
    data_dir     = os.path.join(project_root, "data")
    
    labels_csv = os.path.join(project_root, "data", "iemocap_labels.csv")
    bert_dir    = os.path.join(project_root, "data", "bert_cls_features")
    out_path    = os.path.join(project_root, "data", "text_cls_embeddings.npy")

    df = pd.read_csv(labels_csv)
    feats = []
    for utt in df["utt_id"]:
        fp = os.path.join(bert_dir, f"{utt}.npy")
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Missing BERT vector: {fp}")
        feats.append(np.load(fp))  # shape (768,)

    X = np.stack(feats, axis=0)  # (N, 768)
    np.save(out_path, X)
    print(f"Saved merged BERT embeddings to {out_path}, shape={X.shape}")

if __name__ == "__main__":
    main()
