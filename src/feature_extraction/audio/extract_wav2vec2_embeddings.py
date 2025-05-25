import os
import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

LABEL_CSV = "/Users/whitesungun/Desktop/ser_multimodal_project/ser_multimodal_project/data/iemocap_labels.csv"
SAVE_DIR = "/Users/whitesungun/Desktop/ser_multimodal_project/ser_multimodal_project/data/wav2vec2_embeddings"
os.makedirs(SAVE_DIR, exist_ok=True)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()
if torch.cuda.is_available():
    model.to("cuda")

def fix_path(original_path):
    parts = original_path.split("/")
    filename = parts[-1] 
    foldername = "_".join(filename.split("_")[:2])  
    parts[-2] = foldername
    return "/".join(parts)

def extract_embedding(wav_path):
    speech, sr = librosa.load(wav_path, sr=16000, mono=True)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state  
        sentence_embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
    return sentence_embedding

def main():
    df = pd.read_csv(LABEL_CSV)
    df["wav_path"] = df["wav_path"].apply(fix_path)

    saved_items = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        utt_id = row["utt_id"]
        wav_path = row["wav_path"]

        if not os.path.exists(wav_path):
            print(f"❌ Missing: {wav_path}")
            continue

        try:
            emb = extract_embedding(wav_path)
            npy_path = os.path.join(SAVE_DIR, f"{utt_id}.npy")
            np.save(npy_path, emb)
            saved_items.append({
                "utt_id": utt_id,
                "embedding_path": npy_path
            })
        except Exception as e:
            print(f"⚠️ Error processing {wav_path}: {e}")

    pd.DataFrame(saved_items).to_csv(
        os.path.join(SAVE_DIR, "iemocap_wav2vec2_index.csv"),
        index=False
    )
    print(f"\n✅ Done. Extracted embeddings saved to: {SAVE_DIR}")

if __name__ == "__main__":
    main()
