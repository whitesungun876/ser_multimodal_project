import os
import pandas as pd
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

CSV_PATH = "/Users/whitesungun/Desktop/ser_multimodal_project/ser_multimodal_project/data/iemocap_4emo.csv"
SAVE_DIR = "/Users/whitesungun/Desktop/ser_multimodal_project/ser_multimodal_project/data/mel_spectrograms"
os.makedirs(SAVE_DIR, exist_ok=True)

def extract_mel(wav_path, sr=16000, n_mels=64):
    y, orig_sr = librosa.load(wav_path, sr=None)
    y = librosa.resample(y=y, orig_sr=orig_sr, target_sr=sr)  
    y = librosa.util.normalize(y)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def main():
    df = pd.read_csv(CSV_PATH)
    missing = 0
    failed = 0
    for i, row in tqdm(df.iterrows(), total=len(df)):
        wav_path = row['wav_path']
        if not os.path.exists(wav_path):
            missing += 1
            print(f"❌ Missing: {wav_path}")
            continue
        try:
            mel = extract_mel(wav_path)
            npy_path = os.path.join(SAVE_DIR, f"{i}.npy")
            np.save(npy_path, mel)
        except Exception as e:
            failed += 1
            print(f"⚠️ Error processing {wav_path}: {e}")
    print(f"\n✅ Done. Saved Mel spectrograms to: {SAVE_DIR}")
    if missing > 0:
        print(f"⚠️ Missing files: {missing}")
    if failed > 0:
        print(f"⚠️ Failed to process files: {failed}")

if __name__ == "__main__":
    main()




