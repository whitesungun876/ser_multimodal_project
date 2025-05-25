import os
import pandas as pd
from tqdm import tqdm

IEMOCAP_PATH = "/Volumes/MI USB/iemocap/IEMOCAP_full_release"
SAVE_PATH = "/Users/whitesungun/Desktop/ser_multimodal_project/ser_multimodal_project/data/iemocap_labels.csv"

def find_wav_path(wav_root, utt_id):
    for root, dirs, files in os.walk(wav_root):
        if f"{utt_id}.wav" in files:
            return os.path.join(root, f"{utt_id}.wav")
    return None

def build_label_list(iemocap_path):
    labels = []
    sessions = [f"Session{i}" for i in range(1, 6)]

    for sess in sessions:
        emo_eval_path = os.path.join(iemocap_path, sess, "dialog", "EmoEvaluation")
        wav_root_path = os.path.join(iemocap_path, sess, "sentences", "wav")

        for eval_file in os.listdir(emo_eval_path):
            if not eval_file.endswith(".txt"):
                continue

            with open(os.path.join(emo_eval_path, eval_file), 'r', encoding="utf-8", errors='ignore') as f:
                for line in f:
                    if line.startswith("["):
                        try:
                            parts = line.strip().split()
                            utt_id = parts[3]
                            emotion = parts[4]
                            if emotion not in ["ang", "hap", "sad", "neu"]:
                                continue

                            wav_path = find_wav_path(wav_root_path, utt_id)
                            if wav_path is None:
                                print(f"❌ Missing: {utt_id}")
                                continue

                            labels.append({
                                "utt_id": utt_id,
                                "emotion": emotion,
                                "session": sess,
                                "wav_path": wav_path
                            })

                        except Exception as e:
                            print(f"⚠️ Error parsing line: {line.strip()} - {e}")
    return labels

def main():
    label_data = build_label_list(IEMOCAP_PATH)
    df = pd.DataFrame(label_data)
    df.to_csv(SAVE_PATH, index=False)
    print(f"\n✅ Labels saved to: {SAVE_PATH}, total: {len(df)}")

if __name__ == "__main__":
    main()
