import os
import csv

IEMOCAP_ROOT = "/Volumes/MI USB/iemocap/IEMOCAP_full_release"
OUTPUT_CSV = "/Users/whitesungun/Desktop/ser_multimodal_project/ser_multimodal_project/data/iemocap_transcriptions.csv"

def extract_transcriptions():
    rows = []
    for session_id in range(1, 6):
        trans_dir = os.path.join(IEMOCAP_ROOT, f"Session{session_id}", "dialog", "transcriptions")
        if not os.path.exists(trans_dir):
            print(f"❌ Directory not found: {trans_dir}")
            continue

        for filename in os.listdir(trans_dir):
            if not filename.endswith(".txt"):
                continue

            with open(os.path.join(trans_dir, filename), 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split(":", 1)
                    if len(parts) != 2:
                        continue
                    utt_id, text = parts
                    utt_id = utt_id.strip()
                    text = text.strip()
                    rows.append({"utt_id": utt_id, "text": text})

    with open(OUTPUT_CSV, "w", encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["utt_id", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ extract completed，totally {len(rows)} are transcripted，saved to：{OUTPUT_CSV}")

if __name__ == "__main__":
    extract_transcriptions()
