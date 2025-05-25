import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch

CSV_PATH = "/Users/whitesungun/Desktop/ser_multimodal_project/ser_multimodal_project/data/iemocap_transcriptions.csv"
SAVE_DIR = "/Users/whitesungun/Desktop/ser_multimodal_project/ser_multimodal_project/data/bert_cls_features"
os.makedirs(SAVE_DIR, exist_ok=True)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=False)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def extract_cls_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  
    return cls_embedding  

def main():
    df = pd.read_csv(CSV_PATH)
    index_data = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        raw_id = str(row["utt_id"])  
        utt_id = raw_id.split()[0]

        text = str(row["text"])  
        
        try:
            cls_vec = extract_cls_embedding(text)
            save_path = os.path.join(SAVE_DIR, f"{utt_id}.npy")
            np.save(save_path, cls_vec)
            index_data.append({
                "utt_id": utt_id,
                "embedding_path": save_path
            })
        except Exception as e:
            print(f"⚠️ Error processing line {i}: {e}")

    # Save index file
    index_df = pd.DataFrame(index_data)
    index_df.to_csv(os.path.join(SAVE_DIR, "iemocap_bert_index.csv"), index=False)
    print(f"\n✅ Done! BERT CLS vectors and index saved to: {SAVE_DIR}")

if __name__ == "__main__":
    main()
