import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.utils.dataset_loader       import get_dataloaders
from src.models.audio.cnn_model     import CNNModel
from src.models.text.transformer_model import TransformerModel
from src.models.audio.cnn_resnet       import ResNet18
from src.models.fusion.transformer_fused import FusedTransformer

try:
    from src.evaluation.metrics import compute_metrics
except ImportError:
    def compute_metrics(y_true, y_pred):
        acc = (np.array(y_true) == np.array(y_pred)).mean()
        print(f"[metrics] accuracy = {acc:.4f}")


def parse_args():
    p = argparse.ArgumentParser("Train SER multimodal model")
    p.add_argument("--fusion_data",  type=str, default=None,
                   help="Path to early-fused feature .npy (X.npy)")
    p.add_argument("--spectro_data", type=str, default=None,
                   help="Path to spectrogram/Mel feature .npy (X.npy)")
    p.add_argument("--labels",       type=str, required=True,
                   help="Path to labels (y.npy or pickle etc.)")
    p.add_argument("--model_type",   type=str,
                   choices=["mlp", "cnn", "transformer", "resnet", "xfmr"],
                   default="cnn", help="Which model to train")
    p.add_argument("--save_dir",     type=str, default="checkpoints",
                   help="Directory to save checkpoints")
    p.add_argument("--epochs",       type=int, default=20)
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--num_classes",  type=int,   default=4)
    p.add_argument("--val_split",    type=float, default=0.2)
    return p.parse_args()


def _load_numpy(path: str) -> np.ndarray:
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"❌ file not found: {path}")
    return np.load(path, allow_pickle=False)


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)

    is_spec = args.model_type == "resnet"
    feat_path = args.spectro_data if is_spec else args.fusion_data
    if feat_path is None:
        flag = "--spectro_data" if is_spec else "--fusion_data"
        raise ValueError(f"{flag} must be given for model_type={args.model_type}")

    if feat_path.endswith(".npz"):
        with np.load(feat_path) as d:
            X = d["X"].astype(np.float32)
    else:
        X = np.load(feat_path).astype(np.float32)
    # load labels (npz or npy)
    if args.labels.endswith(".npz"):
        with np.load(args.labels) as d:
            y = d["y"]
    else:
        y = np.load(args.labels)
    # sanity check
    if len(X) != len(y):
        raise ValueError(f"X ({len(X)}) and y ({len(y)}) length mismatch")
    print(f"[data] X shape {X.shape}, y shape {y.shape}")

    train_loader, val_loader = get_dataloaders(
        x_path    = feat_path,
        y_path    = args.labels,
        batch_size= args.batch_size,
        val_split = args.val_split,
        shuffle   = True,          
    )

    input_dim = X.shape[1] if X.ndim == 2 else None
    if args.model_type == "mlp":
        model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, args.num_classes)
        )
    elif args.model_type == "cnn":
        model = CNNModel(input_dim=input_dim, num_classes=args.num_classes)
    elif args.model_type == "transformer":
        model = TransformerModel(input_dim=input_dim, num_classes=args.num_classes)
    elif args.model_type == "resnet":
        model = ResNet18(num_classes=args.num_classes)
    elif args.model_type == "xfmr":
        model = FusedTransformer(input_dim=input_dim, num_classes=args.num_classes)
    else:
        raise ValueError("Unknown model type")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"[Epoch {epoch}/{args.epochs}] train-loss = {train_loss/len(train_loader):.4f}")

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        compute_metrics(all_labels, all_preds)

        ckpt = os.path.join(args.save_dir, f"{args.model_type}_ep{epoch}.pt")
        torch.save(model.state_dict(), ckpt)


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
