import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data_augmentation.ser_dataset import SERAugmentedDataset
from src.models.audio.cnn_model         import CNNModel
from src.models.audio.cnn_resnet        import ResNet18
from src.models.audio.cnn_bilstm        import CNNBiLSTMSER
from src.models.audio.audio_transformer import AudioTransformer
from src.models.text.transformer_model  import TransformerModel
from src.models.fusion.transformer_fused import ImprovedFusedTransformer

try:
    from src.evaluation.metrics import compute_metrics
except ImportError:
    def compute_metrics(y_true, y_pred):
        acc = (np.array(y_true) == np.array(y_pred)).mean()
        print(f"[metrics] Accuracy = {acc:.4f}")


def parse_args():
    p = argparse.ArgumentParser("Train SER multimodal model (E3B)")
    p.add_argument("--wav_list",            type=str,   required=True)
    p.add_argument("--precomputed_spectro", type=str,   required=True)
    p.add_argument("--text_data",           type=str,   default=None)
    p.add_argument("--labels",              type=str,   required=True)
    p.add_argument(
        "--model_type",
        choices=["mlp","cnn","transformer","resnet","cnnbilstm","xfmr","audio_tf"],
        default="cnn"
    )
    p.add_argument("--augment",             action="store_true")
    p.add_argument("--save_dir",            type=str,   default="checkpoints")
    p.add_argument("--epochs",              type=int,   default=30)
    p.add_argument("--batch_size",          type=int,   default=64)
    p.add_argument("--lr",                  type=float, default=1e-4)
    p.add_argument("--num_classes",         type=int,   default=4)
    p.add_argument("--val_split",           type=float, default=0.2)
    p.add_argument("--num_workers",         type=int,   default=4)

    p.add_argument(
        "--use_early_stop",
        action="store_true",
        help="Use Early Stopping"
    )
    p.add_argument(
        "--early_stop_patience",
        type=int,
        default=3,
        help="Early Stopping Patience"
    )
    return p.parse_args()


def load_numpy(path: str):
    """Load .npy / .npz – if .npz pick first of X/x/y."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        for k in ("X", "x", "y"):
            if k in data:
                return data[k]
        return data[list(data.keys())[0]]
    return np.load(path, allow_pickle=True)


def make_dataloaders(
    wav_list: str,
    precomputed_spectro: str,
    text_data: str,
    labels: str,
    batch_size: int,
    val_split: float,
    augment: bool,
    num_workers: int
):
    ds = SERAugmentedDataset(
        wav_list     = wav_list,
        spectro_data = precomputed_spectro,
        labels       = labels,
        text_data    = text_data,
        augment      = augment,
    )

    n_val   = int(len(ds) * val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader   = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # 1) Inspect pre-computed Mel shape ➜ n_mels, T_max
    sample = load_numpy(args.precomputed_spectro)
    if sample.ndim != 3:
        raise ValueError("precomputed_spectro must be (N,n_mels,T_max)")
    n_mels, T_max = sample.shape[1:]

    # 2) Text dim if any
    text_dim = 0
    if args.text_data:
        txt_arr = load_numpy(args.text_data)
        if txt_arr.ndim == 2:
            text_dim = txt_arr.shape[1]   # e.g. 768
        else:
            text_dim = 0

    # 3) DataLoaders
    train_loader, val_loader = make_dataloaders(
        wav_list            = args.wav_list,
        precomputed_spectro = args.precomputed_spectro,
        text_data           = args.text_data,
        labels              = args.labels,
        batch_size          = args.batch_size,
        val_split           = args.val_split,
        augment             = args.augment,
        num_workers         = args.num_workers,
    )

    # 4) Build model
    patch_size = 256
    audio_dim = n_mels * T_max
    model_type = args.model_type.lower()

    if model_type == "mlp":
        # If text_data was actually provided, text_dim > 0.
        # Otherwise, text_dim remains 0 and we do not concatenate any text at all.
        input_dim = audio_dim + text_dim
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, args.num_classes),
        )

    elif model_type == "cnn":
        model = CNNModel(input_dim=audio_dim, num_classes=args.num_classes)

    elif model_type == "transformer":
        model = TransformerModel(input_dim=n_mels, num_classes=args.num_classes)

    elif model_type == "resnet":
        model = ResNet18(num_classes=args.num_classes)

    elif model_type == "cnnbilstm":
        model = CNNBiLSTMSER(num_classes=args.num_classes)

    elif model_type == "xfmr":
        model = ImprovedFusedTransformer(
            audio_dim = audio_dim,
            text_dim  = text_dim or 768,
            patch_size= patch_size,
            d_model   = 256,
            n_head    = 4,
            n_layer   = 2,
            n_classes = args.num_classes,
            dropout   = 0.1,
        )

    elif model_type == "audio_tf":
        model = AudioTransformer(
            n_mels   = n_mels,
            d_model  = 128,
            n_heads  = 4,
            n_layers = 4,
            num_classes=args.num_classes,
        )

    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # 5) Optimizer + Scheduler + Criterion
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=2
    )

    criterion = nn.CrossEntropyLoss()

    # 5.3) Early Stopping related variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = args.early_stop_patience

    def _prepare_batch(xb: torch.Tensor) -> torch.Tensor:
        """Ensure xb’s time dimension is divisible by patch_size/n_mels; if not, trim."""
        frames_per_patch = patch_size // xb.shape[1]  # e.g. 256 // 64 -> 4
        T_new = (xb.shape[2] // frames_per_patch) * frames_per_patch
        return xb[:, :, :T_new]

    # 6) Epoch loop
    for epoch in range(1, args.epochs + 1):
        # —— Train ——
        model.train()
        total_loss = 0.0

        for xb, txt, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            # For ResNet or CNN+BiLSTM, add a channel dimension.
            # For audio_tf (speech-only Transformer), remove that extra channel.
            if model_type in ("resnet", "cnnbilstm"):
                xb = xb.unsqueeze(1)    # shape: (B, 1, 64, 998)
            elif model_type == "audio_tf":
                xb = xb.squeeze(1)      # shape: (B, 64, 998)

            if model_type == "xfmr":
                # Cross‐attention fusion uses both xb and txt
                xb = _prepare_batch(xb)
                if not isinstance(txt, torch.Tensor):
                    # If txt is not a tensor (e.g. raw string), replace with zeros of shape (B, text_dim)
                    txt = torch.zeros(xb.size(0), text_dim or 768, device=device)
                txt = txt.to(device, non_blocking=True)
                logits = model(xb, txt)

            elif model_type == "mlp":
                # Early-fusion MLP: flatten the spectrogram and concatenate with text if available
                B = xb.size(0)
                flat_audio = xb.view(B, -1)  # (B, 64*998 = 63872)
                if text_dim > 0:
                    # We have valid text embeddings
                    if not isinstance(txt, torch.Tensor):
                        txt = torch.zeros(B, text_dim, device=device)
                    txt = txt.to(device, non_blocking=True)  # (B, text_dim)
                    mlp_input = torch.cat([flat_audio, txt], dim=1)  # (B, 63872+text_dim)
                else:
                    # No text_data provided, just use audio
                    mlp_input = flat_audio  # (B, 63872)

                logits = model(mlp_input)

            else:
                # cnn, transformer(text-only), resnet, audio_tf all consume only xb
                logits = model(xb)

            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch}/{args.epochs}] Train Loss: {avg_train_loss:.4f}")

        # —— Validation ——
        model.eval()
        preds, labels_all = [], []
        val_loss_sum = 0.0
        val_samples = 0

        with torch.no_grad():
            for xb, txt, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                if model_type in ("resnet", "cnnbilstm"):
                    xb = xb.unsqueeze(1)
                elif model_type == "audio_tf":
                    xb = xb.squeeze(1)

                if model_type == "xfmr":
                    xb = _prepare_batch(xb)
                    if not isinstance(txt, torch.Tensor):
                        txt = torch.zeros(xb.size(0), text_dim or 768, device=device)
                    txt = txt.to(device, non_blocking=True)
                    out = model(xb, txt)

                elif model_type == "mlp":
                    B = xb.size(0)
                    flat_audio = xb.view(B, -1)
                    if text_dim > 0:
                        if not isinstance(txt, torch.Tensor):
                            txt = torch.zeros(B, text_dim, device=device)
                        txt = txt.to(device, non_blocking=True)
                        mlp_input = torch.cat([flat_audio, txt], dim=1)
                    else:
                        mlp_input = flat_audio
                    out = model(mlp_input)

                else:
                    out = model(xb)

                loss = criterion(out, yb)
                val_loss_sum += loss.item() * xb.size(0)
                val_samples += xb.size(0)

                preds.extend(out.argmax(1).cpu().tolist())
                labels_all.extend(yb.cpu().tolist())

        avg_val_loss = val_loss_sum / val_samples
        compute_metrics(labels_all, preds)

        # —— Scheduler step ——
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # —— Early Stopping logic ——
        if args.use_early_stop:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Save the current best model
                torch.save(
                    model.state_dict(),
                    os.path.join(args.save_dir, f"best_{model_type}.pt")
                )
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stop_patience:
                print(
                    f"Validation loss did not improve for {early_stop_patience} epochs. "
                    f"Early stopping at epoch={epoch}."
                )
                break

        # —— Save checkpoint each epoch (optional) ——
        torch.save(
            model.state_dict(),
            os.path.join(args.save_dir, f"{model_type}_ep{epoch}.pt")
        )
        print(f"[Epoch {epoch}] current_lr={current_lr:.2e}, best_val_loss={best_val_loss:.4f}")

    # —— After training, load best weights if early stopping was used ——
    if args.use_early_stop:
        best_path = os.path.join(args.save_dir, f"best_{model_type}.pt")
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path))


if __name__ == "__main__":
    train(parse_args())

