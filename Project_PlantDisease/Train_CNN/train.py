import torch
import os
import eff_train as etr
import eff_val as ev
import Checkpoint as cb
from processing_data import build_dataloader
import resume as rs
from torch.amp import GradScaler

#EPOCHS = 20
# 40
# 60
# 80
# 100
TOTAL_EPOCHS = 60

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =====================
    # DATA
    # =====================
    train_loader, val_loader = build_dataloader()

    # =====================
    # MODEL / LOSS / OPT
    # =====================
    model = etr.model.to(device)
    criterion = etr.criterion
    optimizer = etr.optimizer

    scaler = GradScaler("cuda")

    best_acc = 0.0
    start_epoch = 0

    # =====================
    # RESUME (NẾU CÓ)
    # =====================
    if os.path.exists("D:/Train_CNN/weight/last.pt"):
        start_epoch, best_acc = rs.load_checkpoint(
            "D:/Train_CNN/weight/last.pt", model, optimizer, scaler
        )
        print(f"Resume from epoch {start_epoch}, best_acc={best_acc:.4f}")

    # =====================
    # TRAIN LOOP
    # =====================
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        train_loss = etr.train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device
        )

        val_loss, val_acc = ev.validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f} | "
            f"Acc {val_acc:.4f}"
        )

        # Save last checkpoint
        cb.save_checkpoint(
            epoch,
            model,
            optimizer,
            scaler,
            best_acc,
            "D:/Train_CNN/weight/last.pt"
        )

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            cb.save_checkpoint(
                epoch,
                model,
                optimizer,
                scaler,
                best_acc,
                "D:/Train_CNN/weight/best.pt"
            )

        # =====================
        # UNFREEZE BACKBONE
        # =====================
        if epoch == 5 and start_epoch <= 5:
            print("Unfreezing backbone...")
            for p in model.parameters():
                p.requires_grad = True

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=1e-4
            )

        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
