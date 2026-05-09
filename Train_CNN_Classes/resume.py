import torch
import eff_train as tr


# def load_checkpoint(path, model, optimizer):
#     ckpt = torch.load(path)
#     model.load_state_dict(ckpt["model"])
#     optimizer.load_state_dict(ckpt["optimizer"])
#     return ckpt["epoch"], ckpt["best_acc"]
def load_checkpoint(path, model, optimizer, scaler):
    ckpt = torch.load(path, map_location=tr.device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])

    start_epoch = ckpt["epoch"] + 1
    best_acc = ckpt.get("best_acc", 0.0)

    return start_epoch, best_acc

