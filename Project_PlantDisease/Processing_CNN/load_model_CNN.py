import torch
import timm
import os

def load_model_for_test(checkpoint_path,
                        num_classes,
                        device=None):

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    print(f"Loading model on device: {device}")

    model = timm.create_model(
        "efficientnet_b3",
        pretrained=False,
        num_classes=num_classes
    )

    # Use map_location to handle different devices
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # hỗ trợ cả 2 dạng checkpoint
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model
