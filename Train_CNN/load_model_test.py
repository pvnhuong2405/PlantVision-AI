import torch
import torch.nn as nn
import timm


def load_model_for_test(checkpoint_path, num_classes, device="cuda"):
    model = timm.create_model(
        "efficientnet_b3",
        pretrained=False,
        num_classes=num_classes
    )

    # 2. Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model"])

    # 3. Eval
    model.to(device)
    model.eval()

    return model
