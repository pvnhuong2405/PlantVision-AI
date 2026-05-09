import os
import torch
import torch.nn.functional as F
from load_model_test import load_model_for_test
from handle_test import handle_data


TEST_DIR = "D:/Project_PlantDisease/Train_CNN_Classes/Dataset_new/test"
WEIGHT_PATH = "D:/Project_PlantDisease/Train_CNN_Classes/weight/best.pt"
NUM_CLASSES = 13
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class_names = [
   "Apple", "Blueberry"," Cherry_(including_sour)", "Corn_(maize)", "Grape", "Orange", "Peach", "Pepper,_bell", "Potato", "Raspberry", "Soybean", "Strawberry", "Tomato"
]
# =========================================


def test_model():
    # 1. Load model
    model = load_model_for_test(
        checkpoint_path=WEIGHT_PATH,
        num_classes=NUM_CLASSES,
        device=DEVICE
    )

    # 2. Load test data
    test_loader = handle_data(
        path=TEST_DIR,
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    # 3. Inference
    model.eval()
    with torch.no_grad():
        for images, paths in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            confs, preds = torch.max(probs, dim=1)

            for img_path, pred, conf in zip(paths, preds, confs):
                print(
                    f"{os.path.basename(img_path)} "
                    f"-> {class_names[pred.item()]} "
                    f"({conf.item() * 100:.2f}%)"
                )


if __name__ == "__main__":
    test_model()
