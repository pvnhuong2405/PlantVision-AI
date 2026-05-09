import os
import torch
import torch.nn.functional as F
from load_model_test import load_model_for_test
from handle_test import handle_data


TEST_DIR = r"D:/Train_CNN/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/test"
WEIGHT_PATH = r"D:/Train_CNN/weight/best.pt"
NUM_CLASSES = 38
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
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
