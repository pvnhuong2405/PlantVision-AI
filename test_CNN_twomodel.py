import os
import torch
import torch.nn.functional as F
from Train_CNN.load_model_test import load_model_for_test
from Train_CNN.handle_test import handle_data
# import glob 
# ==============================
# CONFIG
# ==============================

TEST_PATH = "D:/Project_PlantDisease/IMG_TEST_CNN_AfterSAM"

PLANT_WEIGHT = "D:/Project_PlantDisease/Train_CNN_Classes/weight/best.pt"
DISEASE_WEIGHT = "D:/Project_PlantDisease/Train_CNN/weight/best.pt"

PLANT_CLASSES = 13
DISEASE_CLASSES = 38

BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# paths = glob.glob(TEST_PATH)
# ==============================
# CLASS NAMES
# ==============================

plant_class_names = [
    "Apple", "Blueberry", "Cherry", "Corn", "Grape",
    "Orange", "Peach", "Pepper,_bell", "Potato",
    "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato"
]

disease_class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight", "Corn___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

# ==============================
# BUILD DISEASE INDEX MAP
# ==============================

plant_to_disease_indices = {}

for idx, disease_name in enumerate(disease_class_names):
    plant_name = disease_name.split("___")[0]
    if plant_name not in plant_to_disease_indices:
        plant_to_disease_indices[plant_name] = []
    plant_to_disease_indices[plant_name].append(idx)

# ==============================
# TEST FUNCTION
# ==============================

def test_two_stage():

    print(f"--- Load Plant Model ---")
    plant_model = load_model_for_test(
        checkpoint_path=PLANT_WEIGHT,
        num_classes=PLANT_CLASSES,
        device=DEVICE
    )
    plant_model.eval()

    print(f"--- Load Disease Model ---")
    disease_model = load_model_for_test(
        checkpoint_path=DISEASE_WEIGHT,
        num_classes=DISEASE_CLASSES,
        device=DEVICE
    )
    disease_model.eval()

    test_loader = handle_data(
        path=TEST_PATH,
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    print("\n--- START INFERENCE ---")

    with torch.no_grad():
        for images, paths in test_loader:

            images = images.to(DEVICE)

            # ===== Stage 1: Predict Plant =====
            plant_outputs = plant_model(images)
            plant_probs = F.softmax(plant_outputs, dim=1)
            plant_confs, plant_preds = torch.max(plant_probs, dim=1)

            # ===== Stage 2: Predict Disease =====
            disease_outputs = disease_model(images)
            disease_probs = F.softmax(disease_outputs, dim=1)

            for i in range(len(images)):

                img_path = paths[i]

                predicted_plant = plant_class_names[plant_preds[i].item()]
                plant_conf = plant_confs[i].item()

                # Lấy index các disease thuộc plant đó
                valid_indices = plant_to_disease_indices[predicted_plant]

                # Lọc chỉ disease thuộc plant đó
                #filtered_probs = disease_probs[i][valid_indices]
                filtered_probs = disease_probs[i][valid_indices]
                filtered_probs = filtered_probs / filtered_probs.sum()

                best_idx_in_subset = torch.argmax(filtered_probs).item()
                best_disease_index = valid_indices[best_idx_in_subset]

                predicted_disease = disease_class_names[best_disease_index]
                disease_conf = filtered_probs[best_idx_in_subset].item()

                print("\n----------------------------------")
                print(f"Image   : {os.path.basename(img_path)}")
                print(f"Plant   : {predicted_plant} ({plant_conf*100:.2f}%)")
                print(f"Disease : {predicted_disease} ({disease_conf*100:.2f}%)")

    print("\nInference completed.")

# ==============================

if __name__ == "__main__":
    test_two_stage()
