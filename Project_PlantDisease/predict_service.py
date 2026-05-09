import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

from Processing_CNN.load_model_CNN import load_model_for_test
from Processing_CNN.handle_CNN import handle_data
from utils.path_helper import get_absolute_path

# ==============================
# CONFIG & DEVICE
# ==============================

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# Paths using path_helper
YOLO_WEIGHT = get_absolute_path("Train_YOLOv11/weights_yolov11s/best.pt")
SAM_CHECKPOINT = get_absolute_path("Pretrain_SAM/model/sam_vit_b_01ec64.pth")
PLANT_WEIGHT = get_absolute_path("Train_CNN_Classes/weight/best.pt")
DISEASE_WEIGHT = get_absolute_path("Train_CNN/weight/best.pt")

PLANT_CLASSES = 13
DISEASE_CLASSES = 38
PLANT_THRESHOLD = 0.5
DISEASE_THRESHOLD = 0.5

# ==============================
# LOAD MODELS
# ==============================

print("Loading YOLOv11...")
yolo_model = YOLO(YOLO_WEIGHT)

print("Loading SAM...")
sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

print("Loading CNN Models...")
plant_model = load_model_for_test(PLANT_WEIGHT, PLANT_CLASSES, DEVICE)
disease_model = load_model_for_test(DISEASE_WEIGHT, DISEASE_CLASSES, DEVICE)

plant_model.eval()
disease_model.eval()

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

# Mapping
plant_to_disease_indices = {}
for idx, disease_name in enumerate(disease_class_names):
    plant_name = disease_name.split("___")[0]
    plant_to_disease_indices.setdefault(plant_name, []).append(idx)

# ==============================
# HELPER FUNCTIONS
# ==============================

def get_segmented_leaf(image_bgr, bbox):
    """Uses SAM to segment the leaf and returns a white-background crop."""
    try:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        sam_predictor.set_image(image_rgb)

        x1, y1, x2, y2 = bbox
        input_box = np.array([x1, y1, x2, y2])

        # Point Prompt: trọng tâm của bbox
        input_point = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
        input_label = np.array([1])

        # Wrap trong inference_mode để tiết kiệm bộ nhớ và tăng tốc
        with torch.inference_mode():
            masks, scores, _ = sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box[None, :],
                multimask_output=True  # Lấy 3 mask, chọn mask tốt nhất
            )

        # Chọn mask có score cao nhất
        best_mask_idx = int(np.argmax(scores))
        mask = masks[best_mask_idx]

        # Kiểm tra mask hợp lệ: phải chiếm ít nhất 5% diện tích bbox
        bbox_area  = (x2 - x1) * (y2 - y1)
        mask_area  = mask.sum()
        if mask_area < bbox_area * 0.05:
            print(f"SAM: Mask quá nhỏ ({mask_area}/{bbox_area:.0f}px). Fallback to YOLO crop.")
            return None

        # Tạo nền trắng
        white_bg  = np.ones_like(image_bgr) * 255
        mask_3ch  = np.stack([mask] * 3, axis=-1)
        segmented = np.where(mask_3ch, image_bgr, white_bg)

        # Crop tới bbox với padding nhỏ
        h, w  = image_bgr.shape[:2]
        pad   = 10
        rx1   = max(0, int(x1 - pad))
        ry1   = max(0, int(y1 - pad))
        rx2   = min(w, int(x2 + pad))
        ry2   = min(h, int(y2 + pad))

        crop = segmented[ry1:ry2, rx1:rx2]
        sam_predictor.reset_image()
        return crop

    except Exception as e:
        print(f"SAM Error: {e}. Falling back to YOLO crop.")
        sam_predictor.reset_image()
        return None

# ==============================
# PREDICTION FUNCTIONS
# ==============================

def predict_plant(image_tensor):
    with torch.no_grad():
        outputs = plant_model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
        plant_name = plant_class_names[pred.item()]
        plant_conf = conf.item()
    
    return {"plant": plant_name, "plant_conf": plant_conf}

def predict_disease(image_tensor, plant_name):
    with torch.no_grad():
        outputs = disease_model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        valid_indices = plant_to_disease_indices[plant_name]
        subset = probs[0][valid_indices]
        if subset.sum() == 0: return None
        subset = subset / subset.sum()
        best_idx = torch.argmax(subset).item()
        disease_index = valid_indices[best_idx]
        return {
            "disease": disease_class_names[disease_index],
            "disease_conf": subset[best_idx].item()
        }


def _apply_white_background(image_bgr, threshold=30):
    """
    Chuyển nền tối thành nền trắng.
    CNN được train trên ảnh lá trên nền trắng, nên cần xử lý nền
    để kết quả chính xác hơn.
    
    Sử dụng 2 phương pháp:
    1. HSV: phát hiện vùng xanh lá (leaf) và giữ lại
    2. Brightness: các pixel quá tối (nền đất/cành) → trắng
    """
    # Chuyển sang HSV để phát hiện vùng xanh lá
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Phạm vi màu xanh lá trong HSV (mở rộng để bắt cả lá bệnh vàng/nâu)
    lower_green = np.array([15, 20, 30])   # Mở rộng hue để bắt cả nâu
    upper_green = np.array([95, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Bắt thêm vùng nâu/vàng (đốm bệnh)
    lower_brown = np.array([5, 30, 40])
    upper_brown = np.array([25, 255, 200])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Kết hợp 2 mask
    mask = cv2.bitwise_or(mask_green, mask_brown)
    
    # Morphology: lấp lỗ nhỏ, nối vùng gần nhau
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Làm mượt biên
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = (mask > 127).astype(np.uint8)
    
    # Áp dụng: giữ lá, nền → trắng
    mask_3ch = np.stack([mask] * 3, axis=-1)
    white_bg = np.ones_like(image_bgr, dtype=np.uint8) * 255
    result = np.where(mask_3ch, image_bgr, white_bg)
    
    return result


def _run_cnn_on_image(image_bgr_crop, source_label):
    """Chạy CNN (plant + disease) trên 1 ảnh crop. Trả về dict kết quả."""
    leaf_pil = Image.fromarray(cv2.cvtColor(image_bgr_crop, cv2.COLOR_BGR2RGB))
    loader = handle_data(leaf_pil)
    for images, _ in loader:
        image_tensor = images.to(DEVICE)

    plant_res = predict_plant(image_tensor)
    disease_res = predict_disease(image_tensor, plant_res["plant"])

    if not disease_res:
        return None

    is_healthy = "healthy" in disease_res["disease"].lower()

    return {
        "source": source_label,
        "plant": plant_res["plant"],
        "plant_conf": plant_res["plant_conf"],
        "disease": disease_res["disease"],
        "disease_conf": disease_res["disease_conf"],
        "is_healthy": is_healthy,
    }


def _choose_best_result(sam_result, yolo_result):
    """
    Logic bỏ phiếu Dual Pipeline:
    - Nếu 1 bên nói bệnh, 1 bên nói khỏe → ƯU TIÊN bên nói BỆNH
    - Nếu cả 2 nói bệnh → chọn bệnh có confidence cao hơn
    - Nếu cả 2 nói khỏe → chọn bên có confidence cao hơn
    """
    if sam_result is None and yolo_result is None:
        return None, "both_failed"
    if sam_result is None:
        return yolo_result, "yolo_only"
    if yolo_result is None:
        return sam_result, "sam_only"

    sam_healthy  = sam_result["is_healthy"]
    yolo_healthy = yolo_result["is_healthy"]

    # Trường hợp 1: Một bên bệnh, một bên khỏe → ƯU TIÊN BỆNH
    if sam_healthy and not yolo_healthy:
        return yolo_result, "yolo_detected_disease"
    if not sam_healthy and yolo_healthy:
        return sam_result, "sam_detected_disease"

    # Trường hợp 2: Cả hai đều nói bệnh → chọn confidence cao hơn
    if not sam_healthy and not yolo_healthy:
        if sam_result["disease_conf"] >= yolo_result["disease_conf"]:
            return sam_result, "both_diseased_sam_wins"
        else:
            return yolo_result, "both_diseased_yolo_wins"

    # Trường hợp 3: Cả hai đều khỏe → chọn confidence cao hơn
    if sam_result["disease_conf"] >= yolo_result["disease_conf"]:
        return sam_result, "both_healthy_sam_wins"
    else:
        return yolo_result, "both_healthy_yolo_wins"


# ==============================
# MAIN PIPELINE (DUAL + MULTI-LEAF)
# ==============================

MAX_LEAVES = 5  # Tối đa phân tích 5 lá

def predict_pipeline(image_path):
    # 1. Load Image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return {"status": "error", "message": "Could not read image"}

    # 2. YOLO Detection
    results = yolo_model(image_path, verbose=False)
    if len(results[0].boxes) == 0:
        return {"status": "no_leaf_detected", "message": "No leaf detected by YOLO"}

    # Lấy tất cả boxes, sắp xếp theo confidence giảm dần
    boxes = results[0].boxes
    confs = boxes.conf.cpu().numpy()
    sorted_indices = np.argsort(confs)[::-1][:MAX_LEAVES]

    all_leaf_results = []
    h_img, w_img = image_bgr.shape[:2]

    for leaf_idx, box_idx in enumerate(sorted_indices):
        bbox = boxes[box_idx].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, bbox)
        pad = 10

        # --- Luồng 1: SAM → CNN ---
        sam_crop = get_segmented_leaf(image_bgr, bbox)
        sam_res = None
        if sam_crop is not None:
            sam_res = _run_cnn_on_image(sam_crop, f"SAM_leaf{leaf_idx}")

        # --- Luồng 2: YOLO crop + white bg → CNN ---
        yolo_crop = image_bgr[
            max(0, y1 - pad) : min(h_img, y2 + pad),
            max(0, x1 - pad) : min(w_img, x2 + pad)
        ]
        yolo_crop_white = _apply_white_background(yolo_crop)
        yolo_res = _run_cnn_on_image(yolo_crop_white, f"YOLO_leaf{leaf_idx}")

        # Chọn kết quả tốt nhất cho lá này
        best_leaf, reason = _choose_best_result(sam_res, yolo_res)
        if best_leaf is not None:
            best_leaf["leaf_index"] = leaf_idx
            best_leaf["decision"] = reason
            all_leaf_results.append(best_leaf)

    if not all_leaf_results:
        return {"status": "error", "message": "CNN failed on all leaves"}

    # ==============================
    # TỔNG HỢP: Chọn kết quả cuối cùng
    # ==============================
    # Ưu tiên: nếu BẤT KỲ lá nào bị bệnh → cây bị bệnh
    diseased = [r for r in all_leaf_results if not r["is_healthy"]]
    healthy  = [r for r in all_leaf_results if r["is_healthy"]]

    if diseased:
        # Chọn kết quả bệnh có confidence cao nhất
        final = max(diseased, key=lambda r: r["disease_conf"])
        status = "ok"
        if final["disease_conf"] < DISEASE_THRESHOLD:
            status = "uncertain"
    else:
        # Tất cả lá đều khỏe mạnh
        final = max(healthy, key=lambda r: r["disease_conf"])
        status = "healthy"

    return {
        "status": status,
        "segmentation": final.get("decision", "N/A"),
        "plant": final["plant"],
        "plant_conf": final["plant_conf"],
        "disease": final["disease"],
        "disease_conf": final["disease_conf"],
        # Debug
        "leaves_analyzed": len(all_leaf_results),
        "leaves_diseased": len(diseased),
        "leaves_healthy": len(healthy),
        "debug_all_leaves": all_leaf_results,
        "decision_reason": f"{len(diseased)}_diseased/{len(all_leaf_results)}_total",
    }