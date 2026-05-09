import os
import sys

# Thêm project root vào sys.path để dùng path_helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from Pretrain_SAM.convertYOLO_sangSAM1 import get_best_bboxes
from utils.path_helper import get_absolute_path

# ==============================
# CONFIG - Dùng đường dẫn động
# ==============================

CHECKPOINT = get_absolute_path("Pretrain_SAM/model/sam_vit_b_01ec64.pth")
OUT_DIR     = get_absolute_path("Pretrain_SAM/SAM_Output/crops1")
PATH        = get_absolute_path("Pretrain_SAM/IMG_TEST_SAM")
LBL         = get_absolute_path("Train_YOLOv11/img_reference/labels")

# Tự động chọn device tốt nhất cho Mac và Windows
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

# ==============================
# LOAD SAM
# ==============================

print("Loading SAM...")
sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
sam.to(device=DEVICE)
sam.eval()

tasks = get_best_bboxes(PATH, LBL)
print(f"Total images to process: {len(tasks)}")

MAX_SIZE  = 640  # Tăng từ 520 lên 640 để giữ chi tiết tốt hơn
predictor = SamPredictor(sam)

# ==============================
# PROCESS EACH IMAGE
# ==============================

for i, task in enumerate(tasks):
    try:
        image_bgr = cv2.imread(task["img_path"])
        if image_bgr is None:
            print(f"Không đọc được ảnh: {task['img_path']}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        H, W = image_bgr.shape[:2]
        x1, y1, x2, y2 = task["bbox"]

        # -----------------
        # Crop & Resize
        # -----------------
        pad = int(max(x2-x1, y2-y1) * 0.25)
        x1_crop = max(0, int(x1 - pad))
        y1_crop = max(0, int(y1 - pad))
        x2_crop = min(W, int(x2 + pad))
        y2_crop = min(H, int(y2 + pad))

        crop_img = image_rgb[y1_crop:y2_crop, x1_crop:x2_crop]
        h_c, w_c = crop_img.shape[:2]
        scale = 1.0

        if max(h_c, w_c) > MAX_SIZE:
            scale = MAX_SIZE / max(h_c, w_c)
            crop_img = cv2.resize(crop_img, (int(w_c*scale), int(h_c*scale)))

        # -----------------
        # SAM Inference với Point Prompt
        # (Cải tiến: dùng multimask_output=True để chọn mask tốt nhất)
        # -----------------
        box_crop = np.array([
            (x1 - x1_crop) * scale,
            (y1 - y1_crop) * scale,
            (x2 - x1_crop) * scale,
            (y2 - y1_crop) * scale
        ], dtype=np.float32)

        # Point Prompt: trọng tâm của box
        cx = (box_crop[0] + box_crop[2]) / 2
        cy = (box_crop[1] + box_crop[3]) / 2
        input_point = np.array([[cx, cy]])
        input_label = np.array([1])

        # Dùng inference_mode cho tất cả device (không dùng cuda.amp.autocast)
        with torch.inference_mode():
            predictor.set_image(crop_img)
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=box_crop[None, :],
                multimask_output=True   # Lấy 3 mask, chọn mask tốt nhất
            )

        # Chọn mask có score cao nhất
        best_idx  = int(np.argmax(scores))
        mask_crop = masks[best_idx].astype(np.float32)
        predictor.reset_image()

        # -----------------
        # Anti-alias mask resize về kích thước gốc
        # -----------------
        crop_w = x2_crop - x1_crop
        crop_h = y2_crop - y1_crop

        mask_crop = cv2.resize(mask_crop, (crop_w*2, crop_h*2), interpolation=cv2.INTER_CUBIC)
        mask_crop = cv2.GaussianBlur(mask_crop, (7,7), 0)
        mask_crop = cv2.resize(mask_crop, (crop_w, crop_h), interpolation=cv2.INTER_AREA)
        mask_crop = (mask_crop > 0.5).astype(np.uint8)

        mask_full = np.zeros((H, W), dtype=np.uint8)
        mask_full[y1_crop:y2_crop, x1_crop:x2_crop] = mask_crop

        # -----------------
        # Post-process: Làm sạch mask
        # -----------------
        mask_u8 = (mask_full * 255).astype(np.uint8)

        # Dùng kernel lớn hơn để lấp đầy lỗ hổng trong mask (bệnh đốm gây ra)
        kernel  = np.ones((5, 5), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        mask_u8 = cv2.GaussianBlur(mask_u8, (5,5), 0)

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"Không tìm thấy contour: {task['filename']}")
            continue

        # Lấy tất cả contour đủ lớn (> 1% diện tích ảnh) thay vì chỉ lấy lớn nhất
        # Điều này giúp không bỏ sót các đốm bệnh tách rời
        min_area    = H * W * 0.01
        large_conts = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not large_conts:
            large_conts = [max(contours, key=cv2.contourArea)]  # fallback

        clean_mask = np.zeros_like(mask_u8)
        cv2.drawContours(clean_mask, large_conts, -1, 255, -1)

        # -----------------
        # Feather Alpha + Apply to Image
        # -----------------
        mask_f   = clean_mask.astype(np.float32) / 255.0
        mask_f   = cv2.GaussianBlur(mask_f, (7,7), 0)
        mask_3   = cv2.merge([mask_f, mask_f, mask_f])

        # Nền trắng thay vì nền đen
        white_bg  = np.ones_like(image_bgr, dtype=np.float32) * 255
        segmented = image_bgr.astype(np.float32) * mask_3 + white_bg * (1 - mask_3)
        segmented = segmented.astype(np.uint8)

        # -----------------
        # Crop lá + lưu ảnh
        # -----------------
        all_pts = np.vstack(large_conts)
        rx, ry, rw, rh = cv2.boundingRect(all_pts)

        final_pad = int(max(rw, rh) * 0.08)
        fx1 = max(0, rx - final_pad)
        fy1 = max(0, ry - final_pad)
        fx2 = min(W, rx + rw + final_pad)
        fy2 = min(H, ry + rh + final_pad)

        crop_leaf = segmented[fy1:fy2, fx1:fx2]

        save_path = os.path.join(OUT_DIR, f"crop_{i:04d}_{task['filename']}")
        cv2.imwrite(save_path, crop_leaf, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # -----------------
        # Giải phóng RAM
        # -----------------
        del masks, scores, mask_crop, mask_full, mask_u8, clean_mask, segmented, crop_leaf
        if DEVICE == "cuda" and i % 5 == 0:
            torch.cuda.empty_cache()

        print(f"Done [{i+1}/{len(tasks)}]: {task['filename']} -> {save_path}")

    except Exception as e:
        print(f"Error on {task.get('filename', '?')}: {e}")

print("\nFinished! All images saved to:", OUT_DIR)