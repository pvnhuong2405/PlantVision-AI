import os
import cv2
import numpy as np


def bbox_quality_score(bbox, img_w, img_h):

    x1, y1, x2, y2 = bbox

    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    area = w * h
    area_ratio = area / (img_w * img_h)

    # Aspect ratio score (không bias hình vuông)
    aspect_ratio = w / h
    aspect_score = 1 / (1 + abs(aspect_ratio - 1))

    # Edge penalty nhẹ hơn
    margin = 5
    edge_penalty = 0

    if x1 < margin or y1 < margin or x2 > img_w - margin or y2 > img_h - margin:
        edge_penalty = 0.15

    score = area_ratio * aspect_score * (1 - edge_penalty)

    return score


def adaptive_expand_bbox(bbox, img_w, img_h):

    x1, y1, x2, y2 = bbox

    w = x2 - x1
    h = y2 - y1

    area = w * h
    area_ratio = area / (img_w * img_h)

    # Expand mạnh hơn cho bbox nhỏ
    if area_ratio < 0.05:
        expand_ratio = 0.35
    elif area_ratio < 0.2:
        expand_ratio = 0.18
    else:
        expand_ratio = 0.10

    expand_w = int(w * expand_ratio)
    expand_h = int(h * expand_ratio)

    x1 = max(0, x1 - expand_w)
    y1 = max(0, y1 - expand_h)
    x2 = min(img_w, x2 + expand_w)
    y2 = min(img_h, y2 + expand_h)

    return [x1, y1, x2, y2]


def get_best_bboxes(input_path, label_dir):

    data_list = []

    input_path = os.path.abspath(input_path)
    label_dir = os.path.abspath(label_dir)

    if os.path.isfile(input_path):

        files_to_process = [os.path.basename(input_path)]
        img_dir = os.path.dirname(input_path)

    elif os.path.isdir(input_path):

        files_to_process = os.listdir(input_path)
        img_dir = input_path

    else:

        print(f"Lỗi: Đường dẫn không hợp lệ -> {input_path}")
        return []


    for filename in files_to_process:

        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(img_dir, filename)

        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(label_dir, txt_filename)

        if not os.path.exists(txt_path):
            continue

        img = cv2.imread(img_path)

        if img is None:
            continue

        H, W = img.shape[:2]

        with open(txt_path, "r") as f:

            lines = [l.strip() for l in f.readlines() if l.strip()]


        if not lines:
            continue


        bboxes = []

        for line in lines:

            parts = line.split()

            if len(parts) >= 5:

                cls, xc, yc, w, h = map(float, parts[:5])

                x1 = int((xc - w/2) * W)
                y1 = int((yc - h/2) * H)
                x2 = int((xc + w/2) * W)
                y2 = int((yc + h/2) * H)

                bboxes.append([x1, y1, x2, y2])


        if not bboxes:
            continue


        # ---- CHỌN BBOX TỐT NHẤT ----
        best_bbox = max(bboxes, key=lambda b: bbox_quality_score(b, W, H))


        # ---- EXPAND BBOX ----
        best_bbox = adaptive_expand_bbox(best_bbox, W, H)


        print("-" * 50)
        print(f"File: {filename}")
        print(f" -> Phát hiện: {len(bboxes)} bbox(es)")
        print(f" -> Bbox tốt nhất (adaptive expanded): {best_bbox}")


        data_list.append({

            "img_path": img_path,
            "filename": filename,
            "bbox": best_bbox

        })


    return data_list


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.path_helper import get_absolute_path

    PATH    = get_absolute_path("Pretrain_SAM/IMG_TEST_SAM")
    LBL     = get_absolute_path("Train_YOLOv11/img_reference/labels")

    results = get_best_bboxes(PATH, LBL)

    print("\n" + "="*50)
    print(f"TỔNG KẾT: Đã sẵn sàng {len(results)} ảnh để đưa vào SAM.")
    print("="*50)