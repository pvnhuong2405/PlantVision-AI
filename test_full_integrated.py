import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict_service import predict_pipeline
from API_Gemini.call_api import get_disease_advice

def format_disease_name(disease_label):
    """Tách tên bệnh từ format Plant___Disease"""
    if "___" in disease_label:
        plant, disease = disease_label.split("___")
        return plant, disease
    return disease_label, disease_label

def main():
    print("\n" + "="*52)
    print("   PLANT DISEASE INTEGRATED PIPELINE TEST")
    print("="*52)

    while True:
        # ======================================
        # BƯỚC 1: Nhập đường dẫn ảnh
        # ======================================
        image_path = input("\nNhập đường dẫn ảnh (q để thoát): ").strip()

        # Handle Mac path dragging (single quotes)
        if image_path.startswith("'") and image_path.endswith("'"):
            image_path = image_path[1:-1]

        if image_path.lower() == "q":
            print("Thoát chương trình.")
            break

        if not os.path.exists(image_path):
            print(f"Loi: Khong tim thay file tai {image_path}")
            continue

        # ======================================
        # BƯỚC 2: Phân tích YOLO -> SAM -> CNN
        # ======================================
        print("\nDang phan tich (YOLO -> SAM -> CNN)...")
        t0 = time.time()
        result = predict_pipeline(image_path)
        elapsed = time.time() - t0

        # ======================================
        # BƯỚC 3: Hiển thị kết quả CNN ngay
        # ======================================
        print("\n--- CNN RESULT ---")

        if result["status"] in ["ok", "healthy", "uncertain"]:
            print(f"\n  [KET QUA CHON] {result['disease']} ({round(result['disease_conf']*100,1)}%)")
            print(f"  [TONG HOP]     {result.get('decision_reason', 'N/A')}")
            print(f"  [SO LA]        {result.get('leaves_analyzed', 0)} la phan tich | {result.get('leaves_diseased', 0)} benh | {result.get('leaves_healthy', 0)} khoe")

            # Debug: chi tiết từng lá
            all_leaves = result.get("debug_all_leaves", [])
            if all_leaves:
                print("\n  --- Chi tiet tung la ---")
                for leaf in all_leaves:
                    marker = "BENH" if not leaf["is_healthy"] else "KHOE"
                    print(f"  [La {leaf.get('leaf_index',0)}] [{leaf['source']}] {leaf['disease']} | {round(leaf['disease_conf']*100,1)}% | {marker}")

            print(f"\n(Thoi gian xu ly: {elapsed:.2f}s)")
        elif result["status"] == "no_leaf_detected":
            print("Khong tim thay la cay nao trong anh.")
            print("Vui long chup anh ro net hon va gan la hon.")
            continue
        else:
            print(f"Loi: {result.get('message', 'Unknown error')}")
            continue

        # ======================================
        # BƯỚC 4: Cho người dùng đặt câu hỏi
        # TRƯỚC KHI Gemini trả lời (giống image-1.png)
        # ======================================
        while True:
            user_question = input("\nBan muon hoi them gi? (enter de doi anh): ").strip()

            if user_question == "":
                break  # Thoát vòng lặp hỏi, quay lại nhập ảnh mới

            # ======================================
            # BƯỚC 5: Gửi câu hỏi đến Gemini (STREAMING)
            # ======================================
            print("\n--- CHATBOT ---")
            # stream=True: Gemini in token realtime trực tiếp ra terminal
            # Với các trường hợp đặc biệt (healthy/error), trả về string tĩnh
            advice = get_disease_advice(result, user_question, stream=True)
            # Chỉ in nếu advice là string tĩnh (healthy/error không qua stream)
            if advice and not result["status"] in ["ok", "uncertain"]:
                print(advice)

if __name__ == "__main__":
    main()
