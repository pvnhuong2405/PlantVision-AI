# PlantVision AI — Hệ thống chẩn đoán bệnh cây trồng thông minh

> **Pipeline tích hợp**: YOLOv11 (Phát hiện) → SAM (Phân đoạn) → EfficientNet-B3 (Phân loại) → Gemini AI (Tư vấn chuyên gia)

---

##  Mục lục

- [Tổng quan](#-tổng-quan)
- [Kiến trúc Pipeline](#-kiến-trúc-pipeline)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Cài đặt](#-cài-đặt)
- [Cách chạy](#-cách-chạy)
- [Giải thích Pipeline chi tiết](#-giải-thích-pipeline-chi-tiết)
- [API Endpoints](#-api-endpoints)
- [Kết quả demo](#-kết-quả-demo)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)

---

## Tổng quan

PlantVision AI là hệ thống chẩn đoán bệnh cây trồng hoàn toàn tự động, sử dụng 4 model AI phối hợp:

| Bước | Model | Nhiệm vụ |
|------|-------|----------|
| 1 | **YOLOv11s** | Phát hiện vị trí từng chiếc lá trong ảnh |
| 2 | **SAM (vit_b)** | Phân đoạn (tách lá khỏi nền) |
| 3 | **EfficientNet-B3** | Phân loại cây (13 loại) + Phân loại bệnh (38 loại) |
| 4 | **Gemini AI** | Tư vấn điều trị + Trả lời câu hỏi (Streaming realtime) |

**Hỗ trợ**: 13 loại cây trồng, 38 loại bệnh, chạy được trên **macOS (Apple Silicon MPS)**, **Windows (CUDA)** và **CPU**.

---

## Kiến trúc Pipeline

```
                        Ảnh đầu vào
                            │
                     ┌──────▼──────┐
                     │  YOLOv11s   │  Phát hiện tối đa 5 lá
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              │                           │
      ┌───────▼───────┐          ┌────────▼────────┐
      │   SAM (vit_b) │          │  YOLO Crop +    │
      │   Nền trắng   │          │  HSV White BG   │
      └───────┬───────┘          └────────┬────────┘
              │                           │
      ┌───────▼───────┐          ┌────────▼────────┐
      │ EfficientNet  │          │  EfficientNet   │
      │   (CNN #1)    │          │    (CNN #2)     │
      └───────┬───────┘          └────────┬────────┘
              │                           │
              └──────────┬────────────────┘
                         │
                  ┌──────▼──────┐
                  │  BỎ PHIẾU   │  Dual Pipeline Voting
                  │  (Voting)   │  Ưu tiên kết quả "Bệnh"
                  └──────┬──────┘
                         │
                  ┌──────▼──────┐
                  │  Gemini AI  │  Streaming realtime
                  │  (Tư vấn)   │
                  └─────────────┘
```

### Tại sao cần Dual Pipeline?

CNN được train trên **ảnh lá đơn lẻ trên nền trắng**. Khi nhận ảnh thực tế (nhiều lá, nền phức tạp):
- **SAM** cắt chính xác biên lá nhưng có thể **bỏ sót đốm bệnh** ở rìa
- **YOLO crop + White BG** giữ toàn bộ thông tin nhưng có thể bị **nhiễu nền**

→ Chạy song song cả 2, rồi **bỏ phiếu**: nếu bất kỳ luồng nào phát hiện bệnh → ưu tiên kết quả bệnh.

### Tại sao cần Multi-Leaf?

Ảnh thực tế thường chụp **cả cây** với nhiều lá. Nếu chỉ phân tích 1 lá (lá lớn nhất) có thể trúng vào lá khỏe → sai.

→ Phân tích tối đa **5 lá**, nếu **bất kỳ lá nào** bị bệnh → kết luận cây bị bệnh.

---

## Cấu trúc thư mục

```
Project_PlantDisease/
│
├── predict_service.py          # Pipeline chính (Dual + Multi-Leaf)
├── test_full_integrated.py     # File chạy test qua terminal
│
├── Fast_api/
│   ├── main.py                 # FastAPI server (REST API + SSE Streaming)
│   └── index.html              # Giao diện chatbot (Dark mode)
│
├── API_Gemini/
│   ├── call_api.py             # Gemini AI (Fallback models + Streaming)
│   └── .env                    # GEMINI_API_KEY (Không push lên GitHub!)
│
├── Processing_CNN/
│   ├── handle_CNN.py           # Tiền xử lý ảnh cho CNN (Albumentations)
│   └── load_model_CNN.py       # Load EfficientNet-B3 (CUDA/MPS/CPU)
│
├── Pretrain_SAM/
│   ├── test.py                 # SAM standalone test
│   ├── convertYOLO_sangSAM1.py # Chuyển đổi bbox YOLO → SAM format
│   └── model/
│       └── sam_vit_b_01ec64.pth  # Trọng số SAM (~375MB)
│
├── Train_YOLOv11/
│   └── weights_yolov11s/
│       └── best.pt             # Trọng số YOLO (~25MB)
│
├── Train_CNN/
│   └── weight/
│       └── best.pt             # Trọng số CNN phân loại bệnh
│
├── Train_CNN_Classes/
│   └── weight/
│       └── best.pt             # Trọng số CNN phân loại cây
│
├── utils/
│   ├── __init__.py
│   └── path_helper.py          # Quản lý đường dẫn đa nền tảng
│
├── requirements.txt
└── README1.md                  # File này
```

---

## ⚙ Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/<your-username>/PlantVision-AI.git
cd PlantVision-AI/Project_PlantDisease
```

### 2. Tạo môi trường ảo (khuyên dùng Anaconda)

```bash
conda create -n plant python=3.11 -y
conda activate plant
```

### 3. Cài đặt thư viện

```bash
pip install ultralytics segment-anything opencv-python-headless timm albumentations
pip install google-genai python-dotenv pillow tqdm numpy
pip install fastapi uvicorn python-multipart
```

### 4. Tải trọng số models

| Model | File | Kích thước | Đường dẫn |
|-------|------|-----------|-----------|
| SAM (vit_b) | `sam_vit_b_01ec64.pth` | ~375MB | `Pretrain_SAM/model/` |
| YOLOv11s | `best.pt` | ~25MB | `Train_YOLOv11/weights_yolov11s/` |
| CNN Bệnh | `best.pt` | ~46MB | `Train_CNN/weight/` |
| CNN Cây | `best.pt` | ~46MB | `Train_CNN_Classes/weight/` |

> **Lưu ý**: Các file `.pt` và `.pth` KHÔNG được push lên GitHub (đã thêm trong `.gitignore`). Cần tải riêng.

### 5. Cấu hình API Key

Tạo file `API_Gemini/.env`:

```env
GEMINI_API_KEY=your_api_key_here
```

Lấy API key tại: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

---

## Cách chạy

### Cách 1: Chạy qua Terminal (Test nhanh)

```bash
cd Project_PlantDisease
python test_full_integrated.py
```

**Luồng hoạt động:**
```
1. Nhập đường dẫn ảnh (hoặc kéo thả)
2. Pipeline phân tích (YOLO → SAM → CNN)
3. Hiển thị kết quả + so sánh từng lá
4. Nhập câu hỏi → Gemini trả lời streaming
5. Tiếp tục hỏi hoặc Enter để đổi ảnh
```

### Cách 2: Chạy Web UI (Giao diện chuyên nghiệp)

Đây là cách tốt nhất để demo sản phẩm với giao diện Chatbot hiện đại, hỗ trợ streaming realtime.

1. **Truy cập thư mục Fast_api**:
   ```bash
   cd Project_PlantDisease/Fast_api
   ```

2. **Khởi động Server**:
   ```bash
   python main.py
   ```
   *(Server sẽ tải các model AI vào RAM/GPU. Khi thấy dòng `Uvicorn running on http://0.0.0.0:8000` là sẵn sàng).*

3. **Truy cập giao diện**:
   Mở trình duyệt và truy cập: **[http://localhost:8000](http://localhost:8000)**

**Tính năng nổi bật trên Web UI:**
- **Kéo thả ảnh**: Hỗ trợ upload ảnh nhanh chóng bằng cách kéo thả.
- **Kết quả trực quan**: Hiển thị card chẩn đoán với thanh độ tin cậy và trạng thái bệnh.
- **Chatbot chuyên gia**: Gemini AI trả lời streaming (gõ từng chữ) về cách điều trị và phòng ngừa.
- **Multi-leaf Analysis**: Hệ thống tự động tổng hợp kết quả từ tối đa 5 lá cây khác nhau trong 1 ảnh để đưa ra kết luận chính xác nhất.

---

## Giải thích Pipeline chi tiết

### Bước 1: YOLO Detection (Phát hiện lá)

```python
# Phát hiện tối đa 5 lá, sắp xếp theo confidence
results = yolo_model(image_path, verbose=False)
sorted_indices = np.argsort(confs)[::-1][:MAX_LEAVES]
```

- Model: **YOLOv11s** (train trên dataset lá cây tùy chỉnh)
- Output: Tọa độ bbox `[x1, y1, x2, y2]` cho mỗi lá

### Bước 2: Dual Segmentation (Phân đoạn song song)

**Luồng SAM:**
```python
# SAM sử dụng Box Prompt + Point Prompt (trọng tâm)
masks, scores, _ = sam_predictor.predict(
    point_coords=center_point,
    point_labels=[1],
    box=bbox,
    multimask_output=True  # 3 mask → chọn score cao nhất
)
```

**Luồng YOLO + HSV White Background:**
```python
# Tách lá bằng HSV color masking
# Giữ lại pixel xanh lá + nâu (đốm bệnh)
# Nền → trắng
yolo_crop_white = _apply_white_background(yolo_crop)
```

### Bước 3: CNN Classification (Phân loại)

```python
# Model 1: Phân loại loại cây (13 classes)
plant_res = predict_plant(image_tensor)

# Model 2: Phân loại bệnh (38 classes)
# Chỉ xét các bệnh thuộc loại cây đã nhận diện
disease_res = predict_disease(image_tensor, plant_name)
```

### Bước 4: Voting (Bỏ phiếu)

```
Nếu 1 luồng nói "bệnh", 1 luồng nói "khỏe" → ƯU TIÊN "BỆNH"
Nếu cả 2 nói "bệnh" → chọn confidence CAO HƠN
Nếu cả 2 nói "khỏe" → chọn confidence CAO HƠN
Nếu BẤT KỲ lá nào bị bệnh → CÂY BỊ BỆNH
```

### Bước 5: Gemini AI (Tư vấn)

```python
# Fallback tự động qua 3 model
MODEL_FALLBACK_LIST = [
    "gemini-2.0-flash-lite",   # Ưu tiên (nhẹ, nhanh)
    "gemini-2.5-flash",        # Backup
    "gemini-2.0-flash-001",    # Backup cuối
]

# Streaming: in từng token realtime
for chunk in chat.send_message_stream(question):
    print(chunk.text, end="", flush=True)
```

---

## API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/` | Giao diện chatbot (HTML) |
| `POST` | `/predict` | Upload ảnh → Trả về JSON kết quả pipeline |
| `POST` | `/chat/stream` | Gửi câu hỏi → Stream câu trả lời (SSE) |

### POST /predict

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@leaf_image.jpg"
```

**Response:**
```json
{
  "status": "ok",
  "plant": "Pepper,_bell",
  "plant_conf": 0.99,
  "disease": "Pepper,_bell___Bacterial_spot",
  "disease_conf": 0.991,
  "leaves_analyzed": 5,
  "leaves_diseased": 3,
  "leaves_healthy": 2,
  "decision_reason": "3_diseased/5_total"
}
```

### POST /chat/stream

**Request:**
```bash
curl -X POST http://localhost:8000/chat/stream \
  -F "plant=Pepper,_bell" \
  -F "plant_conf=0.99" \
  -F "disease=Pepper,_bell___Bacterial_spot" \
  -F "disease_conf=0.99" \
  -F "question=Cách điều trị bệnh này?"
```

**Response (SSE Stream):**
```
data: {"token": "Chào "}
data: {"token": "bạn, "}
data: {"token": "bệnh "}
data: {"token": "đốm khuẩn..."}
data: {"done": true}
```

---

## Kết quả demo

### Test với ảnh `bacterial_spot.jpg` (ảnh cả cây ớt chuông)

```
=== KET QUA CUOI CUNG ===
CHON: Pepper,_bell___Bacterial_spot (99.1%)  
TONG HOP: 3_diseased/5_total

--- Chi tiết từng lá ---
[Lá 0] [YOLO_leaf0] healthy     | 97.8% | KHOE
[Lá 1] [SAM_leaf1]  healthy     | 99.9% | KHOE
[Lá 2] [SAM_leaf2]  Bacterial_spot | 89.5% | BENH 
[Lá 3] [SAM_leaf3]  Bacterial_spot | 99.1% | BENH  ← Được chọn
[Lá 4] [YOLO_leaf4] Bacterial_spot | 72.9% | BENH 
```

### Test với ảnh lá đơn (đã qua xử lý SAM)

```
CHON: Pepper,_bell___Bacterial_spot (100.0%)  
```

---

## Công nghệ sử dụng

| Thành phần | Công nghệ | Phiên bản |
|-----------|-----------|-----------|
| Object Detection | YOLOv11s (Ultralytics) | Latest |
| Segmentation | SAM (segment-anything, vit_b) | 1.0 |
| Classification | EfficientNet-B3 (timm) | Latest |
| AI Chatbot | Google Gemini (google-genai) | 2.0+ |
| Backend API | FastAPI + Uvicorn | 0.136+ |
| Image Processing | OpenCV + Albumentations | 4.10+ |
| Deep Learning | PyTorch | 2.2+ |
| Frontend | Vanilla HTML/CSS/JS | - |
| Cross-platform | MPS (Apple Silicon) / CUDA / CPU | - |

---

## Các loại cây và bệnh hỗ trợ

<details>
<summary><b>13 loại cây trồng</b> (click để xem)</summary>

1. Apple (Táo)
2. Blueberry (Việt quất)
3. Cherry (Anh đào)
4. Corn (Ngô)
5. Grape (Nho)
6. Orange (Cam)
7. Peach (Đào)
8. Pepper, bell (Ớt chuông)
9. Potato (Khoai tây)
10. Raspberry (Mâm xôi)
11. Soybean (Đậu nành)
12. Squash (Bí)
13. Strawberry (Dâu tây)
14. Tomato (Cà chua)

</details>

<details>
<summary><b>38 loại bệnh</b> (click để xem)</summary>

| STT | Bệnh | Cây |
|-----|-------|-----|
| 1 | Apple Scab | Apple |
| 2 | Black Rot | Apple |
| 3 | Cedar Apple Rust | Apple |
| 4 | Healthy | Apple |
| 5 | Healthy | Blueberry |
| 6 | Powdery Mildew | Cherry |
| 7 | Healthy | Cherry |
| 8 | Cercospora Leaf Spot | Corn |
| 9 | Common Rust | Corn |
| 10 | Northern Leaf Blight | Corn |
| 11 | Healthy | Corn |
| 12 | Black Rot | Grape |
| 13 | Esca (Black Measles) | Grape |
| 14 | Leaf Blight | Grape |
| 15 | Healthy | Grape |
| 16 | Huanglongbing | Orange |
| 17 | Bacterial Spot | Peach |
| 18 | Healthy | Peach |
| 19 | Bacterial Spot | Pepper |
| 20 | Healthy | Pepper |
| 21 | Early Blight | Potato |
| 22 | Late Blight | Potato |
| 23 | Healthy | Potato |
| 24 | Healthy | Raspberry |
| 25 | Healthy | Soybean |
| 26 | Powdery Mildew | Squash |
| 27 | Leaf Scorch | Strawberry |
| 28 | Healthy | Strawberry |
| 29 | Bacterial Spot | Tomato |
| 30 | Early Blight | Tomato |
| 31 | Late Blight | Tomato |
| 32 | Leaf Mold | Tomato |
| 33 | Septoria Leaf Spot | Tomato |
| 34 | Spider Mites | Tomato |
| 35 | Target Spot | Tomato |
| 36 | Yellow Leaf Curl Virus | Tomato |
| 37 | Mosaic Virus | Tomato |
| 38 | Healthy | Tomato |

</details>

---

## Lưu ý quan trọng

- **Không push file `.env`** chứa API key lên GitHub
- **Không push file trọng số** `.pt`, `.pth` (thêm vào `.gitignore`)
- Ảnh đầu vào nên chụp **rõ nét**, **đủ sáng**, **thấy rõ lá**
- Kết quả AI mang tính **tham khảo**, cần kiểm chứng với chuyên gia thực tế

---
## Video demo



---

## Tác giả

**Phạm Ngọc Hướng**

---

> *PlantVision AI — Bảo vệ cây trồng bằng trí tuệ nhân tạo* 
