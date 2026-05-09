from google import genai
import os
import time
from dotenv import load_dotenv
from utils.path_helper import get_absolute_path

# Load .env from API_Gemini folder
load_dotenv(get_absolute_path("API_Gemini/.env"))

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Danh sách model fallback: thử lần lượt nếu model trước bị giới hạn quota
# Dùng tên chính xác từ API (models/gemini-... nhưng bỏ tiền tố "models/")
MODEL_FALLBACK_LIST = [
    "gemini-2.0-flash-lite",    # Nhẹ nhất, ít tốn quota nhất
    "gemini-2.5-flash",         # Mạnh hơn, thử nếu lite bị giới hạn
    "gemini-2.0-flash-001",     # Bản cụ thể, ổn định
]

def build_prompt(
    plant_name: str,
    plant_conf: float,
    disease_label: str,
    disease_conf: float,
    user_question: str
) -> str:

    # tách tên bệnh nếu có format Plant___Disease
    if "___" in disease_label:
        _, disease_name = disease_label.split("___")
    else:
        disease_name = disease_label

    return f"""
Bạn là một chuyên gia nông nghiệp hàng đầu.

Hệ thống AI của chúng tôi đã phân tích hình ảnh và đưa ra kết quả sau:
- Cây trồng: {plant_name} (Độ tin cậy: {round(plant_conf*100, 2)}%)
- Tình trạng/Bệnh: {disease_name} (Độ tin cậy: {round(disease_conf*100, 2)}%)

Người dùng đang hỏi: "{user_question}"

Nhiệm vụ của bạn:
1. Xác nhận tình trạng của cây dựa trên kết quả AI.
2. Giải thích ngắn gọn về nguyên nhân của bệnh này.
3. Đưa ra các biện pháp xử lý cụ thể (hóa học, sinh học hoặc kỹ thuật canh tác).
4. Đưa ra lời khuyên để phòng tránh bệnh tái phát.

Hãy trả lời bằng tiếng Việt, giọng văn chuyên nghiệp nhưng dễ hiểu, thân thiện.
"""


def _call_gemini_with_fallback(prompt: str, user_question: str) -> str:
    """
    Thử gọi Gemini lần lượt qua các model trong danh sách fallback.
    Nếu gặp lỗi 429 (quota), tự động chuyển sang model tiếp theo.
    """
    for model_name in MODEL_FALLBACK_LIST:
        try:
            print(f"  [Gemini] Đang thử model: {model_name}...")
            chat = client.chats.create(model=model_name)

            # Gửi context chuẩn đoán trước
            chat.send_message(prompt)

            # Sau đó gửi câu hỏi người dùng
            response = chat.send_message(user_question)
            print(f"  [Gemini] Thành công với model: {model_name}")
            return response.text

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                print(f"  [Gemini] Model {model_name} bị giới hạn quota. Chuyển sang model tiếp theo...")
                time.sleep(2)  # Tạm nghỉ nhẹ trước khi thử model tiếp
                continue
            else:
                # Lỗi khác (network, key sai,...) thì dừng lại
                return f"Lỗi kết nối với chuyên gia AI: {err_str}"

    return "Xin lỗi, tất cả các model AI hiện đang bận. Hạn mức miễn phí trong ngày đã hết. Vui lòng thử lại sau vài phút hoặc ngày mai."


def _stream_gemini_with_fallback(prompt: str, user_question: str) -> str:
    """
    Streaming version: in từng token ngay khi nhận được từ Gemini
    giống như ChatGPT đánh máy theo thời gian thực.
    Trả về toàn bộ văn bản sau khi stream xong.
    """
    for model_name in MODEL_FALLBACK_LIST:
        try:
            print(f"  [Gemini] Đang thử model: {model_name}...\n")
            chat = client.chats.create(model=model_name)

            # Gửi context trước (không stream phần này)
            chat.send_message(prompt)

            # Stream câu trả lời theo từng token
            full_text = ""
            for chunk in chat.send_message_stream(user_question):
                token = chunk.text
                print(token, end="", flush=True)
                full_text += token

            print()  # Xuống dòng sau khi stream xong
            return full_text

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                print(f"\n  [Gemini] Model {model_name} bị giới hạn quota. Chuyển sang model tiếp theo...")
                time.sleep(2)
                continue
            elif "send_message_stream" in err_str or "has no attribute" in err_str:
                # Nếu SDK không hỗ trợ stream, fallback sang batch
                print(f"\n  [Gemini] Streaming không hỗ trợ, chuyển sang chế độ thường...")
                return _call_gemini_with_fallback(prompt, user_question)
            else:
                return f"Lỗi kết nối với chuyên gia AI: {err_str}"

    return "Xin lỗi, tất cả các model AI hiện đang bận. Vui lòng thử lại sau."


def get_disease_advice(result, user_question="Hãy tư vấn cho tôi về tình trạng này", stream=True):
    """
    Lấy lời khuyên từ Gemini AI.
    stream=True: In từng token realtime (mặc định)
    stream=False: Trả về toàn bộ văn bản một lần
    """
    if result["status"] == "no_leaf_detected":
        return "Xin lỗi, tôi không tìm thấy lá cây nào trong ảnh. Bạn vui lòng chụp ảnh rõ nét hơn và gần lá hơn nhé."

    if result["status"] == "error":
        return f"Đã xảy ra lỗi trong quá trình phân tích: {result.get('message', 'Lỗi không xác định')}"

    if result["status"] == "healthy":
        return f"Cây {result['plant']} của bạn trông rất khỏe mạnh! Hãy tiếp tục chăm sóc như hiện tại nhé. Bạn có muốn biết thêm mẹo chăm sóc loại cây này không?"

    plant   = result["plant"]
    disease = result["disease"]

    prompt = build_prompt(
        plant,
        result["plant_conf"],
        disease,
        result["disease_conf"],
        user_question
    )

    if stream:
        return _stream_gemini_with_fallback(prompt, user_question)
    else:
        return _call_gemini_with_fallback(prompt, user_question)