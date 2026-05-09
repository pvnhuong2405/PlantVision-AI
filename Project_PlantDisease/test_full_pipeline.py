from predict_service import predict_pipeline
from API_Gemini.call_api import get_disease_advice

while True:

    image_path = input("\nNhập đường dẫn ảnh (q để thoát): ")

    if image_path.lower() == "q":
        break

    result = predict_pipeline(image_path)

    print("\n--- CNN RESULT ---")
    print(result)

    # advice = get_disease_advice(result)

    # print("\n--- GEMINI ADVICE ---")
    # print(advice)

    # if result["status"] != "ok":
    #     print("Ảnh này không phải là bệnh")
    #     continue
    if result["status"] == "healthy":
        print("\n--- THÔNG BÁO ---")
        print("Ảnh bạn gửi không phát hiện bệnh. Cây có vẻ khỏe mạnh.")
        print("Bạn có cần mình tư vấn cách chăm sóc hoặc phòng bệnh không?")

    elif result["status"] != "ok":
        print("\nModel chưa đủ tự tin để chẩn đoán từ ảnh này.")
        print("Bạn vẫn có thể hỏi thêm để mình tư vấn chung về cây.")

    while True:
        user_question = input("\nBạn muốn hỏi thêm gì? (enter để đổi ảnh): ")

        if user_question == "":
            break

        advice = get_disease_advice(result, user_question)

        print("\n--- CHATBOT ---")
        print(advice)