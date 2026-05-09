import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import tempfile
import json

from predict_service import predict_pipeline
from API_Gemini.call_api import (
    build_prompt,
    _stream_gemini_with_fallback,
    _call_gemini_with_fallback,
    get_disease_advice,
    client,
    MODEL_FALLBACK_LIST
)
import time

app = FastAPI(title="PlantVision AI", version="1.0.0")

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory=os.path.dirname(__file__)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Nhận ảnh, chạy pipeline YOLO → SAM → CNN, trả về kết quả JSON.
    """
    # Lưu ảnh tạm thời
    suffix = os.path.splitext(file.filename)[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        t0 = time.time()
        result = predict_pipeline(tmp_path)
        elapsed = round(time.time() - t0, 2)
        result["elapsed_time"] = elapsed
        return result
    finally:
        os.unlink(tmp_path)  # Xóa file tạm sau khi xử lý


@app.post("/chat/stream")
async def chat_stream(
    plant: str = Form(...),
    plant_conf: float = Form(...),
    disease: str = Form(...),
    disease_conf: float = Form(...),
    question: str = Form(...)
):
    """
    Nhận kết quả CNN + câu hỏi người dùng, stream câu trả lời từ Gemini.
    """
    result = {
        "status": "ok",
        "plant": plant,
        "plant_conf": plant_conf,
        "disease": disease,
        "disease_conf": disease_conf,
    }

    prompt = build_prompt(plant, plant_conf, disease, disease_conf, question)

    async def generate():
        for model_name in MODEL_FALLBACK_LIST:
            try:
                chat = client.chats.create(model=model_name)
                # Gửi context trước
                chat.send_message(prompt)
                # Stream câu trả lời
                for chunk in chat.send_message_stream(question):
                    token = chunk.text
                    # Gửi mỗi token dưới dạng Server-Sent Event
                    yield f"data: {json.dumps({'token': token})}\n\n"
                    await asyncio.sleep(0)  # Nhường control để không block

                yield f"data: {json.dumps({'done': True})}\n\n"
                return

            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    yield f"data: {json.dumps({'info': f'Chuyển model...'})}\n\n"
                    await asyncio.sleep(2)
                    continue
                else:
                    yield f"data: {json.dumps({'error': err})}\n\n"
                    return

        yield f"data: {json.dumps({'error': 'Tất cả model đều bận. Vui lòng thử lại sau.'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
