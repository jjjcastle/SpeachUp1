import cv2
import time
import threading
import os
from fastapi import FastAPI, Response, Form, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from uuid import uuid4
from vision_model import run_vision_model
from audio_recoder import start_audio_recording, stop_audio_recording
from Whisper_model import transcribe_audio,save_transcript
app = FastAPI()
FRAME_SAVE_INTERVAL = 1.0 / 6.0  # 초당 6프레임
output_folder = "captured_frames"
os.makedirs(output_folder, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

# 각 세션마다 프레임을 저장하는 쓰레드 함수
def start_camera_capture(session_id):
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print(f"❌ [{session_id}] 카메라 열기 실패")
        return

    session_folder = os.path.join(output_folder, session_id)
    os.makedirs(session_folder, exist_ok=True)

    frame_counter = 1
    last_save_time = 0.0

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        current_time = time.time()
        if current_time - last_save_time >= FRAME_SAVE_INTERVAL:
            filename = os.path.join(session_folder, f"{session_id}_{frame_counter}.jpg")
            cv2.imwrite(filename, frame)
            frame_counter += 1
            last_save_time = current_time

        time.sleep(0.01)

    camera.release()
    print(f"✅ [{session_id}] 카메라 캡처 종료")

# 영상 스트리밍용
def video_stream_generator():
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(video_stream_generator(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

# @app.post("/start-video")
# def start_video(session_id: str = Form(None)):
#     if session_id is None:
#         session_id = str(uuid4())[:8]
#
#     # 세션별 프레임 저장 스레드 시작
#     capture_thread = threading.Thread(target=start_camera_capture, args=(session_id,), daemon=True)
#     capture_thread.start()
#
#     # 오디오 녹음 시작
#     start_audio_recording()
#     return {"message": "Video streaming started", "session_id": session_id}
#
# @app.post("/stop-video")
# def stop_video(session_id: str = Form(None)):
#     # 오디오 녹음 정지 (세션 ID 필요)
#     if session_id:
#         stop_audio_recording(session_id)
#     return {"message": "Video streaming stopped"}
@app.post("/start-video")
def start_video(session_id: str = Form(None)):
    if session_id is None:
        session_id = str(uuid4())[:8]

    threading.Thread(target=start_camera_capture, args=(session_id,), daemon=True).start()
    threading.Thread(target=start_audio_recording, args=(session_id,), daemon=True).start()

    return {"message": "Video & Audio streaming started", "session_id": session_id}

@app.post("/stop-video")
def stop_video(session_id: str = Form(None)):
    if session_id:
        stop_audio_recording(session_id)
    return {"message": "Video streaming stopped"}

# 이미지 → mp4 영상 변환 함수
def images_to_video(session_folder, output_path, fps=6):
    images = sorted([img for img in os.listdir(session_folder) if img.endswith(".jpg")])
    if not images:
        raise ValueError("No images found in the folder.")

    first_image_path = os.path.join(session_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for image in images:
        img_path = os.path.join(session_folder, image)
        frame = cv2.imread(img_path)
        out.write(frame)
    out.release()

@app.post("/process-video")
async def process_video(request: Request):
    data = await request.form()
    session_id = data.get("session_id")
    if not session_id:
        return {"error": "세션 ID가 없습니다."}

    session_folder = os.path.join(output_folder, session_id)
    mp4_path = os.path.join(session_folder, f"{session_id}.mp4")
    audio_path = os.path.join("audio", session_id, f"{session_id}.wav")
    try:
        images_to_video(session_folder, mp4_path)
    except Exception as e:
        return {"error": f"영상 생성 실패: {str(e)}"}

    result = run_vision_model(mp4_path)
    transcript = transcribe_audio(audio_path, session_id)
    # 나중에 llm 모델의 입력으로 transcript 전달
    #ted = ted_transcript(transcript) 아래에도 추가
    return {
        "session_id": session_id,
        "result": result,
        "transcript": transcript
    }

