import os
import whisper

model = whisper.load_model("base")  # 또는 "small", "medium", "large"


def transcribe_audio(audio_path: str, session_id: str) -> str:
    print(f"🔍 Whisper 전사 시작: {audio_path}")

    result = model.transcribe(audio_path,language="en")
    transcript = result["text"]

    # transcripts/<session_id>/<session_id>_original.txt 저장
    save_transcript(session_id, transcript)

    return transcript


def save_transcript(session_id: str, transcript: str):
    folder = os.path.join("transcripts", session_id)
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, f"{session_id}_original.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(transcript)

    print(f"✅ Whisper 전사 결과 저장됨: {path}")

