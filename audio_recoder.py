# audio_recoder.py
import os
import threading
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

audio_recording_thread = None
is_recording_audio = False
audio_buffer = []
fs = 44100
current_session_id = None  # 세션 ID 저장

def _record_audio_loop():
    global audio_buffer, is_recording_audio
    audio_buffer = []
    print("🎙️ 오디오 녹음 시작")

    try:
        with sd.InputStream(samplerate=fs, channels=1, dtype='int16') as stream:
            while is_recording_audio:
                try:
                    frame, _ = stream.read(1024)
                    if frame is not None:
                        audio_buffer.append(frame.copy())
                        print("🎧 프레임 저장 중")
                except Exception as e:
                    print(f"[오디오] read 오류: {e}")
                    break
    except Exception as e:
        print(f"[오디오] InputStream 오류: {e}")

    print("🛑 오디오 녹음 종료 준비")

def start_audio_recording(session_id):
    global is_recording_audio, audio_recording_thread, current_session_id
    current_session_id = session_id
    is_recording_audio = True
    audio_recording_thread = threading.Thread(target=_record_audio_loop)
    audio_recording_thread.start()

def stop_audio_recording(session_id):
    global is_recording_audio, audio_buffer
    is_recording_audio = False
    if audio_recording_thread is not None:
        audio_recording_thread.join(timeout=5)

    if not audio_buffer:
        print(f"⚠️ {session_id} 오디오 버퍼가 비어있습니다. 저장하지 않음.")
        return

    try:
        audio = np.concatenate(audio_buffer, axis=0)
        folder = os.path.join("audio", session_id)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{session_id}.wav")
        write(path, fs, audio)
        print(f"✅ 오디오 저장 완료: {path}")
        return path
    except Exception as e:
        print(f"❌ 오디오 저장 중 오류 발생: {e}")
