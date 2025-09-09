import os
import sys
import asyncio
import json
from collections import deque
import base64
import shutil

import cv2
import numpy as np
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image

# --- 프로젝트 모듈 ---
import modules.holistic_module as hm
from modules.utils import Vector_Normalization  # (로컬 추론 폴백 시 사용 가능)

# =========================
# 환경 변수 / 설정
# =========================
WS_HOST = os.getenv("WS_HOST", "localhost")    # ai 서버 호스트/IP
WS_PORT = int(os.getenv("WS_PORT", "8001"))
WS_ROOM = os.getenv("WS_ROOM", "debug")
WS_ROLE = os.getenv("WS_ROLE", "client")       # 이 스크립트는 client
AI_TOKEN = os.getenv("AI_TOKEN", "")           # 서버가 토큰 검증하면 넣기
FONT_PATH = os.getenv("FONT_PATH", "fonts/HMKMMAG.TTF")

SEQ_LEN = int(os.getenv("SEQ_LEN", "10"))
MIN_CONF = float(os.getenv("MIN_CONFIDENCE", "0.30"))
ALWAYS_EMIT = os.getenv("ALWAYS_EMIT_CAPTION", "") == "1"

# 모델 저장 관련 설정
MODEL_DST = os.getenv("TFLITE_PATH", "models/multi_hand_gesture_classifier.tflite")
TFLITE_SRC = os.getenv("TFLITE_SRC", "")      # 기존 파일 경로가 있으면 여기서 복사
TFLITE_B64 = os.getenv("TFLITE_B64", "")      # base64로 모델을 넘길 경우
AI_LANGUAGE_DIR = os.getenv("AI_LANGUAGE_DIR", "/fastapp/AI_Language")

# ws://HOST:PORT/ai?role=client&room=debug(&token=...)
_qs = [f"role={WS_ROLE}", f"room={WS_ROOM}"]
if AI_TOKEN:
    _qs.append(f"token={AI_TOKEN}")
WS_URL = f"ws://{WS_HOST}:{WS_PORT}/ai?{'&'.join(_qs)}"

# =========================
# 모델 저장 유틸 (주석 유지하되 파일은 준비)
# =========================
def ensure_model_saved():
    """
    로컬 추론 코드는 주석 상태이지만,
    나중에 주석 해제 시 바로 사용할 수 있도록 모델 파일을 저장/준비한다.
    우선순위:
      1) TFLITE_SRC 경로에서 복사
      2) TFLITE_B64 디코드 저장
      3) AI_LANGUAGE_DIR/models/multi_hand_gesture_classifier.tflite 에서 복사
    """
    dst = MODEL_DST
    dst_dir = os.path.dirname(dst) or "."
    os.makedirs(dst_dir, exist_ok=True)

    if os.path.isfile(dst):
        print(f"[MODEL] already exists: {dst}")
        return True

    # 1) TFLITE_SRC 복사
    if TFLITE_SRC and os.path.isfile(TFLITE_SRC):
        try:
            shutil.copy2(TFLITE_SRC, dst)
            print(f"[MODEL] copied from TFLITE_SRC → {dst}")
            return True
        except Exception as e:
            print(f"[WARN] copy from TFLITE_SRC failed: {e}")

    # 2) TFLITE_B64 디코드 저장
    if TFLITE_B64:
        try:
            blob = base64.b64decode(TFLITE_B64)
            with open(dst, "wb") as f:
                f.write(blob)
            print(f"[MODEL] written from TFLITE_B64 → {dst} (size={len(blob)} bytes)")
            return True
        except Exception as e:
            print(f"[WARN] write from TFLITE_B64 failed: {e}")

    # 3) AI_LANGUAGE_DIR 기본 경로에서 복사
    default_src = os.path.join(AI_LANGUAGE_DIR, "models", "multi_hand_gesture_classifier.tflite")
    if os.path.isfile(default_src):
        try:
            shutil.copy2(default_src, dst)
            print(f"[MODEL] copied from AI_LANGUAGE_DIR → {dst}")
            return True
        except Exception as e:
            print(f"[WARN] copy from AI_LANGUAGE_DIR failed: {e}")

    print(f"[WARN] model not prepared. Set TFLITE_SRC or TFLITE_B64 or place file at: {dst}")
    return False

# 실행 시 한 번 모델 파일 준비 시도
ensure_model_saved()

# =========================
# 폰트
# =========================
try:
    font = ImageFont.truetype(FONT_PATH, 40)
except Exception:
    font = ImageFont.load_default()

# =========================
# MediaPipe Detector
# =========================
detector = hm.HolisticDetector(min_detection_confidence=0.3)

def right_hand_landmarks21(img):
    """
    return (points21, vis_img)
    points21: list[dict{x:float,y:float}] 길이 21 (없으면 None)
    """
    img = detector.findHolistic(img, draw=True)
    _, right = detector.findRighthandLandmark(img)
    if right is None:
        return None, img

    pts = [{"x": float(lm.x), "y": float(lm.y)} for lm in right.landmark]
    if len(pts) < 21:
        pts += [{"x": 0.0, "y": 0.0}] * (21 - len(pts))
    return pts[:21], img

# =========================
# (선택) 로컬 TFLite 폴백 관련 유틸
#  - 기본 경로는 서버 추론 사용.
#  - 원하면 주석 해제하고 로컬 추론도 활성화 가능.
# =========================
# import tensorflow as tf
# interpreter = tf.lite.Interpreter(model_path=MODEL_DST)
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# actions = [
#     'ㄱ','ㄴ','ㄷ','ㄹ','ㅁ','ㅂ','ㅅ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ',
#     'ㅏ','ㅑ','ㅓ','ㅕ','ㅗ','ㅛ','ㅜ','ㅠ','ㅡ','ㅣ','ㅐ','ㅒ','ㅔ','ㅖ','ㅢ','ㅚ','ㅟ'
# ]
# def predict_local(window_1x10x55):
#     interpreter.set_tensor(input_details[0]['index'], window_1x10x55.astype(np.float32, copy=False))
#     interpreter.invoke()
#     y = interpreter.get_tensor(output_details[0]['index'])[0]
#     e = np.exp(y - np.max(y))
#     probs = e / np.sum(e) if np.sum(e) > 0 else np.zeros_like(e, dtype=np.float32)
#     i = int(np.argmax(probs))
#     return actions[i], float(probs[i])

# =========================
# 웹소켓 클라이언트
# =========================
import websockets  # pip install websockets

async def main():
    print(f"[INFO] connect → {WS_URL}")
    async with websockets.connect(WS_URL, max_size=2**23) as ws:
        print("[INFO] connected")

        # 최신 캡션 공유 객체
        latest = {"text": "", "confidence": 0.0}
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] cannot open camera(0)")
            return

        seq = deque(maxlen=SEQ_LEN)

        async def recv_loop():
            # 서버에서 caption 수신
            while True:
                try:
                    msg = await ws.recv()
                except websockets.ConnectionClosed:
                    break
                except Exception:
                    continue
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                if data.get("type") == "caption":
                    latest["text"] = str(data.get("text") or "")
                    latest["confidence"] = float(data.get("confidence") or 0.0)

        recv_task = asyncio.create_task(recv_loop())

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                pts21, vis_img = right_hand_landmarks21(frame)

                # 시퀀스 버퍼 채우기
                if pts21 is None:
                    seq.append([{"x": 0.0, "y": 0.0} for _ in range(21)])
                else:
                    seq.append(pts21)

                # 10프레임이 모이면 서버로 전송
                if len(seq) == SEQ_LEN:
                    payload = {
                        "type": "hand_landmarks_sequence",
                        "frame_sequence": list(seq),  # [[{x,y}×21] ×10]
                        # "room_id": WS_ROOM  # 쿼리스트링으로 지정했으니 생략 가능
                    }
                    try:
                        await ws.send(json.dumps(payload, ensure_ascii=False))
                    except websockets.ConnectionClosed:
                        print("[ERROR] websocket closed while sending")
                        break

                # 캡션 오버레이
                text = latest["text"]
                conf = latest["confidence"]
                pil = Image.fromarray(vis_img)
                draw = ImageDraw.Draw(pil)

                if len(seq) < SEQ_LEN:
                    draw.text((10, 30), f"buffer: {len(seq)}/{SEQ_LEN}", font=font, fill=(255,255,255))
                elif text:
                    draw.text((10, 30), f"{text} ({conf:.2f})", font=font, fill=(255,255,255))
                else:
                    draw.text((10, 30), "…", font=font, fill=(255,255,255))

                vis_img = np.array(pil)
                cv2.imshow("Sign (WS client)", vis_img)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            recv_task.cancel()
            try:
                await recv_task
            except Exception:
                pass
            print("[INFO] closed")

if __name__ == "__main__":
    asyncio.run(main())
