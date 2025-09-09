import os
import asyncio
import json
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from PIL import ImageFont, ImageDraw, Image

# --- 사용자 모듈(그대로 사용) ---
import modules.holistic_module as hm  # HolisticDetector(min_detection_confidence=0.3)

# =========================
# 환경 변수(필요시 수정)
# =========================
WS_HOST = os.getenv("WS_HOST", "localhost")    # 또는 ai 서버 IP/호스트
WS_PORT = int(os.getenv("WS_PORT", "8001"))
WS_ROOM = os.getenv("WS_ROOM", "debug")
WS_ROLE = os.getenv("WS_ROLE", "client")       # 이 스크립트는 client 역할
WS_TOKEN = os.getenv("AI_TOKEN", "")           # 토큰 검증이 있다면 사용
FONT_PATH = os.getenv("FONT_PATH", "fonts/HMKMMAG.TTF")

# ws://HOST:PORT/ai?role=client&room=debug(&token=...)
_qs = [f"role={WS_ROLE}", f"room={WS_ROOM}"]
if WS_TOKEN:
    _qs.append(f"token={WS_TOKEN}")
WS_URL = f"ws://{WS_HOST}:{WS_PORT}/ai?{'&'.join(_qs)}"

# =========================
# 렌더링용 폰트
# =========================
try:
    font = ImageFont.truetype(FONT_PATH, 40)
except Exception:
    font = ImageFont.load_default()

# =========================
# MediaPipe (오른손만 사용)
# =========================
detector = hm.HolisticDetector(min_detection_confidence=0.3)

# 프레임 → 21개 (x,y) 리스트로 변환
def landmark21_from_right_hand(img):
    """
    return: list[dict{x:float,y:float}] 길이 21
            손이 없으면 None
    """
    img = detector.findHolistic(img, draw=True)
    _, right = detector.findRighthandLandmark(img)
    if right is None:
        return None, img

    pts = []
    for lm in right.landmark:
        pts.append({"x": float(lm.x), "y": float(lm.y)})  # (z는 서버에서 사용 안함)
    # 부족할 경우 패딩
    if len(pts) < 21:
        pts += [{"x": 0.0, "y": 0.0}] * (21 - len(pts))
    return pts[:21], img

# =========================
# 웹소켓 송수신
# =========================
import websockets  # pip install websockets

async def run():
    print(f"[INFO] connect → {WS_URL}")
    async with websockets.connect(WS_URL) as ws:
        print("[INFO] connected")

        # 수신 루프: 최신 캡션을 보관
        latest_caption = {"text": "", "confidence": 0.0}
        caption_lock = asyncio.Lock()
        stop_flag = {"stop": False}

        async def receiver():
            try:
                while not stop_flag["stop"]:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
                    except websockets.ConnectionClosed:
                        break

                    try:
                        data = json.loads(msg)
                    except Exception:
                        continue

                    if data.get("type") == "caption":
                        async with caption_lock:
                            latest_caption["text"] = str(data.get("text") or "")
                            latest_caption["confidence"] = float(data.get("confidence") or 0.0)
            finally:
                stop_flag["stop"] = True

        recv_task = asyncio.create_task(receiver())

        # 비디오 캡처 & 송신 루프
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] cannot open camera(0)")
            stop_flag["stop"] = True
            await recv_task
            return

        # 10프레임 시퀀스 버퍼
        seq = deque(maxlen=10)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 손 좌표 추출
                pts21, vis_img = landmark21_from_right_hand(frame)

                # 10프레임 시퀀스 채우기
                if pts21 is None:
                    # 손이 없으면 마지막 프레임 또는 제로 프레임 사용
                    seq.append([{"x": 0.0, "y": 0.0} for _ in range(21)])
                else:
                    seq.append(pts21)

                # 10프레임이 모이면 서버에 시퀀스 전송
                if len(seq) == 10:
                    payload = {
                        "type": "hand_landmarks_sequence",
                        "frame_sequence": list(seq),   # [[{x,y}x21] x10]
                        # "room_id": WS_ROOM  # room 쿼리로 이미 묶이므로 생략 가능
                    }
                    try:
                        await ws.send(json.dumps(payload, ensure_ascii=False))
                    except websockets.ConnectionClosed:
                        print("[ERROR] websocket closed while sending")
                        break

                # 최신 캡션을 화면 좌상단에 오버레이
                async with caption_lock:
                    text = latest_caption["text"]
                    conf = latest_caption["confidence"]
                if text:
                    pil = Image.fromarray(vis_img)
                    draw = ImageDraw.Draw(pil)
                    draw.text((10, 30), f"{text} ({conf:.2f})", font=font, fill=(255, 255, 255))
                    vis_img = np.array(pil)

                cv2.imshow("Sign Client (WS)", vis_img)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

        finally:
            stop_flag["stop"] = True
            cap.release()
            cv2.destroyAllWindows()
            await recv_task
            print("[INFO] closed")

if __name__ == "__main__":
    asyncio.run(run())
