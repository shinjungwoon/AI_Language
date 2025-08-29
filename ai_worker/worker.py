import os
import json
import asyncio
import logging
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import websockets  # pip install websockets

# ---- TensorFlow Lite Interpreter (tensorflow or tflite-runtime 중 하나만 사용) ----
try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except Exception:
    from tflite_runtime.interpreter import Interpreter  # type: ignore

# === 환경변수 ===
WS_URL   = os.getenv("WS_URL", "ws://ai:8001/ai")   # compose 기준: 서비스명 ai, 포트 8001
AI_TOKEN = os.getenv("AI_TOKEN", "changeme")
ROLE     = os.getenv("ROLE", "ai")
ROOM     = os.getenv("ROOM", "")

# 모델 경로
DEFAULT_TFLITE = os.getenv(
    "TFLITE_PATH",
    "/AI_Language/models/multi_hand_gesture_classifier.tflite"
)

# (선택) 라벨
LABELS = os.getenv("LABELS", "").split(",") if os.getenv("LABELS") else None


class GestureClassifier:
    """
    간단한 TFLite 래퍼.
    입력: 한 손의 21개 랜드마크 (x,y,z) → (1, 63) float32
    전처리:
      - 포인트가 부족하면 0 패딩, 초과하면 21개로 자름
      - 0번(wrist)을 원점으로 평행이동
      - bbox 대각선 길이로 스케일 정규화 (0으로 나눔 방지)
    """
    def __init__(self, tflite_path: str):
        self.interp = Interpreter(model_path=tflite_path)
        self.interp.allocate_tensors()
        self.input_details = self.interp.get_input_details()
        self.output_details = self.interp.get_output_details()
        in0 = self.input_details[0]["shape"]
        out0 = self.output_details[0]["shape"]
        logging.info(f"[worker] TFLite input shape: {in0}, output shape: {out0}")

    @staticmethod
    def _fix_length_21(points: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """길이 21에 맞게 자르거나 0으로 패딩."""
        pts = list(points[:21])
        while len(pts) < 21:
            pts.append({"x": 0.0, "y": 0.0, "z": 0.0})
        return pts

    def _preprocess(self, points: List[Dict[str, float]]) -> np.ndarray:
        """
        points: [{"x":..., "y":..., "z":...}, ...] (21개 예상)
        반환: (1, 63) float32
        """
        if not points:
            raise ValueError("Empty landmarks")

        pts = self._fix_length_21(points)

        # 1) 원점 이동: 0번(wrist)을 기준으로 평행이동
        wx, wy, wz = pts[0]["x"], pts[0]["y"], pts[0]["z"]
        xs = np.array([p["x"] - wx for p in pts], dtype=np.float32)
        ys = np.array([p["y"] - wy for p in pts], dtype=np.float32)
        zs = np.array([p["z"] - wz for p in pts], dtype=np.float32)

        # 2) 크기 정규화: bbox 대각선 길이로 나눔
        min_x, max_x = float(xs.min()), float(xs.max())
        min_y, max_y = float(ys.min()), float(ys.max())
        min_z, max_z = float(zs.min()), float(zs.max())
        dx, dy, dz = (max_x - min_x), (max_y - min_y), (max_z - min_z)
        diag = (dx * dx + dy * dy + dz * dz) ** 0.5
        scale = diag if diag > 1e-6 else 1.0

        xs /= scale
        ys /= scale
        zs /= scale

        arr = np.empty(63, dtype=np.float32)
        arr[0::3] = xs
        arr[1::3] = ys
        arr[2::3] = zs

        return arr.reshape(1, -1)

    def infer(self, points: List[Dict[str, float]]) -> Dict[str, Any]:
        x = self._preprocess(points)

        input_index = self.input_details[0]["index"]
        input_shape = self.input_details[0]["shape"]

        if int(np.prod(input_shape)) == x.size:
            x_reshaped = x.reshape(input_shape).astype(np.float32)
        else:
            # 필요 시 모양 맞춤 추가
            x_reshaped = x.astype(np.float32)

        self.interp.set_tensor(input_index, x_reshaped)
        self.interp.invoke()

        output_index = self.output_details[0]["index"]
        y = self.interp.get_tensor(output_index)  # (1, C)
        scores = y[0].tolist()
        top_idx = int(np.argmax(scores))
        conf = float(scores[top_idx])

        if LABELS and 0 <= top_idx < len(LABELS):
            label = LABELS[top_idx]
        else:
            label = f"class_{top_idx}"

        return {"label": label, "score": conf, "index": top_idx}


async def run_worker():
    logging.basicConfig(level=logging.INFO)
    # 모델 로드
    try:
        clf = GestureClassifier(DEFAULT_TFLITE)
        logging.info("[worker] Model loaded.")
    except Exception:
        logging.error("[worker] Model load failed:\n%s", traceback.format_exc())
        raise

    # 쿼리스트링 구성
    qs = []
    if AI_TOKEN:
        qs.append(f"token={AI_TOKEN}")
    if ROLE:
        qs.append(f"role={ROLE}")
    if ROOM:
        qs.append(f"room={ROOM}")
    url = WS_URL + (("?" + "&".join(qs)) if qs else "")

    backoff = 1
    while True:
        try:
            logging.info(f"[worker] connecting to {url}")
            async with websockets.connect(
                url,
                max_size=10 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=20,
            ) as ws:
                logging.info("[worker] connected.")
                backoff = 1

                while True:
                    msg = await ws.recv()  # str(JSON) or bytes
                    if not isinstance(msg, str):
                        # (옵션) 프레임 바이트 수신 모드: 현재는 미사용
                        continue

                    # JSON 메시지 파싱
                    try:
                        data = json.loads(msg)
                    except Exception:
                        continue

                    mtype = data.get("type")
                    if mtype != "hand_landmarks":
                        # 다른 타입은 무시
                        continue

                    # === 스키마에 맞춰 처리 ===
                    # landmarks: [hand1(21점), hand2(21점)]
                    hands: List[List[Dict[str, float]]] = data.get("landmarks") or []
                    # 단일손 모델 가정: 첫 번째 손만 사용
                    main_hand: List[Dict[str, float]] = hands[0] if hands else []

                    if not main_hand:
                        # 유효하지 않으면 스킵
                        continue

                    try:
                        result = clf.infer(main_hand)
                        out = {
                            "type": "ai_result",
                            "text": result.get("label", "UNK"),
                            "score": result.get("score"),
                            "frame_id": data.get("frame_id"),
                            "room_id": data.get("room_id"),
                        }
                        await ws.send(json.dumps(out))
                    except Exception:
                        logging.error("[worker] infer error:\n%s", traceback.format_exc())

        except (websockets.ConnectionClosed, ConnectionRefusedError) as e:
            logging.warning(f"[worker] disconnected: {e}. retry in {backoff}s")
        except Exception:
            logging.error("[worker] error:\n%s", traceback.format_exc())

        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 10)


if __name__ == "__main__":
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        pass