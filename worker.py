# import os
# import json
# import asyncio
# import logging
# import traceback
# from typing import Any, Dict, List

# import numpy as np
# import websockets  

# # ---- 로깅 설정 ----
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s %(name)s - %(message)s"
# )
# logger = logging.getLogger(__name__)

# # ---- TensorFlow Lite Interpreter (tensorflow or tflite-runtime 중 하나만 사용) ----
# try:
#     import tensorflow as tf
#     Interpreter = tf.lite.Interpreter
# except Exception:
#     from tflite_runtime.interpreter import Interpreter  # type: ignore

# # === 환경변수 ===
# WS_URL   = os.getenv("WS_URL", "ws://ai:8001/ai")   # compose 기준: 서비스명 ai, 포트 8001
# AI_TOKEN = os.getenv("AI_TOKEN", "changeme")
# ROLE     = os.getenv("ROLE", "ai")
# ROOM     = os.getenv("ROOM", "")


# ws_url = f"{WS_URL}?role={ROLE}&room={ROOM}"

# # 모델 경로
# DEFAULT_TFLITE = os.getenv(
#     "TFLITE_PATH",
#     "/ai/AI_Language/models/multi_hand_gesture_classifier.tflite"
# )

# # (선택) 라벨
# LABELS = os.getenv("LABELS", "").split(",") if os.getenv("LABELS") else None


# class GestureClassifier:
#     """
#     간단한 TFLite 래퍼.
#     입력: 한 손의 21개 랜드마크 (x,y,z) → (1, 63) float32
#     전처리:
#       - 포인트가 부족하면 0 패딩, 초과하면 21개로 자름
#       - 0번(wrist)을 원점으로 평행이동
#       - bbox 대각선 길이로 스케일 정규화 (0으로 나눔 방지)
#     """
#     def __init__(self, tflite_path: str):
#         self.interp = Interpreter(model_path=tflite_path)
#         self.interp.allocate_tensors()
#         self.input_details = self.interp.get_input_details()
#         self.output_details = self.interp.get_output_details()
#         in0 = self.input_details[0]["shape"]
#         out0 = self.output_details[0]["shape"]
#         logger.info(f"[worker] TFLite input shape: {in0}, output shape: {out0}")  # ★

#     @staticmethod
#     def _fix_length_21(points: List[Dict[str, float]]) -> List[Dict[str, float]]:
#         """길이 21에 맞게 자르거나 0으로 패딩."""
#         pts = list(points[:21])
#         while len(pts) < 21:
#             pts.append({"x": 0.0, "y": 0.0, "z": 0.0})
#         return pts

#     def _preprocess(self, points: List[Dict[str, float]]) -> np.ndarray:
#         """
#         points: [{"x":..., "y":..., "z":...}, ...] (21개 예상)
#         반환: (1, 63) float32
#         """
#         if not points:
#             raise ValueError("Empty landmarks")

#         pts = self._fix_length_21(points)

#         # 1) 원점 이동: 0번(wrist)을 기준으로 평행이동
#         wx, wy, wz = pts[0]["x"], pts[0]["y"], pts[0]["z"]
#         xs = np.array([p["x"] - wx for p in pts], dtype=np.float32)
#         ys = np.array([p["y"] - wy for p in pts], dtype=np.float32)
#         zs = np.array([p["z"] - wz for p in pts], dtype=np.float32)

#         # 2) 크기 정규화: bbox 대각선 길이로 나눔
#         min_x, max_x = float(xs.min()), float(xs.max())
#         min_y, max_y = float(ys.min()), float(ys.max())
#         min_z, max_z = float(zs.min()), float(zs.max())
#         dx, dy, dz = (max_x - min_x), (max_y - min_y), (max_z - min_z)
#         diag = (dx * dx + dy * dy + dz * dz) ** 0.5
#         scale = diag if diag > 1e-6 else 1.0

#         xs /= scale
#         ys /= scale
#         zs /= scale

#         arr = np.empty(63, dtype=np.float32)
#         arr[0::3] = xs
#         arr[1::3] = ys
#         arr[2::3] = zs

#         return arr.reshape(1, -1)

#     def infer(self, points: List[Dict[str, float]]) -> Dict[str, Any]:
#         x = self._preprocess(points)

#         input_index = self.input_details[0]["index"]
#         input_shape = self.input_details[0]["shape"]

#         if int(np.prod(input_shape)) == x.size:
#             x_reshaped = x.reshape(input_shape).astype(np.float32)
#         else:
#             # 필요 시 모양 맞춤 추가
#             x_reshaped = x.astype(np.float32)

#         self.interp.set_tensor(input_index, x_reshaped)
#         self.interp.invoke()

#         output_index = self.output_details[0]["index"]
#         y = self.interp.get_tensor(output_index)  # (1, C)
#         scores = y[0].tolist()
#         top_idx = int(np.argmax(scores))
#         conf = float(scores[top_idx])

#         if LABELS and 0 <= top_idx < len(LABELS):
#             label = LABELS[top_idx]
#         else:
#             label = f"class_{top_idx}"

#         return {"label": label, "score": conf, "index": top_idx}


# async def run_worker():
#     try:
#         logger.info("모델 로드 시작") # ★
#         clf = GestureClassifier(DEFAULT_TFLITE)
#         logger.info("모델 로드 완료")# ★
#     except Exception:
#         logger.error("모델 로드 실패:\n%s", traceback.format_exc())
#         raise

#     # 쿼리스트링 구성
#     qs = []
#     if AI_TOKEN:
#         qs.append(f"token={AI_TOKEN}")
#     if ROLE:
#         qs.append(f"role={ROLE}")

#     url = WS_URL + (("?" + "&".join(qs)) if qs else "")

#     backoff = 1
#     while True:
#         try:
#             logger.info(f"서버 접속 시도 {url}")  # ★
#             async with websockets.connect(
#                 url,
#                 max_size=10 * 1024 * 1024,
#                 ping_interval=20,
#                 ping_timeout=20,
#             ) as ws:
#                 logger.info("서버 접속 완료")  # ★ # 영통 전에 되는거. 됨 
#                 backoff = 1

#                 while True:
#                     msg = await ws.recv()  # str(JSON) or bytes
#                     if not isinstance(msg, str):
#                         continue

#                     # JSON 메시지 파싱
#                     try:
#                         data = json.loads(msg)
#                     except Exception:
#                         logger.exception("JSON 파싱 실패")
#                         continue
                        

#                     mtype = data.get("type")
#                     if mtype != "hand_landmarks":
#                         # 다른 타입은 무시
#                         continue

#                     # === 스키마에 맞춰 처리 ===
#                     # landmarks: [hand1(21점), hand2(21점)]
#                     hands: List[List[Dict[str, float]]] = data.get("landmarks") or []
#                     # 단일손 모델 가정: 첫 번째 손만 사용
#                     main_hand: List[Dict[str, float]] = hands[0] if hands else []

#                     if not main_hand:
#                         # 유효하지 않으면 스킵
#                         continue

#                     try:
#                         result = clf.infer(main_hand)
#                         out = {
#                             "type": "ai_result",
#                             "text": result.get("label", "UNK"),
#                             "score": result.get("score"),
#                             "frame_id": data.get("frame_id"),
#                             "room_id": data.get("room_id"),
#                         }
#                         await ws.send(json.dumps(out))
#                         logger.info(f"번역 완료 → {out['text']} (score={out['score']:.2f})")
#                     except Exception:
#                         logger.error("번역 실패:\n%s", traceback.format_exc())

#         except (websockets.ConnectionClosed, ConnectionRefusedError) as e:
#             logger.warning(f"서버 연결 끊김: {e}. {backoff}초 후 재시도")
#         except Exception:
#             logger.error("알 수 없는 오류:\n%s", traceback.format_exc())

#         await asyncio.sleep(backoff)
#         backoff = min(backoff * 2, 10)


# if __name__ == "__main__":
#     try:
#         asyncio.run(run_worker())
#     except KeyboardInterrupt:
#         logger.info("worker.py 종료")



# import os
# import json
# import asyncio
# import logging
# import traceback
# from typing import Any, Dict, List, Tuple
# import time
# import uuid
# import numpy as np
# import websockets  # pip install websockets

# # ---------- 로깅 ----------
# logging.basicConfig(
#     level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
#     format="%(asctime)s %(levelname)s %(name)s - %(message)s",
#     force=True,
# )
# logger = logging.getLogger("ai_worker")

# # ---------- TFLite Interpreter ----------
# try:
#     import tensorflow as tf
#     Interpreter = tf.lite.Interpreter
# except Exception:
#     from tflite_runtime.interpreter import Interpreter  # type: ignore

# # ---------- 환경변수 ----------
# WS_URL   = os.getenv("WS_URL", "ws://ai:8001/ai")         # 내부 도커 네트워크 기준
# AI_TOKEN = os.getenv("AI_TOKEN", "change-me-dev")
# ROLE     = os.getenv("ROLE", "ai")
# ROOM     = os.getenv("ROOM", "")

# qs = []
# if AI_TOKEN:
#     qs.append(f"token={AI_TOKEN}")
# if ROLE:
#     qs.append(f"role={ROLE}")
# if ROOM:
#     qs.append(f"room={ROOM}")
# URL = WS_URL + (("?" + "&".join(qs)) if qs else "")

# # 모델/라벨
# DEFAULT_TFLITE = os.getenv(
#     "TFLITE_PATH",
#     "/ai/AI_Language/models/multi_hand_gesture_classifier.tflite"
# )
# LABELS = os.getenv("LABELS", "")
# LABELS = LABELS.split(",") if LABELS else None


# class GestureClassifier:
#     """TFLite 추론 래퍼: 21개 랜드마크(x,y,z) -> (1,63)"""
#     def __init__(self, tflite_path: str):
#         self.interp = Interpreter(model_path=tflite_path)
#         self.interp.allocate_tensors()
#         self.input_details = self.interp.get_input_details()
#         self.output_details = self.interp.get_output_details()
#         in0 = self.input_details[0]["shape"]
#         out0 = self.output_details[0]["shape"]
#         logger.info("[worker] TFLite input=%s output=%s", in0, out0)

#     @staticmethod
#     def _fix_length_21(points: List[Dict[str, float]]) -> List[Dict[str, float]]:
#         pts = list(points[:21])
#         while len(pts) < 21:
#             pts.append({"x": 0.0, "y": 0.0, "z": 0.0})
#         return pts

#     def _preprocess(self, points: List[Dict[str, float]]) -> np.ndarray:
#         if not points:
#             raise ValueError("Empty landmarks")
#         pts = self._fix_length_21(points)

#         wx, wy, wz = pts[0]["x"], pts[0]["y"], pts[0]["z"]
#         xs = np.array([p["x"] - wx for p in pts], dtype=np.float32)
#         ys = np.array([p["y"] - wy for p in pts], dtype=np.float32)
#         zs = np.array([p["z"] - wz for p in pts], dtype=np.float32)

#         dx, dy, dz = (xs.max() - xs.min()), (ys.max() - ys.min()), (zs.max() - zs.min())
#         diag = float((dx * dx + dy * dy + dz * dz) ** 0.5) or 1.0
#         xs, ys, zs = xs / diag, ys / diag, zs / diag

#         arr = np.empty(63, dtype=np.float32)
#         arr[0::3], arr[1::3], arr[2::3] = xs, ys, zs
#         return arr.reshape(1, -1)

#     def infer(self, points: List[Dict[str, float]]) -> Dict[str, Any]:
#         x = self._preprocess(points)
#         in_idx = self.input_details[0]["index"]
#         in_shape = self.input_details[0]["shape"]
#         x = x.astype(np.float32)
#         if int(np.prod(in_shape)) == x.size:
#             x = x.reshape(in_shape)
#         self.interp.set_tensor(in_idx, x)
#         self.interp.invoke()
#         out_idx = self.output_details[0]["index"]
#         y = self.interp.get_tensor(out_idx)[0].tolist()
#         top = int(np.argmax(y))
#         score = float(y[top])
#         label = LABELS[top] if LABELS and 0 <= top < len(LABELS) else f"class_{top}"
#         return {"label": label, "score": score, "index": top}


# async def run_worker():
#     # 1) 모델 로드 (한 번만)
#     try:
#         logger.info("모델 로드 시작")
#         clf = GestureClassifier(DEFAULT_TFLITE)
#         logger.info("모델 로드 완료")
#     except Exception:
#         logger.error("모델 로드 실패:\n%s", traceback.format_exc())
#         return

#     # 2) 서버 연결 및 메시지 루프
#     backoff = 1
#     while True:
#         try:
#             logger.info("서버 접속 시도 %s", URL)
#             async with websockets.connect(
#                 URL, max_size=10 * 1024 * 1024, ping_interval=20, ping_timeout=20
#             ) as ws:
#                 logger.info("서버 접속 완료. 메시지 대기중...")
#                 backoff = 1

#                 while True:
#                     msg = await ws.recv()
#                     if not isinstance(msg, str):
#                         logger.debug("binary msg len=%d", len(msg))
#                         continue

#                     try:
#                         data = json.loads(msg)
#                     except Exception:
#                         logger.exception("JSON 파싱 실패")
#                         continue

#                     mtype = data.get("type")
#                     if mtype == "bind":
#                         logger.info("바인딩됨 room=%s", data.get("room"))
#                         continue
#                     if mtype != "hand_landmarks":
#                         logger.debug("무시 타입=%s", mtype)
#                         continue

#                     hands: List[List[Dict[str, float]]] = data.get("landmarks") or []
#                     __main__ = hands[0] if hands else []
#                     if not __main__:
#                         logger.debug("빈 landmarks 수신")
#                         continue

#                     try:
#                         result = clf.infer(__main__)
#                         out = {
#                             "type": "ai_result",
#                             "text": result["label"],
#                             "score": result["score"],
#                             "frame_id": data.get("frame_id"),
#                             "room_id": data.get("room_id"),
#                         }
#                         await ws.send(json.dumps(out))
#                         logger.info(
#                             "번역 완료 → %s (score=%.2f, room=%s, frame=%s)",
#                             out["text"], out["score"], out["room_id"], out["frame_id"]
#                         )
#                     except Exception:
#                         logger.error("번역 실패:\n%s", traceback.format_exc())

#         except (websockets.ConnectionClosed, ConnectionRefusedError) as e:
#             logger.warning("서버 연결 끊김: %s. %d초 후 재시도", e, backoff)
#         except Exception:
#             logger.error("알 수 없는 오류:\n%s", traceback.format_exc())

#         await asyncio.sleep(backoff)
#         backoff = min(backoff * 2, 10)
# async def recognize_sign(payload: Dict[str, Any]) -> Tuple[str, float]:
#     """
#     수어 인식 → 원문 텍스트와 신뢰도(score)를 반환
#     """
#     return payload.get("text") or "[recognized]", 0.9

# async def translate_text(text: str) -> str:
#     """
#     번역 수행 (현재는 그대로 반환)
#     """
#     return text

# async def handle_frame(frame_payload: Dict[str, Any]) -> Dict[str, Any]:
#     """FastAPI에서 받은 메시지 처리"""
#     corr_id = frame_payload.get("corr_id") or str(uuid.uuid4())
#     t0 = time.time()

#     recognized, score = await recognize_sign(frame_payload)
#     translated = await translate_text(recognized)

#     ms = int((time.time() - t0) * 1000)
#     log.info({
#         "event": "inference",
#         "corr_id": corr_id,
#         "origin": recognized,
#         "translated": translated,
#         "ms": ms,
#         "score": score,
#     })
#     return {"corr_id": corr_id, "text": translated, "score": score}

# if __name__ == "__main__":
#     asyncio.run(run_worker())

import os
import json
import asyncio
import logging
import traceback
from typing import Any, Dict, List, Tuple
import time
import uuid
import numpy as np
import websockets  # pip install websockets

# ---------- 로깅 ----------
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    force=True,
)
logger = logging.getLogger("ai_worker")

# ---------- TFLite Interpreter ----------
try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except Exception:
    from tflite_runtime.interpreter import Interpreter  # type: ignore

# ---------- 환경변수 ----------
WS_URL   = os.getenv("WS_URL", "ws://ai:8001/ai")         # 내부 도커 네트워크 기준
AI_TOKEN = os.getenv("AI_TOKEN") 
ROLE     = os.getenv("ROLE", "ai")                        # 워커는 ai
ROOM     = os.getenv("ROOM", "")                          # 필요시 room 지정(없으면 Hub가 바인딩)

qs = []
if AI_TOKEN:
    qs.append(f"token={AI_TOKEN}")
if ROLE:
    qs.append(f"role={ROLE}")
if ROOM:
    qs.append(f"room={ROOM}")
URL = WS_URL + (("?" + "&".join(qs)) if qs else "")

# 모델/라벨
DEFAULT_TFLITE = os.getenv(
    "TFLITE_PATH",
    "/ai/AI_Language/models/multi_hand_gesture_classifier.tflite"
)
LABELS = os.getenv("LABELS", "")
LABELS = LABELS.split(",") if LABELS else None


class GestureClassifier:
    """TFLite 추론 래퍼: 21개 랜드마크(x,y,z) -> (1,63)"""
    def __init__(self, tflite_path: str):
        self.interp = Interpreter(model_path=tflite_path)
        self.interp.allocate_tensors()
        self.input_details = self.interp.get_input_details()
        self.output_details = self.interp.get_output_details()
        in0 = self.input_details[0]["shape"]
        out0 = self.output_details[0]["shape"]
        logger.info("[worker] TFLite input=%s output=%s", in0, out0)

    @staticmethod
    def _fix_length_21(points: List[Dict[str, float]]) -> List[Dict[str, float]]:
        pts = list(points[:21])
        while len(pts) < 21:
            pts.append({"x": 0.0, "y": 0.0, "z": 0.0})
        return pts

    def _preprocess(self, points: List[Dict[str, float]]) -> np.ndarray:
        if not points:
            raise ValueError("Empty landmarks")
        pts = self._fix_length_21(points)

        wx, wy, wz = pts[0]["x"], pts[0]["y"], pts[0]["z"]
        xs = np.array([p["x"] - wx for p in pts], dtype=np.float32)
        ys = np.array([p["y"] - wy for p in pts], dtype=np.float32)
        zs = np.array([p["z"] - wz for p in pts], dtype=np.float32)

        dx, dy, dz = (xs.max() - xs.min()), (ys.max() - ys.min()), (zs.max() - zs.min())
        diag = float((dx * dx + dy * dy + dz * dz) ** 0.5) or 1.0
        xs, ys, zs = xs / diag, ys / diag, zs / diag

        arr = np.empty(63, dtype=np.float32)
        arr[0::3], arr[1::3], arr[2::3] = xs, ys, zs
        return arr.reshape(1, -1)

    def infer(self, points: List[Dict[str, float]]) -> Dict[str, Any]:
        x = self._preprocess(points)
        in_idx = self.input_details[0]["index"]
        in_shape = self.input_details[0]["shape"]
        x = x.astype(np.float32)
        if int(np.prod(in_shape)) == x.size:
            x = x.reshape(in_shape)
        self.interp.set_tensor(in_idx, x)
        self.interp.invoke()
        out_idx = self.output_details[0]["index"]
        y = self.interp.get_tensor(out_idx)[0].tolist()
        top = int(np.argmax(y))
        score = float(y[top])
        label = LABELS[top] if LABELS and 0 <= top < len(LABELS) else f"class_{top}"
        return {"label": label, "score": score, "index": top}


async def run_worker():
    # 1) 모델 로드 (한 번만)
    try:
        logger.info("모델 로드 시작")
        clf = GestureClassifier(DEFAULT_TFLITE)
        logger.info("모델 로드 완료")
    except Exception:
        logger.error("모델 로드 실패:\n%s", traceback.format_exc())
        return

    # 2) 서버 연결 및 메시지 루프
    backoff = 1
    while True:
        try:
            logger.info("서버 접속 시도 %s", URL)
            async with websockets.connect(
                URL, max_size=10 * 1024 * 1024, ping_interval=20, ping_timeout=20
            ) as ws:
                logger.info("서버 접속 완료. 메시지 대기중...")
                backoff = 1

                while True:
                    msg = await ws.recv()
                    if not isinstance(msg, str):
                        logger.debug("binary msg len=%d", len(msg))
                        continue

                    try:
                        data = json.loads(msg)
                    except Exception:
                        logger.exception("JSON 파싱 실패")
                        continue

                    mtype = data.get("type")
                    if mtype == "bind":
                        # Hub가 워커를 방에 바인딩할 때 수신
                        logger.info("바인딩됨 room=%s", data.get("room"))
                        continue

                    # ---------- 여기부터 핵심 변경 ----------
                    if mtype != "coords":  # <<< CHANGED (기존: hand_landmarks)
                        logger.debug("무시 타입=%s", mtype)
                        continue

                    # 서버가 중계한 표준 스키마: hands = [ [ {x,y,z}*21 ], [ ... ] ]
                    hands: List[List[Dict[str, float]]] = data.get("hands") or []  # <<< CHANGED
                    __main__ = hands[0] if hands else []
                    if not __main__:
                        logger.debug("빈 landmarks 수신")
                        continue

                    try:
                        result = clf.infer(__main__)
                        out = {
                            "type": "ai_result",             # <<< CHANGED (응답 타입 통일)
                            "text": result["label"],
                            "score": result["score"],
                            "frame_id": data.get("frame_id"),
                            "room_id": data.get("room_id"),
                            "corr_id": data.get("corr_id"),  # <<< CHANGED (트래킹 유지)
                        }
                        await ws.send(json.dumps(out))
                        logger.info(
                            "번역 완료 → %s (score=%.2f, room=%s, frame=%s, corr=%s)",
                            out["text"], out["score"], out["room_id"], out["frame_id"], out["corr_id"]
                        )
                    except Exception:
                        logger.error("번역 실패:\n%s", traceback.format_exc())
                    # ---------- 핵심 변경 끝 ----------

        except (websockets.ConnectionClosed, ConnectionRefusedError) as e:
            logger.warning("서버 연결 끊김: %s. %d초 후 재시도", e, backoff)
        except Exception:
            logger.error("알 수 없는 오류:\n%s", traceback.format_exc())

        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 10)


# 아래 함수들은 현재 미사용(서버측에 동일 기능이 있으므로 유지만)
async def recognize_sign(payload: Dict[str, Any]) -> Tuple[str, float]:
    return payload.get("text") or "[recognized]", 0.9

async def translate_text(text: str) -> str:
    return text

async def handle_frame(frame_payload: Dict[str, Any]) -> Dict[str, Any]:
    corr_id = frame_payload.get("corr_id") or str(uuid.uuid4())
    t0 = time.time()
    recognized, score = await recognize_sign(frame_payload)
    translated = await translate_text(recognized)
    ms = int((time.time() - t0) * 1000)
    logger.info({
        "event": "inference",
        "corr_id": corr_id,
        "origin": recognized,
        "translated": translated,
        "ms": ms,
        "score": score,
    })
    return {"corr_id": corr_id, "text": translated, "score": score}


if __name__ == "__main__":
    asyncio.run(run_worker())