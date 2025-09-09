import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import modules.holistic_module as hm
from tensorflow.keras.models import load_model  # (안 써도 무방)
import math
from modules.utils import Vector_Normalization
from PIL import ImageFont, ImageDraw, Image

# ----------------- 설정값 -----------------
fontpath = "fonts/HMKMMAG.TTF"
font = ImageFont.truetype(fontpath, 40)

actions = [
    'ㄱ','ㄴ','ㄷ','ㄹ','ㅁ','ㅂ','ㅅ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ',
    'ㅏ','ㅑ','ㅓ','ㅕ','ㅗ','ㅛ','ㅜ','ㅠ','ㅡ','ㅣ',
    'ㅐ','ㅒ','ㅔ','ㅖ','ㅢ','ㅚ','ㅟ'
]
seq_length = 10
MIN_CONF = float(os.getenv("MIN_CONFIDENCE", "0.30"))   # 환경변수로 조절 가능
ALWAYS_EMIT = os.getenv("ALWAYS_EMIT_CAPTION", "") == "1"  # 디버깅용: 점수 낮아도 표시

# ----------------- MediaPipe -----------------
detector = hm.HolisticDetector(min_detection_confidence=0.3)

# ----------------- TFLite -----------------
interpreter = tf.lite.Interpreter(model_path="models/multi_hand_gesture_classifier.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 디버깅용 정보 출력
def _qinfo(d):
    q = None
    if "quantization_parameters" in d:
        qp = d["quantization_parameters"]
        s = qp.get("scales", [])
        z = qp.get("zero_points", [])
        if len(s) > 0:
            q = (float(s[0] or 1.0), int(z[0] if len(z) else 0))
    if not q and "quantization" in d and d["quantization"] is not None:
        s, z = d["quantization"]
        if isinstance(s, (list, tuple)): s = s[0] if s else 1.0
        if isinstance(z, (list, tuple)): z = z[0] if z else 0
        q = (float(s or 1.0), int(z or 0))
    return q

in_q = _qinfo(input_details[0])
out_q = _qinfo(output_details[0])
print(f"[TFLite] input dtype={input_details[0]['dtype']} quant={in_q}")
print(f"[TFLite] output dtype={output_details[0]['dtype']} quant={out_q}")

def _softmax(row: np.ndarray) -> np.ndarray:
    r = row.astype(np.float32)
    r -= np.max(r)
    e = np.exp(r)
    s = np.sum(e)
    return e / s if s > 0 else np.zeros_like(e, dtype=np.float32)

def _quantize_input(x: np.ndarray, det: dict) -> np.ndarray:
    dtype = det["dtype"]
    if np.issubdtype(dtype, np.floating):
        return x.astype(np.float32, copy=False)
    # int8/uint8
    q = _qinfo(det)
    scale, zero = q if q else (1.0, 0)
    qx = np.round(x / (scale or 1.0) + zero)
    # dtype 범위로 클리핑
    if dtype == np.uint8:
        qx = np.clip(qx, 0, 255)
    elif dtype == np.int8:
        qx = np.clip(qx, -128, 127)
    return qx.astype(dtype, copy=False)

def _dequantize_output(y: np.ndarray, det: dict) -> np.ndarray:
    dtype = det["dtype"]
    y = y.astype(np.float32, copy=False)
    if np.issubdtype(dtype, np.floating):
        return y
    # int8/uint8 -> float로 복구
    q = _qinfo(det)
    scale, zero = q if q else (1.0, 0)
    return (y - float(zero)) * float(scale)

def predict_once(x_1x10x55: np.ndarray):
    """(1,10,55) -> (idx, prob) : 양자화 대응 + softmax 점수"""
    xin = _quantize_input(x_1x10x55, input_details[0])
    interpreter.set_tensor(input_details[0]['index'], xin)
    interpreter.invoke()

    y = interpreter.get_tensor(output_details[0]['index'])[0]   # (num_classes,)
    y_float = _dequantize_output(y, output_details[0])          # logits or scores -> float
    probs = _softmax(y_float)                                   # 확률화(로그릿일 수도 있으니)
    i_pred = int(np.argmax(probs))
    conf = float(probs[i_pred])
    return i_pred, conf, probs

# ----------------- 비디오 루프 -----------------
cap = cv2.VideoCapture(0)
seq = []
action_seq = []
last_action = None

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = detector.findHolistic(img, draw=True)
    # _, left_hand_lmList = detector.findLefthandLandmark(img)
    _, right_hand_lmList = detector.findRighthandLandmark(img)

    if right_hand_lmList is not None:
        joint = np.zeros((42, 2), dtype=np.float32)

        # 오른손 21개 좌표만 사용 (0~20 채움)
        for j, lm in enumerate(right_hand_lmList.landmark):
            joint[j] = [lm.x, lm.y]

        # 벡터 정규화 -> (vector(20x2), angle(15,))
        vector, angle_label = Vector_Normalization(joint)

        # (55,) 특징
        d = np.concatenate([vector.flatten(), angle_label.flatten()]).astype(np.float32)
        seq.append(d)

        if len(seq) < seq_length:
            # 버퍼 차는 동안 화면만 표기
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            draw.text((10, 30), f'buffer: {len(seq)}/{seq_length}', font=font, fill=(255,255,255))
            img = np.array(img_pil)
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # 시퀀스 윈도우 (1,10,55)
        window = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

        # 예측
        i_pred, conf, probs = predict_once(window)

        # 임계값 체크
        if conf < MIN_CONF and not ALWAYS_EMIT:
            # 점수 낮으면 스킵
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            draw.text((10, 30), f'conf: {conf:.3f} (skip)', font=font, fill=(255,255,255))
            img = np.array(img_pil)
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        action = actions[i_pred]
        action_seq.append(action)

        this_action = action
        if len(action_seq) >= 3 and not (action_seq[-1] == action_seq[-2] == action_seq[-3]):
            # 최근 3개가 같지 않으면 안정화 전 상태로 간주
            this_action = action_seq[-1]

        # 표시
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 30), f'{this_action}  (conf={conf:.3f})', font=font, fill=(255, 255, 255))
        img = np.array(img_pil)

    # 화면 출력 / ESC 종료
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
