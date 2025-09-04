import sys
import os   
# sys.path.append('pingpong')
# from pingpong.pingpongthread import PingPongThread
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import modules.holistic_module as hm
from tensorflow.keras.models import load_model
import math, time
from modules.utils import Vector_Normalization
from PIL import ImageFont, ImageDraw, Image
from collections import deque
from unicode import join_jamos

# <<< ADDED: WebSocket 송신을 위한 모듈/상수/헬퍼 >>>
import asyncio, json, uuid
import websockets  # pip install websockets

# 서버(웹소켓) 연결 정보
WS_URI = os.getenv(
    "WS_URI",
    "ws://localhost:8001/ai?token=change-me-dev&role=ai&room=debug"  # 로컬 테스트
)
WS_ORIGIN = {"Origin": os.getenv("WS_ORIGIN", "https://5range.site")}  # Origin 체크 시 필요

def _extract_21(hand):
    """MediaPipe hand → 21개 (x,y,z) dict 리스트로 변환"""
    return [{"x": float(lm.x), "y": float(lm.y), "z": float(getattr(lm, "z", 0.0))}
            for lm in hand.landmark[:21]]

async def _send_coords(ws, l_hand, r_hand, room_id="debug", frame_id=None):
    """좌표 payload를 websockets.py로 전송"""
    hands = []
    if l_hand is not None:
        hands.append(_extract_21(l_hand))
    if r_hand is not None:
        hands.append(_extract_21(r_hand))
    if not hands:
        return
    msg = {
        "type": "coords",
        "room_id": room_id,
        "corr_id": str(uuid.uuid4()),
        "hands": hands,
    }
    if frame_id is not None:
        msg["frame_id"] = frame_id
    await ws.send(json.dumps(msg))
# <<< ADDED 끝 >>>

BIMANUAL_ORDER = "left_to_right"

# === UI: 윈도우/버튼 설정 & 마우스 콜백 ===
WINDOW_NAME = 'Sign→Korean Compose'

# 버튼 위치 (x1, y1, x2, y2)
BTN_CLEAR   = (10,  90, 120, 125)   # "지우기"
BTN_NEWLINE = (130, 90, 260, 125)   # "줄바꿈"

_trigger_clear = False
_trigger_newline = False

def _point_in_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2

def _on_mouse(event, x, y, flags, param):
    global _trigger_clear, _trigger_newline
    if event == cv2.EVENT_LBUTTONDOWN:
        if _point_in_rect(x, y, BTN_CLEAR):
            _trigger_clear = True
        elif _point_in_rect(x, y, BTN_NEWLINE):
            _trigger_newline = True
            
# =========================
# 설정: 폰트 & 액션(자모 레이블)
# =========================
FONT_PATH = "fonts/HMKMMAG.TTF"  # 없으면 자동 폴백
FONT_SIZE = 40

ACTIONS = [
    'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
    'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
    'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ'
]
SEQ_LENGTH = 10
CONF_TH = 0.90               # 신뢰도 임계치
STABLE_COUNT = 3            # 같은 자모가 연속 몇 번 나와야 확정할지
SYL_TIMEOUT = 3           # 초/중/종성 조합 후 이 시간(초) 입력 없으면 음절 확정
WORD_TIMEOUT = 5          # 더 오래 입력 없으면 띄어쓰기
SHOW_MAX_CHARS = 40          # 화면에 보여줄 최대 글자수
# ===== 추가: 흔들림/중복 방지 튜닝 =====
HOLD_MIN_FR = 5              # 손이 일정히 고정된 프레임 수(최소)
COOLDOWN_FR = 6              # 같은 글자 확정 후 잠깐 쉬기
MOVE_TH = 0.010              # 손 중심 이동 임계치(정규화 좌표)
EMA_MOMENTUM = 0.7           # 확률 지수이동평균(0.6~0.8 권장)

# =========================
# 한글 합성 유틸
# =========================
CHO_LIST = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
JUNG_LIST = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
JONG_LIST = ['', 'ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

VOWELS = set(['ㅏ','ㅑ','ㅓ','ㅕ','ㅗ','ㅛ','ㅜ','ㅠ','ㅡ','ㅣ','ㅐ','ㅒ','ㅔ','ㅖ','ㅢ','ㅚ','ㅟ'])
CONSONANTS = set([a for a in ACTIONS if a not in VOWELS])

# 겹초성(쌍자음) 규칙: 같은 자음 반복 시 초성 쌍자음으로
CHO_DOUBLE = {('ㄱ','ㄱ'):'ㄲ', ('ㄷ','ㄷ'):'ㄸ', ('ㅂ','ㅂ'):'ㅃ', ('ㅅ','ㅅ'):'ㅆ', ('ㅈ','ㅈ'):'ㅉ'}
# 겹받침 규칙
JONG_DOUBLE = {
    ('ㄱ','ㅅ'):'ㄳ', ('ㄴ','ㅈ'):'ㄵ', ('ㄴ','ㅎ'):'ㄶ',
    ('ㄹ','ㄱ'):'ㄺ', ('ㄹ','ㅁ'):'ㄻ', ('ㄹ','ㅂ'):'ㄼ',
    ('ㄹ','ㅅ'):'ㄽ', ('ㄹ','ㅌ'):'ㄾ', ('ㄹ','ㅍ'):'ㄿ', ('ㄹ','ㅎ'):'ㅀ',
    ('ㅂ','ㅅ'):'ㅄ'
}
# 복모음 규칙
JUNG_COMPOSE = {
    ('ㅗ','ㅏ'):'ㅘ', ('ㅗ','ㅐ'):'ㅙ', ('ㅗ','ㅣ'):'ㅚ',
    ('ㅜ','ㅓ'):'ㅝ', ('ㅜ','ㅔ'):'ㅞ', ('ㅜ','ㅣ'):'ㅟ',
    ('ㅡ','ㅣ'):'ㅢ'
}

def _idx_or_none(lst, x):
    try:
        return lst.index(x)
    except ValueError:
        return None

def compose_syllable(cho, jung, jong=''):
    """초/중/(종)으로 한 글자 합성"""
    ic = _idx_or_none(CHO_LIST, cho)
    iv = _idx_or_none(JUNG_LIST, jung)
    if ic is None or iv is None:
        return (cho or '') + (jung or '') + (jong or '')
    ij = _idx_or_none(JONG_LIST, jong)
    if ij is None: ij = 0
    return chr(0xAC00 + (ic * 21 + iv) * 28 + ij)

class HangulComposer:
    def __init__(self, syl_timeout=SYL_TIMEOUT, word_timeout=WORD_TIMEOUT):
        self.reset_block()
        self.text = ""
        self.last_input_t = time.time()
        self.syl_timeout = syl_timeout
        self.word_timeout = word_timeout

    def reset_block(self):
        self.cho = None
        self.jung = None
        self.jong = None
        self._last_jamo = None

    def _try_compose_jung(self, base, add):
        return JUNG_COMPOSE.get((base, add))

    def _try_compose_cho_double(self, base, add):
        return CHO_DOUBLE.get((base, add))

    def _try_compose_jong_double(self, base, add):
        return JONG_DOUBLE.get((base, add))

    def feed(self, jamo):
        now = time.time()
        self.last_input_t = now

        # 1) 아직 중성이 없을 때
        if self.jung is None:
            if jamo in CONSONANTS:
                if self.cho is None:
                    # 초성 시작
                    self.cho = jamo
                else:
                    # 같은 자음 두 번 → 쌍자음 시도
                    dbl = self._try_compose_cho_double(self.cho, jamo)
                    if dbl:
                        self.cho = dbl
                    else:
                        # 새 초성 시작(이전 블럭은 초성만이라 글자 아님 → 그냥 텍스트로 추가)
                        self._commit_if_partial()
                        self.cho = jamo
            else:  # 모음
                if self.cho is None:
                    # 초성 없이 모음만 → 임시로 ㅇ 초성
                    self.cho = 'ㅇ'
                self.jung = jamo
            return

        # 2) 중성은 있고 종성은 아직
        if self.jong is None:
            if jamo in VOWELS:
                # 복모음 시도
                new_v = self._try_compose_jung(self.jung, jamo)
                if new_v:
                    self.jung = new_v
                else:
                    # 새 모음 → 이전 음절 확정하고 이번은 새 음절 시작(초성=ㅇ 가정)
                    self.commit_block()
                    self.cho, self.jung = 'ㅇ', jamo
            else:
                # 종성 후보
                self.jong = jamo
            return

        # 3) 종성이 있는 상태
        if jamo in CONSONANTS:
            # 겹받침 시도
            new_jong = self._try_compose_jong_double(self.jong, jamo)
            if new_jong:
                self.jong = new_jong
            else:
                # 겹받침 불가 → 이전 음절 확정, 새 음절 초성 찍고 대기
                self.commit_block()
                self.cho = jamo
        else:
            # 모음이 오면 이전 종성은 다음 음절의 초성으로 이동
            carry_cho = self.jong
            self.jong = None
            self.commit_block()
            # 종성 한글(겹받침 포함)에서 초성 가능한 단위만 남김(겹받침이면 앞 요소를 초성으로 가정)
            if carry_cho in JONG_LIST:
                # 종성→초성 맵핑(동일 문자 대부분 호환됨)
                self.cho = carry_cho if carry_cho != '' else 'ㅇ'
            else:
                self.cho = 'ㅇ'
            self.jung = jamo

    def _commit_if_partial(self):
        # 초성 단독 등은 그대로 텍스트에 추가
        if self.cho and not self.jung and not self.jong:
            self.text += self.cho
        elif self.cho and self.jung:
            # 종성 없이도 한 글자
            self.text += compose_syllable(self.cho, self.jung)
        self.reset_block()

    def commit_block(self):
        if self.cho and self.jung:
            ch = compose_syllable(self.cho, self.jung, self.jong or '')
            self.text += ch
        else:
            # 불완전 블럭은 있는 그대로
            if self.cho: self.text += self.cho
            if self.jung: self.text += self.jung
            if self.jong: self.text += self.jong
        self.reset_block()

    def maybe_timeout(self):
        now = time.time()
        # 음절 확정
        if (self.cho or self.jung or self.jong) and (now - self.last_input_t >= self.syl_timeout):
            self.commit_block()
        # 띄어쓰기
        if (now - self.last_input_t >= self.word_timeout):
            if len(self.text) > 0 and not self.text.endswith(' '):
                self.text += ' '

    def backspace(self):
        # 블럭 지우기 우선
        if self.cho or self.jung or self.jong:
            self.reset_block()
            return
        if len(self.text) > 0:
            self.text = self.text[:-1]

    def get_preview_block(self):
        if self.cho and self.jung:
            return compose_syllable(self.cho, self.jung, self.jong or '')
        # 미완성은 자모 그대로 보여줌
        return (self.cho or '') + (self.jung or '') + (self.jong or '')

    def get_text(self):
        return self.text

# =========================
# 예측 디바운서
# =========================
class JamoStabilizer:
    def __init__(self, stable_count=STABLE_COUNT):
        self.buf = deque(maxlen=stable_count)
        self.last_emitted = None

    def push(self, jamo):
        self.buf.append(jamo)
        if len(self.buf) == self.buf.maxlen and len(set(self.buf)) == 1:
            if self.last_emitted != jamo:
                self.last_emitted = jamo
                return jamo
        return None
    
seqL, seqR = [], []
stabL = JamoStabilizer(STABLE_COUNT)
stabR = JamoStabilizer(STABLE_COUNT)

# =========================
# 폰트 준비
# =========================
def get_font(path=FONT_PATH, size=FONT_SIZE):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        # 폴백: PIL 기본 폰트(한글 글리프 없을 수 있음)
        return ImageFont.load_default()

font = get_font()

# =========================
# MediaPipe & TFLite 로딩
# =========================
detector = hm.HolisticDetector(min_detection_confidence=0.3)

from pathlib import Path, PurePath
BASE_DIR = Path(__file__).resolve().parent          # ...\Sign_Language_Translation
MODEL_PATH = str((BASE_DIR.parent / "models" / "multi_hand_gesture_classifier.tflite").resolve())

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =========================
# 실행 (WebSocket 송신 포함)
# =========================
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, _on_mouse)

seq = []
stabilizer = JamoStabilizer(STABLE_COUNT)
composer = HangulComposer(SYL_TIMEOUT, WORD_TIMEOUT)

last_action_draw = ''
last_conf = 0.0

# ===== 추가: 스무딩/홀드/쿨다운/이동량 상태 =====
y_smooth_L = None
y_smooth_R = None
last_center = None
hold_frames = 0
cooldownL = 0
cooldownR = 0

# <<< ADDED: OpenCV 루프를 비동기로 실행하여 WS로 좌표 전송 >>>
async def _run_with_ws():
    global y_smooth_L, y_smooth_R, last_center, hold_frames
    global cooldownL, cooldownR, last_action_draw, last_conf
    global _trigger_clear, _trigger_newline, seqL, seqR

    # websockets.py에 연결 (Origin 헤더는 서버 설정에 맞춰 유지)
    async with websockets.connect(WS_URI, extra_headers=WS_ORIGIN) as ws:
        print(f"[WS] connected -> {WS_URI}")
        cap = cv2.VideoCapture(0)
        frame_id = 0

        try:
            while cap.isOpened():
                ret, img = cap.read()
                if not ret:
                    break

                img = detector.findHolistic(img, draw=True)
                # 왼손/오른손
                _, r_hand = detector.findRighthandLandmark(img)
                _, l_hand = detector.findLefthandLandmark(img)

                hasL = l_hand is not None
                hasR = r_hand is not None

                # === 여기서 좌표를 서버로 전송 ===
                if hasL or hasR:
                    await _send_coords(ws, l_hand, r_hand, room_id="debug", frame_id=frame_id)

                # ---- (아래는 기존 추론/합성/오버레이 로직 그대로 유지) ----
                if hasL or hasR:
                    # --- 손 1개 → 특징 벡터 만들기(0..20 슬롯 사용) ---
                    def _feat_from_hand(hand):
                        if hand is None:
                            return None
                        joint = np.zeros((42, 2), dtype=np.float32)
                        for j, lm in enumerate(hand.landmark):
                            if j < 21:
                                joint[j] = [lm.x, lm.y]
                        vector, angle_label = Vector_Normalization(joint)
                        return np.concatenate([vector.flatten(), angle_label.flatten()])

                    # --- 손 중심 이동(흔들림) 측정 → 흔들리면 버퍼 리셋 ---
                    def _center_xy(hand):
                        xs = [lm.x for lm in hand.landmark[:21]]
                        ys = [lm.y for lm in hand.landmark[:21]]
                        return float(np.mean(xs)), float(np.mean(ys))

                    centers = []
                    if hasL: centers.append(_center_xy(l_hand))
                    if hasR: centers.append(_center_xy(r_hand))
                    center_now = None if not centers else tuple(np.mean(centers, axis=0))

                    if center_now is not None:
                        if last_center is not None:
                            dx = center_now[0] - last_center[0]
                            dy = center_now[1] - last_center[1]
                            dist = (dx*dx + dy*dy)**0.5
                            if dist > MOVE_TH:
                                hold_frames = 0
                                stabL.buf.clear(); stabR.buf.clear()
                            else:
                                hold_frames = hold_frames + 1 if isinstance(hold_frames, int) else 1
                        else:
                            hold_frames = 1
                        # NOTE: 파이썬 클로저에서 쓰는 바깥 변수는 nonlocal이 필요할 수 있음.
                        # 여기서는 단순 사용으로 둡니다.
                        # last_center를 갱신
                        # nonlocal last_center
                        last_center = center_now

                    outs = []  # [(xpos, jamo)]

                    # ===== 왼손 추론 (EMA + 홀드 + 쿨다운) =====
                    fL = _feat_from_hand(l_hand)
                    if fL is not None:
                        # nonlocal y_smooth_L, cooldownL, last_action_draw, last_conf
                        seqL.append(fL)
                        if len(seqL) >= SEQ_LENGTH:
                            inpL = np.expand_dims(np.array(seqL[-SEQ_LENGTH:], dtype=np.float32), axis=0)
                            interpreter.set_tensor(input_details[0]['index'], inpL)
                            interpreter.invoke()
                            yL_raw = interpreter.get_tensor(output_details[0]['index'])[0]
                            # 확률 스무딩(EMA)
                            if y_smooth_L is None:
                                y_smooth_L = yL_raw.copy()
                            else:
                                y_smooth_L = EMA_MOMENTUM * y_smooth_L + (1.0 - EMA_MOMENTUM) * yL_raw
                            iL = int(np.argmax(y_smooth_L)); confL = float(y_smooth_L[iL])
                            last_action_draw, last_conf = ACTIONS[iL], confL  # 디버그 표시

                            if confL >= CONF_TH and hold_frames >= HOLD_MIN_FR and cooldownL == 0:
                                actL = ACTIONS[iL]
                                stL = stabL.push(actL)
                                if stL is not None:
                                    xL = float(np.mean([lm.x for lm in l_hand.landmark[:21]])) if l_hand else 0.0
                                    outs.append((xL, stL))
                                    cooldownL = COOLDOWN_FR

                    # ===== 오른손 추론 (EMA + 홀드 + 쿨다운) =====
                    fR = _feat_from_hand(r_hand)
                    if fR is not None:
                        # nonlocal y_smooth_R, cooldownR, last_action_draw, last_conf
                        seqR.append(fR)
                        if len(seqR) >= SEQ_LENGTH:
                            inpR = np.expand_dims(np.array(seqR[-SEQ_LENGTH:], dtype=np.float32), axis=0)
                            interpreter.set_tensor(input_details[0]['index'], inpR)
                            interpreter.invoke()
                            yR_raw = interpreter.get_tensor(output_details[0]['index'])[0]
                            if y_smooth_R is None:
                                y_smooth_R = yR_raw.copy()
                            else:
                                y_smooth_R = EMA_MOMENTUM * y_smooth_R + (1.0 - EMA_MOMENTUM) * yR_raw
                            iR = int(np.argmax(y_smooth_R)); confR = float(y_smooth_R[iR])
                            last_action_draw, last_conf = ACTIONS[iR], confR

                            if confR >= CONF_TH and hold_frames >= HOLD_MIN_FR and cooldownR == 0:
                                actR = ACTIONS[iR]
                                stR = stabR.push(actR)
                                if stR is not None:
                                    xR = float(np.mean([lm.x for lm in r_hand.landmark[:21]])) if r_hand else 1.0
                                    outs.append((xR, stR))
                                    cooldownR = COOLDOWN_FR

                    # --- 쿨다운 감소 ---
                    if cooldownL > 0: cooldownL -= 1
                    if cooldownR > 0: cooldownR -= 1

                    # ===== 이번 프레임 확정 출력 적용 =====
                    if outs:
                        outs.sort(key=lambda t: t[0])  # 왼쪽→오른쪽
                        for _, jamo in outs:
                            composer.feed(jamo)

                # 타임아웃 체크(음절 확정 & 자동 띄어쓰기)
                composer.maybe_timeout()

                # =========================
                # 오버레이: 텍스트 그리기
                # =========================
                preview_block = composer.get_preview_block()
                joined_text = join_jamos(composer.get_text() + preview_block)   # 자모 → 음절 조합
                text_line = joined_text[-SHOW_MAX_CHARS:]
                
                lines = joined_text.splitlines() or [joined_text]
                to_draw = '\n'.join(lines[-3:])

                img_pil = Image.fromarray(img)
                draw = ImageDraw.Draw(img_pil)

                # 현재 문장(프리뷰 포함)
                draw.rectangle([(0,0),(img.shape[1],30)], fill=(0,0,0,128))
                draw.text((10, 10), f"{text_line}", font=font, fill=(255,255,255), spacing=6)

                # 디버그: 마지막 확신 자모/신뢰도
                if last_action_draw:
                    draw.text((10, 45), f"Pred: {last_action_draw} ({last_conf:.2f})", font=font, fill=(90,90,90))
                
                draw.rectangle([(BTN_CLEAR[0], BTN_CLEAR[1]), (BTN_CLEAR[2], BTN_CLEAR[3])], fill=(30,30,30))
                draw.text((BTN_CLEAR[0]+12, BTN_CLEAR[1]+4), "지우기", font=font, fill=(255,255,255))

                draw.rectangle([(BTN_NEWLINE[0], BTN_NEWLINE[1]), (BTN_NEWLINE[2], BTN_NEWLINE[3])], fill=(30,30,30))
                draw.text((BTN_NEWLINE[0]+12, BTN_NEWLINE[1]+4), "줄바꿈", font=font, fill=(255,255,255))

                img = np.array(img_pil)
                cv2.imshow(WINDOW_NAME, img)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == 8:  # Backspace
                    stabilizer.last_emitted = None
                    composer.backspace()
                elif key == 32:  # Space -> 공백 커밋
                    composer.commit_block()
                    if not composer.get_text().endswith(' '):
                        composer.text += ' '
                elif key in (ord('c'), ord('C')):  # Clear
                    stabilizer.last_emitted = None
                    composer.reset_block()
                    composer.text = ""
                elif key in (ord('n'), ord('N'), 13):  # Newline (Enter 포함)
                    composer.commit_block()
                    composer.text += '\n'

                # --- 마우스 버튼 트리거 ---
                global _trigger_clear, _trigger_newline
                if _trigger_clear:
                    stabilizer.last_emitted = None
                    composer.reset_block()
                    composer.text = ""
                    _trigger_clear = False

                if _trigger_newline:
                    composer.commit_block()
                    composer.text += '\n'
                    _trigger_newline = False

                frame_id += 1

        finally:
            cap.release()
            cv2.destroyAllWindows()
# <<< ADDED 끝 >>>

# <<< CHANGED: 엔트리포인트를 asyncio로 실행 >>>
if __name__ == "__main__":
    asyncio.run(_run_with_ws())
    