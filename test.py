#     # img = detector.findHolistic(img, draw=True)
#     # # 왼손/오른손
#     # _, r_hand = detector.findRighthandLandmark(img)
#     # _, l_hand = detector.findLefthandLandmark(img)

#     # if (r_hand is not None) or (l_hand is not None):
#     #     # joint shape: (42, 2) — 왼손 0~20, 오른손 21~41 (없으면 0)
#     #     joint = np.zeros((42, 2), dtype=np.float32)

#     #     if l_hand is not None:
#     #         for j, lm in enumerate(l_hand.landmark):
#     #             if j < 21:
#     #                 joint[j] = [lm.x, lm.y]

#     #     if r_hand is not None:
#     #         for j, lm in enumerate(r_hand.landmark):
#     #             if j < 21:
#     #                 joint[j + 21] = [lm.x, lm.y]

        
# # if hasL or hasR:
# #         # joint shape: (42, 2) — 왼손 0~20, 오른손 21~41 (없으면 0)
# #         joint = np.zeros((42, 2), dtype=np.float32)

# #         if hasL:
# #             for j, lm in enumerate(l_hand.landmark):
# #                 if j < 21:
# #                     joint[j] = [lm.x, lm.y]          # 왼손 → 0..20

# #         if hasR:
# #             for j, lm in enumerate(r_hand.landmark):
# #                 if j < 21:
# #                     joint[j + 21] = [lm.x, lm.y]     # 오른손 → 21..41

# #         # 🔧 핵심: 오른손만 잡힌 경우 → 모델이 보는 '왼손 슬롯(0..20)'로 복사
# #         if (not hasL) and hasR:
# #             joint[0:21] = joint[21:42]
# #             joint[21:42] = 0.0  

# #         # 벡터 정규화 (사용자 모듈)
# #         vector, angle_label = Vector_Normalization(joint)  # 반환 형태: (42,2)->벡터 & 각도
# #         d = np.concatenate([vector.flatten(), angle_label.flatten()])  # 위치 의존성 제거 특징

# #         seq.append(d)
# #         if len(seq) >= SEQ_LENGTH:
# #             inp = np.expand_dims(np.array(seq[-SEQ_LENGTH:], dtype=np.float32), axis=0)
# #             interpreter.set_tensor(input_details[0]['index'], inp)
# #             interpreter.invoke()

# #             y_pred = interpreter.get_tensor(output_details[0]['index'])[0]
# #             i_pred = int(np.argmax(y_pred))
# #             conf = float(y_pred[i_pred])
# #             if conf >= CONF_TH:
# #                 action = ACTIONS[i_pred]
# #                 last_action_draw, last_conf = action, conf
# #                 stable = stabilizer.push(action)
# #                 if stable is not None:
# #                     composer.feed(stable)
# #             else:
# #                 last_action_draw, last_conf = '', 0.0

# #     if hasL or hasR:
# #         # --- 손 1개 → 특징 벡터 만들기: 어느 손이든 왼손 슬롯(0..20)에 넣어 추출 ---
# #         def _feat_from_hand(hand):
# #             if hand is None:
# #                 return None
# #             joint = np.zeros((42, 2), dtype=np.float32)
# #             for j, lm in enumerate(hand.landmark):
# #                 if j < 21:
# #                     joint[j] = [lm.x, lm.y]   # 0..20 채움
# #             vector, angle_label = Vector_Normalization(joint)
# #             return np.concatenate([vector.flatten(), angle_label.flatten()])

# #         outs = []  # [(xpos, jamo)] 이번 프레임 확정 출력들

# #         # ===== 왼손 추론 =====
# #         fL = _feat_from_hand(l_hand)
# #         if fL is not None:
# #             seqL.append(fL)
# #             if len(seqL) >= SEQ_LENGTH:
# #                 inpL = np.expand_dims(np.array(seqL[-SEQ_LENGTH:], dtype=np.float32), axis=0)
# #                 interpreter.set_tensor(input_details[0]['index'], inpL)
# #                 interpreter.invoke()
# #                 yL = interpreter.get_tensor(output_details[0]['index'])[0]
# #                 iL = int(np.argmax(yL)); confL = float(yL[iL])
# #                 if confL >= CONF_TH:
# #                     actL = ACTIONS[iL]
# #                     last_action_draw, last_conf = actL, confL  # 디버그 표시
# #                     stL = stabL.push(actL)
# #                     if stL is not None:
# #                         xL = float(np.mean([lm.x for lm in l_hand.landmark[:21]])) if l_hand else 0.0
# #                         outs.append((xL, stL))

# #         # ===== 오른손 추론 =====
# #         fR = _feat_from_hand(r_hand)
# #         if fR is not None:
# #             seqR.append(fR)
# #             if len(seqR) >= SEQ_LENGTH:
# #                 inpR = np.expand_dims(np.array(seqR[-SEQ_LENGTH:], dtype=np.float32), axis=0)
# #                 interpreter.set_tensor(input_details[0]['index'], inpR)
# #                 interpreter.invoke()
# #                 yR = interpreter.get_tensor(output_details[0]['index'])[0]
# #                 iR = int(np.argmax(yR)); confR = float(yR[iR])
# #                 if confR >= CONF_TH:
# #                     actR = ACTIONS[iR]
# #                     last_action_draw, last_conf = actR, confR  # 디버그 표시
# #                     stR = stabR.push(actR)
# #                     if stR is not None:
# #                         xR = float(np.mean([lm.x for lm in r_hand.landmark[:21]])) if r_hand else 1.0
# #                         outs.append((xR, stR))

# #         # ===== 같은 프레임에 둘 다 확정되면: 화면 왼쪽→오른쪽 순서로 송출 =====
# #         if outs:
# #             outs.sort(key=lambda t: t[0])  # x작은 것(왼쪽) 먼저
# #             for _, jamo in outs:
# #                 composer.feed(jamo)
# ------------------------------------------------------
# import sys
# # sys.path.append('pingpong')
# # from pingpong.pingpongthread import PingPongThread
# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# import modules.holistic_module as hm
# from tensorflow.keras.models import load_model
# import math, time
# from modules.utils import Vector_Normalization
# from PIL import ImageFont, ImageDraw, Image
# from collections import deque
# from unicode import join_jamos

# PRINT_ON_EMIT = True  # 자모 확정될 때마다 좌표 출력

# def _dump_landmarks(hand):
#     """MediaPipe hand 객체에서 21개 (x,y,z) 튜플 리스트로 추출(정규화 좌표)."""
#     if hand is None:
#         return []
#     out = []
#     for lm in hand.landmark[:21]:
#         x = float(lm.x)
#         y = float(lm.y)
#         z = float(getattr(lm, "z", 0.0))
#         out.append((x, y, z))
#     return out

# def _print_emit(hand_side, jamo, conf, hand):
#     """자모가 확정될 때 좌표 로그 출력."""
#     pts = _dump_landmarks(hand)
#     if not PRINT_ON_EMIT:
#         return
#     if pts:
#         cx = sum(p[0] for p in pts) / len(pts)
#         cy = sum(p[1] for p in pts) / len(pts)
#         print(f"[EMIT] {hand_side} -> '{jamo}' (conf={conf:.3f}) center=({cx:.4f}, {cy:.4f})")
#         for idx, (x, y, z) in enumerate(pts):
#             print(f"  {hand_side}[{idx:02d}] x={x:.4f}  y={y:.4f}  z={z:.4f}")
#     else:
#         print(f"[EMIT] {hand_side} -> '{jamo}' (conf={conf:.3f}) (no landmarks)")

# BIMANUAL_ORDER = "left_to_right"

# # === UI: 윈도우/버튼 설정 & 마우스 콜백 ===
# WINDOW_NAME = 'Sign→Korean Compose'

# # 버튼 위치 (x1, y1, x2, y2)
# BTN_CLEAR   = (10,  90, 120, 125)   # "지우기"
# BTN_NEWLINE = (130, 90, 260, 125)   # "줄바꿈"

# _trigger_clear = False
# _trigger_newline = False

# def _point_in_rect(x, y, rect):
#     x1, y1, x2, y2 = rect
#     return x1 <= x <= x2 and y1 <= y <= y2

# def _on_mouse(event, x, y, flags, param):
#     global _trigger_clear, _trigger_newline
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if _point_in_rect(x, y, BTN_CLEAR):
#             _trigger_clear = True
#         elif _point_in_rect(x, y, BTN_NEWLINE):
#             _trigger_newline = True
            
# # =========================
# # 설정: 폰트 & 액션(자모 레이블)
# # =========================
# FONT_PATH = "fonts/HMKMMAG.TTF"  # 없으면 자동 폴백
# FONT_SIZE = 40

# ACTIONS = [
#     'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
#     'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
#     'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ'
# ]
# SEQ_LENGTH = 10
# CONF_TH = 0.90               # 신뢰도 임계치
# STABLE_COUNT = 2           # 같은 자모가 연속 몇 번 나와야 확정할지
# # 클래스별 상향 임계치 / 안정화 길이 (헷갈리는 자모 강조)
# PER_CLASS_CONF = {'ㅅ': 0.90, 'ㅈ': 0.90}
# PER_CLASS_STABLE = {'ㅅ': 3 , 'ㅈ': 3}
# SYL_TIMEOUT = 5           # 초/중/종성 조합 후 이 시간(초) 입력 없으면 음절 확정
# WORD_TIMEOUT = 5          # 더 오래 입력 없으면 띄어쓰기
# SHOW_MAX_CHARS = 40          # 화면에 보여줄 최대 글자수
# # ===== 추가: 흔들림/중복 방지 튜닝 =====
# HOLD_MIN_FR = 5              # 손이 일정히 고정된 프레임 수(최소)
# COOLDOWN_FR = 6              # 같은 글자 확정 후 잠깐 쉬기
# MOVE_TH = 0.010              # 손 중심 이동 임계치(정규화 좌표)
# EMA_MOMENTUM = 0.7           # 확률 지수이동평균(0.6~0.8 권장)

# # =========================
# # 한글 합성 유틸
# # =========================
# CHO_LIST = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
# JUNG_LIST = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
# JONG_LIST = ['', 'ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

# VOWELS = set(['ㅏ','ㅑ','ㅓ','ㅕ','ㅗ','ㅛ','ㅜ','ㅠ','ㅡ','ㅣ','ㅐ','ㅒ','ㅔ','ㅖ','ㅢ','ㅚ','ㅟ'])
# CONSONANTS = set([a for a in ACTIONS if a not in VOWELS])

# # 겹초성(쌍자음) 규칙: 같은 자음 반복 시 초성 쌍자음으로
# CHO_DOUBLE = {('ㄱ','ㄱ'):'ㄲ', ('ㄷ','ㄷ'):'ㄸ', ('ㅂ','ㅂ'):'ㅃ', ('ㅅ','ㅅ'):'ㅆ', ('ㅈ','ㅈ'):'ㅉ'}
# # 겹받침 규칙
# JONG_DOUBLE = {
#     ('ㄱ','ㅅ'):'ㄳ', ('ㄴ','ㅈ'):'ㄵ', ('ㄴ','ㅎ'):'ㄶ',
#     ('ㄹ','ㄱ'):'ㄺ', ('ㄹ','ㅁ'):'ㄻ', ('ㄹ','ㅂ'):'ㄼ',
#     ('ㄹ','ㅅ'):'ㄽ', ('ㄹ','ㅌ'):'ㄾ', ('ㄹ','ㅍ'):'ㄿ', ('ㄹ','ㅎ'):'ㅀ',
#     ('ㅂ','ㅅ'):'ㅄ'
# }
# # 복모음 규칙
# JUNG_COMPOSE = {
#     ('ㅗ','ㅏ'):'ㅘ', ('ㅗ','ㅐ'):'ㅙ', ('ㅗ','ㅣ'):'ㅚ',
#     ('ㅜ','ㅓ'):'ㅝ', ('ㅜ','ㅔ'):'ㅞ', ('ㅜ','ㅣ'):'ㅟ',
#     ('ㅡ','ㅣ'):'ㅢ'
# }

# def _idx_or_none(lst, x):
#     try:
#         return lst.index(x)
#     except ValueError:
#         return None

# def compose_syllable(cho, jung, jong=''):
#     """초/중/(종)으로 한 글자 합성"""
#     ic = _idx_or_none(CHO_LIST, cho)
#     iv = _idx_or_none(JUNG_LIST, jung)
#     if ic is None or iv is None:
#         return (cho or '') + (jung or '') + (jong or '')
#     ij = _idx_or_none(JONG_LIST, jong)
#     if ij is None: ij = 0
#     return chr(0xAC00 + (ic * 21 + iv) * 28 + ij)

# class HangulComposer:
#     def __init__(self, syl_timeout=SYL_TIMEOUT, word_timeout=WORD_TIMEOUT):
#         self.reset_block()
#         self.text = ""
#         self.last_input_t = time.time()
#         self.syl_timeout = syl_timeout
#         self.word_timeout = word_timeout

#     def reset_block(self):
#         self.cho = None
#         self.jung = None
#         self.jong = None
#         self._last_jamo = None

#     def _try_compose_jung(self, base, add):
#         return JUNG_COMPOSE.get((base, add))

#     def _try_compose_cho_double(self, base, add):
#         return CHO_DOUBLE.get((base, add))

#     def _try_compose_jong_double(self, base, add):
#         return JONG_DOUBLE.get((base, add))

#     def feed(self, jamo):
#         now = time.time()
#         self.last_input_t = now

#         # 1) 아직 중성이 없을 때
#         if self.jung is None:
#             if jamo in CONSONANTS:
#                 if self.cho is None:
#                     # 초성 시작
#                     self.cho = jamo
#                 else:
#                     # 같은 자음 두 번 → 쌍자음 시도
#                     dbl = self._try_compose_cho_double(self.cho, jamo)
#                     if dbl:
#                         self.cho = dbl
#                     else:
#                         # 새 초성 시작(이전 블럭은 초성만이라 글자 아님 → 그냥 텍스트로 추가)
#                         self._commit_if_partial()
#                         self.cho = jamo
#             else:  # 모음
#                 if self.cho is None:
#                     # 초성 없이 모음만 → 임시로 ㅇ 초성
#                     self.cho = 'ㅇ'
#                 self.jung = jamo
#             return

#         # 2) 중성은 있고 종성은 아직
#         if self.jong is None:
#             if jamo in VOWELS:
#                 # 복모음 시도
#                 new_v = self._try_compose_jung(self.jung, jamo)
#                 if new_v:
#                     self.jung = new_v
#                 else:
#                     # 새 모음 → 이전 음절 확정하고 이번은 새 음절 시작(초성=ㅇ 가정)
#                     self.commit_block()
#                     self.cho, self.jung = 'ㅇ', jamo
#             else:
#                 # 종성 후보
#                 self.jong = jamo
#             return

#         # 3) 종성이 있는 상태
#         if jamo in CONSONANTS:
#             # 겹받침 시도
#             new_jong = self._try_compose_jong_double(self.jong, jamo)
#             if new_jong:
#                 self.jong = new_jong
#             else:
#                 # 겹받침 불가 → 이전 음절 확정, 새 음절 초성 찍고 대기
#                 self.commit_block()
#                 self.cho = jamo
#         else:
#             # 모음이 오면 이전 종성은 다음 음절의 초성으로 이동
#             carry_cho = self.jong
#             self.jong = None
#             self.commit_block()
#             # 종성 한글(겹받침 포함)에서 초성 가능한 단위만 남김(겹받침이면 앞 요소를 초성으로 가정)
#             if carry_cho in JONG_LIST:
#                 # 종성→초성 맵핑(동일 문자 대부분 호환됨)
#                 self.cho = carry_cho if carry_cho != '' else 'ㅇ'
#             else:
#                 self.cho = 'ㅇ'
#             self.jung = jamo

#     def _commit_if_partial(self):
#         # 초성 단독 등은 그대로 텍스트에 추가
#         if self.cho and not self.jung and not self.jong:
#             self.text += self.cho
#         elif self.cho and self.jung:
#             # 종성 없이도 한 글자
#             self.text += compose_syllable(self.cho, self.jung)
#         self.reset_block()

#     def commit_block(self):
#         if self.cho and self.jung:
#             ch = compose_syllable(self.cho, self.jung, self.jong or '')
#             self.text += ch
#         else:
#             # 불완전 블럭은 있는 그대로
#             if self.cho: self.text += self.cho
#             if self.jung: self.text += self.jung
#             if self.jong: self.text += self.jong
#         self.reset_block()

#     def maybe_timeout(self):
#         now = time.time()
#         # 음절 확정
#         if (self.cho or self.jung or self.jong) and (now - self.last_input_t >= self.syl_timeout):
#             self.commit_block()
#         # 띄어쓰기
#         if (now - self.last_input_t >= self.word_timeout):
#             if len(self.text) > 0 and not self.text.endswith(' '):
#                 self.text += ' '

#     def backspace(self):
#         # 블럭 지우기 우선
#         if self.cho or self.jung or self.jong:
#             self.reset_block()
#             return
#         if len(self.text) > 0:
#             self.text = self.text[:-1]

#     def get_preview_block(self):
#         if self.cho and self.jung:
#             return compose_syllable(self.cho, self.jung, self.jong or '')
#         # 미완성은 자모 그대로 보여줌
#         return (self.cho or '') + (self.jung or '') + (self.jong or '')

#     def get_text(self):
#         return self.text

# # =========================
# # 예측 디바운서
# # =========================
# class JamoStabilizer:
#     def __init__(self, stable_count=STABLE_COUNT):
#         self.buf = deque(maxlen=stable_count)
#         self.last_emitted = None

#     def push(self, jamo):
#         self.buf.append(jamo)
#         if len(self.buf) == self.buf.maxlen and len(set(self.buf)) == 1:
#             if self.last_emitted != jamo:
#                 self.last_emitted = jamo
#                 return jamo
#         return None
    
# seqL, seqR = [], []
# stabL = JamoStabilizer(STABLE_COUNT)
# stabR = JamoStabilizer(STABLE_COUNT)

# # =========================
# # 폰트 준비
# # =========================
# def get_font(path=FONT_PATH, size=FONT_SIZE):
#     try:
#         return ImageFont.truetype(path, size)
#     except Exception:
#         # 폴백: PIL 기본 폰트(한글 글리프 없을 수 있음)
#         return ImageFont.load_default()

# font = get_font()

# # =========================
# # MediaPipe & TFLite 로딩
# # =========================
# detector = hm.HolisticDetector(min_detection_confidence=0.3)

# from pathlib import Path
# BASE_DIR = Path(__file__).resolve().parent          # ...\Sign_Language_Translation
# MODEL_PATH = str((BASE_DIR.parent / "models" / "multi_hand_gesture_classifier.tflite").resolve())

# interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
# interpreter.allocate_tensors()

# # interpreter = tf.lite.Interpreter(model_path="models/multi_hand_gesture_classifier.tflite")
# # interpreter.allocate_tensors()


# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # =========================
# # 실행
# # =========================
# cap = cv2.VideoCapture(0)
# cv2.namedWindow(WINDOW_NAME)
# cv2.setMouseCallback(WINDOW_NAME, _on_mouse)

# seq = []
# stabilizer = JamoStabilizer(STABLE_COUNT)
# composer = HangulComposer(SYL_TIMEOUT, WORD_TIMEOUT)

# last_action_draw = ''
# last_conf = 0.0

# # ===== 추가: 스무딩/홀드/쿨다운/이동량 상태 =====
# y_smooth_L = None
# y_smooth_R = None
# last_center = None
# hold_frames = 0
# cooldownL = 0
# cooldownR = 0

# while cap.isOpened():
#     ret, img = cap.read()
#     if not ret:
#         break

#     img = detector.findHolistic(img, draw=True)
#     # 왼손/오른손
#     _, r_hand = detector.findRighthandLandmark(img)
#     _, l_hand = detector.findLefthandLandmark(img)

#     hasL = l_hand is not None
#     hasR = r_hand is not None

#     if hasL or hasR:
#     # --- 손 1개 → 특징 벡터 만들기(0..20 슬롯 사용) ---
#         def _feat_from_hand(hand):
#             if hand is None:
#                 return None
#             joint = np.zeros((42, 2), dtype=np.float32)
#             for j, lm in enumerate(hand.landmark):
#                 if j < 21:
#                     joint[j] = [lm.x, lm.y]
#             vector, angle_label = Vector_Normalization(joint)
#             return np.concatenate([vector.flatten(), angle_label.flatten()])

#         # --- 손 중심 이동(흔들림) 측정 → 흔들리면 버퍼 리셋 ---
#         def _center_xy(hand):
#             xs = [lm.x for lm in hand.landmark[:21]]
#             ys = [lm.y for lm in hand.landmark[:21]]
#             return float(np.mean(xs)), float(np.mean(ys))

#         centers = []
#         if hasL: centers.append(_center_xy(l_hand))
#         if hasR: centers.append(_center_xy(r_hand))
#         center_now = None if not centers else tuple(np.mean(centers, axis=0))

#         if center_now is not None:
#             if last_center is not None:
#                 dx = center_now[0] - last_center[0]
#                 dy = center_now[1] - last_center[1]
#                 dist = (dx*dx + dy*dy)**0.5
#                 if dist > MOVE_TH:
#                     hold_frames = 0
#                     stabL.buf.clear(); stabR.buf.clear()
#                 else:
#                     hold_frames += 1
#             else:
#                 hold_frames = 1
#             last_center = center_now

#         outs = []  # [(xpos, jamo)]

#         # ===== 왼손 추론 (EMA + 홀드 + 쿨다운) =====
#         fL = _feat_from_hand(l_hand)
#         if fL is not None:
#             seqL.append(fL)
#             if len(seqL) >= SEQ_LENGTH:
#                 inpL = np.expand_dims(np.array(seqL[-SEQ_LENGTH:], dtype=np.float32), axis=0)
#                 interpreter.set_tensor(input_details[0]['index'], inpL)
#                 interpreter.invoke()
#                 yL_raw = interpreter.get_tensor(output_details[0]['index'])[0]
#                 # 확률 스무딩(EMA)
#                 if y_smooth_L is None:
#                     y_smooth_L = yL_raw.copy()
#                 else:
#                     y_smooth_L = EMA_MOMENTUM * y_smooth_L + (1.0 - EMA_MOMENTUM) * yL_raw
#                 iL = int(np.argmax(y_smooth_L)); confL = float(y_smooth_L[iL])
#                 last_action_draw, last_conf = ACTIONS[iL], confL  # 디버그 표시

#                 actL = ACTIONS[iL]
#                 # 클래스별 임계/안정화 길이 반영
#                 thL = PER_CLASS_CONF.get(actL, CONF_TH)
#                 stabL.buf = deque(maxlen=PER_CLASS_STABLE.get(actL, STABLE_COUNT))

#                 # 형상 필터: ㅅ/ㅈ은 모양이 맞는지 추가 검증
#                 geom_ok = True
#                 if actL == 'ㅅ' and l_hand is not None:
#                     geom_ok = globals().get('_is_likely_S', lambda h: True)(l_hand)
#                 elif actL == 'ㅈ' and l_hand is not None:
#                     geom_ok = globals().get('_is_likely_J', lambda h: True)(l_hand)

#                 if (confL >= thL) and geom_ok and (hold_frames >= HOLD_MIN_FR) and (cooldownL == 0):
#                     stL = stabL.push(actL)
#                     if stL is not None:
#                         _print_emit("L", stL, confL, l_hand)
#                         xL = float(np.mean([lm.x for lm in l_hand.landmark[:21]])) if l_hand else 0.0
#                         outs.append((xL, stL))
#                         cooldownL = COOLDOWN_FR
#                 # if confL >= CONF_TH and hold_frames >= HOLD_MIN_FR and cooldownL == 0:
#                 #     actL = ACTIONS[iL]
#                 #     stL = stabL.push(actL)
#                 #     if stL is not None:
#                 #         # ↓↓↓ 추가: 좌표 로그 출력
#                 #         _print_emit("L", stL, confL, l_hand)

#                 #         xL = float(np.mean([lm.x for lm in l_hand.landmark[:21]])) if l_hand else 0.0
#                 #         outs.append((xL, stL))
#                 #         cooldownL = COOLDOWN_FR
#                 # if confL >= CONF_TH and hold_frames >= HOLD_MIN_FR and cooldownL == 0:
#                 #     actL = ACTIONS[iL]
#                 #     stL = stabL.push(actL)
#                 #     if stL is not None:
#                 #         xL = float(np.mean([lm.x for lm in l_hand.landmark[:21]])) if l_hand else 0.0
#                 #         outs.append((xL, stL))
#                 #         cooldownL = COOLDOWN_FR

#         # ===== 오른손 추론 (EMA + 홀드 + 쿨다운) =====
#         fR = _feat_from_hand(r_hand)
#         if fR is not None:
#             seqR.append(fR)
#             if len(seqR) >= SEQ_LENGTH:
#                 inpR = np.expand_dims(np.array(seqR[-SEQ_LENGTH:], dtype=np.float32), axis=0)
#                 interpreter.set_tensor(input_details[0]['index'], inpR)
#                 interpreter.invoke()
#                 yR_raw = interpreter.get_tensor(output_details[0]['index'])[0]
#                 if y_smooth_R is None:
#                     y_smooth_R = yR_raw.copy()
#                 else:
#                     y_smooth_R = EMA_MOMENTUM * y_smooth_R + (1.0 - EMA_MOMENTUM) * yR_raw
#                 iR = int(np.argmax(y_smooth_R)); confR = float(y_smooth_R[iR])
#                 last_action_draw, last_conf = ACTIONS[iR], confR
                
#                 if confR >= CONF_TH and hold_frames >= HOLD_MIN_FR and cooldownR == 0:
#                     actR = ACTIONS[iR]
#                     stR = stabR.push(actR)
#                     if stR is not None:
#                         # ↓↓↓ 추가: 좌표 로그 출력
#                         _print_emit("R", stR, confR, r_hand)

#                         xR = float(np.mean([lm.x for lm in r_hand.landmark[:21]])) if r_hand else 1.0
#                         outs.append((xR, stR))
#                         cooldownR = COOLDOWN_FR
#                 # if confR >= CONF_TH and hold_frames >= HOLD_MIN_FR and cooldownR == 0:
#                 #     actR = ACTIONS[iR]
#                 #     stR = stabR.push(actR)
#                 #     if stR is not None:
#                 #         xR = float(np.mean([lm.x for lm in r_hand.landmark[:21]])) if r_hand else 1.0
#                 #         outs.append((xR, stR))
#                 #         cooldownR = COOLDOWN_FR

#         # --- 쿨다운 감소 ---
#         if cooldownL > 0: cooldownL -= 1
#         if cooldownR > 0: cooldownR -= 1

#         # ===== 이번 프레임 확정 출력 적용 =====
#         if outs:
#             outs.sort(key=lambda t: t[0])  # 왼쪽→오른쪽
#             for _, jamo in outs:
#                 composer.feed(jamo)

#     # 타임아웃 체크(음절 확정 & 자동 띄어쓰기)
#     composer.maybe_timeout()

#     # =========================
#     # 오버레이: 텍스트 그리기
#     # =========================
#     # preview_block = composer.get_preview_block()
#     # text_line = (composer.get_text() + preview_block)[-SHOW_MAX_CHARS:]
#     img_pil = Image.fromarray(img)
#     draw = ImageDraw.Draw(img_pil)
    
#     fps_ema = locals().get('fps_ema', 0.0) 
#     frame_idx = locals().get('frame_idx', 0)  # 없으면 0
#     draw.text((img.shape[1]-220, 10), f"FPS {fps_ema:.1f} | F {frame_idx}", font=font, fill=(180,180,180))
#     preview_block = composer.get_preview_block()
#     joined_text = join_jamos(composer.get_text() + preview_block)   # 자모 → 음절 조합
#     text_line = joined_text[-SHOW_MAX_CHARS:]
    
#     lines = joined_text.splitlines() or [joined_text]
#     to_draw = '\n'.join(lines[-3:])

#     # img_pil = Image.fromarray(img)
#     # draw = ImageDraw.Draw(img_pil)

#     # 현재 문장(프리뷰 포함)
#     draw.rectangle([(0,0),(img.shape[1],30)], fill=(0,0,0,128))
#     draw.text((10, 10), f"{text_line}", font=font, fill=(255,255,255), spacing=6)

#     # 디버그: 마지막 확신 자모/신뢰도
#     if last_action_draw:
#         draw.text((10, 45), f"Pred: {last_action_draw} ({last_conf:.2f})", font=font, fill=(90,90,90))
    
#     draw.rectangle([(BTN_CLEAR[0], BTN_CLEAR[1]), (BTN_CLEAR[2], BTN_CLEAR[3])], fill=(30,30,30))
#     draw.text((BTN_CLEAR[0]+12, BTN_CLEAR[1]+4), "지우기", font=font, fill=(255,255,255))

#     draw.rectangle([(BTN_NEWLINE[0], BTN_NEWLINE[1]), (BTN_NEWLINE[2], BTN_NEWLINE[3])], fill=(30,30,30))
#     draw.text((BTN_NEWLINE[0]+12, BTN_NEWLINE[1]+4), "줄바꿈", font=font, fill=(255,255,255))

#     img = np.array(img_pil)

#     cv2.imshow('Sign→Korean Compose', img)
#     key = cv2.waitKey(1) & 0xFF
#     # --- 단축키 ---
#     if key == 27:  # ESC
#         break
#     elif key == 8:  # Backspace
#         stabilizer.last_emitted = None
#         composer.backspace()
#     elif key == 32:  # Space -> 공백 커밋
#         composer.commit_block()
#         if not composer.get_text().endswith(' '):
#             composer.text += ' '
#     elif key in (ord('c'), ord('C')):  # Clear
#         stabilizer.last_emitted = None
#         composer.reset_block()
#         composer.text = ""
#     elif key in (ord('n'), ord('N'), 13):  # Newline (Enter 포함)
#         composer.commit_block()
#         composer.text += '\n'

#     # --- 마우스 버튼 트리거 ---
#     if _trigger_clear:
#         stabilizer.last_emitted = None
#         composer.reset_block()
#         composer.text = ""
#         _trigger_clear = False

#     if _trigger_newline:
#         composer.commit_block()
#         composer.text += '\n'
#         _trigger_newline = False

