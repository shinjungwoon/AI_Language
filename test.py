    # img = detector.findHolistic(img, draw=True)
    # # 왼손/오른손
    # _, r_hand = detector.findRighthandLandmark(img)
    # _, l_hand = detector.findLefthandLandmark(img)

    # if (r_hand is not None) or (l_hand is not None):
    #     # joint shape: (42, 2) — 왼손 0~20, 오른손 21~41 (없으면 0)
    #     joint = np.zeros((42, 2), dtype=np.float32)

    #     if l_hand is not None:
    #         for j, lm in enumerate(l_hand.landmark):
    #             if j < 21:
    #                 joint[j] = [lm.x, lm.y]

    #     if r_hand is not None:
    #         for j, lm in enumerate(r_hand.landmark):
    #             if j < 21:
    #                 joint[j + 21] = [lm.x, lm.y]

        
# if hasL or hasR:
#         # joint shape: (42, 2) — 왼손 0~20, 오른손 21~41 (없으면 0)
#         joint = np.zeros((42, 2), dtype=np.float32)

#         if hasL:
#             for j, lm in enumerate(l_hand.landmark):
#                 if j < 21:
#                     joint[j] = [lm.x, lm.y]          # 왼손 → 0..20

#         if hasR:
#             for j, lm in enumerate(r_hand.landmark):
#                 if j < 21:
#                     joint[j + 21] = [lm.x, lm.y]     # 오른손 → 21..41

#         # 🔧 핵심: 오른손만 잡힌 경우 → 모델이 보는 '왼손 슬롯(0..20)'로 복사
#         if (not hasL) and hasR:
#             joint[0:21] = joint[21:42]
#             joint[21:42] = 0.0  

#         # 벡터 정규화 (사용자 모듈)
#         vector, angle_label = Vector_Normalization(joint)  # 반환 형태: (42,2)->벡터 & 각도
#         d = np.concatenate([vector.flatten(), angle_label.flatten()])  # 위치 의존성 제거 특징

#         seq.append(d)
#         if len(seq) >= SEQ_LENGTH:
#             inp = np.expand_dims(np.array(seq[-SEQ_LENGTH:], dtype=np.float32), axis=0)
#             interpreter.set_tensor(input_details[0]['index'], inp)
#             interpreter.invoke()

#             y_pred = interpreter.get_tensor(output_details[0]['index'])[0]
#             i_pred = int(np.argmax(y_pred))
#             conf = float(y_pred[i_pred])
#             if conf >= CONF_TH:
#                 action = ACTIONS[i_pred]
#                 last_action_draw, last_conf = action, conf
#                 stable = stabilizer.push(action)
#                 if stable is not None:
#                     composer.feed(stable)
#             else:
#                 last_action_draw, last_conf = '', 0.0

#     if hasL or hasR:
#         # --- 손 1개 → 특징 벡터 만들기: 어느 손이든 왼손 슬롯(0..20)에 넣어 추출 ---
#         def _feat_from_hand(hand):
#             if hand is None:
#                 return None
#             joint = np.zeros((42, 2), dtype=np.float32)
#             for j, lm in enumerate(hand.landmark):
#                 if j < 21:
#                     joint[j] = [lm.x, lm.y]   # 0..20 채움
#             vector, angle_label = Vector_Normalization(joint)
#             return np.concatenate([vector.flatten(), angle_label.flatten()])

#         outs = []  # [(xpos, jamo)] 이번 프레임 확정 출력들

#         # ===== 왼손 추론 =====
#         fL = _feat_from_hand(l_hand)
#         if fL is not None:
#             seqL.append(fL)
#             if len(seqL) >= SEQ_LENGTH:
#                 inpL = np.expand_dims(np.array(seqL[-SEQ_LENGTH:], dtype=np.float32), axis=0)
#                 interpreter.set_tensor(input_details[0]['index'], inpL)
#                 interpreter.invoke()
#                 yL = interpreter.get_tensor(output_details[0]['index'])[0]
#                 iL = int(np.argmax(yL)); confL = float(yL[iL])
#                 if confL >= CONF_TH:
#                     actL = ACTIONS[iL]
#                     last_action_draw, last_conf = actL, confL  # 디버그 표시
#                     stL = stabL.push(actL)
#                     if stL is not None:
#                         xL = float(np.mean([lm.x for lm in l_hand.landmark[:21]])) if l_hand else 0.0
#                         outs.append((xL, stL))

#         # ===== 오른손 추론 =====
#         fR = _feat_from_hand(r_hand)
#         if fR is not None:
#             seqR.append(fR)
#             if len(seqR) >= SEQ_LENGTH:
#                 inpR = np.expand_dims(np.array(seqR[-SEQ_LENGTH:], dtype=np.float32), axis=0)
#                 interpreter.set_tensor(input_details[0]['index'], inpR)
#                 interpreter.invoke()
#                 yR = interpreter.get_tensor(output_details[0]['index'])[0]
#                 iR = int(np.argmax(yR)); confR = float(yR[iR])
#                 if confR >= CONF_TH:
#                     actR = ACTIONS[iR]
#                     last_action_draw, last_conf = actR, confR  # 디버그 표시
#                     stR = stabR.push(actR)
#                     if stR is not None:
#                         xR = float(np.mean([lm.x for lm in r_hand.landmark[:21]])) if r_hand else 1.0
#                         outs.append((xR, stR))

#         # ===== 같은 프레임에 둘 다 확정되면: 화면 왼쪽→오른쪽 순서로 송출 =====
#         if outs:
#             outs.sort(key=lambda t: t[0])  # x작은 것(왼쪽) 먼저
#             for _, jamo in outs:
#                 composer.feed(jamo)