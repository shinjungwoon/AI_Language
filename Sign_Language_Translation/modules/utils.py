import numpy as np

def Vector_Normalization(joint: np.ndarray):
    """
    joint: (42, 2) 또는 (21, 2) 중 '앞 21행'에 오른손 좌표가 들어있다고 가정 (x,y in [0,1])
    return:
      - vector: (20, 2)  각 세그먼트의 단위 방향벡터 (0벡터는 [0,0] 유지)
      - angle_label: (15,)  선택된 인접 세그먼트 페어의 각도(도), float32
    """
    joint = np.asarray(joint, dtype=np.float32)

    # 부모/자식 인덱스 (MediaPipe 손 21 포인트 기준)
    p_idx = [0,1,2,3,  0,5,6,7,  0,9,10,11,  0,13,14,15,  0,17,18,19]
    c_idx = [1,2,3,4,  5,6,7,8,  9,10,11,12, 13,14,15,16, 17,18,19,20]

    v1 = joint[p_idx, :2]   # (20,2)
    v2 = joint[c_idx, :2]   # (20,2)
    v  = v2 - v1            # (20,2)

    # 안전한 정규화
    eps = 1e-7
    n = np.linalg.norm(v, axis=1, keepdims=True)  # (20,1)
    safe = (n > eps).astype(np.float32)
    v_norm = np.zeros_like(v, dtype=np.float32)
    v_norm[safe[:,0] == 1] = v[safe[:,0] == 1] / n[safe[:,0] == 1]

    # 각도 계산에 사용할 페어(총 15개)
    a_idx = [0,1,2,  4,5,6,  8,9,10,  12,13,14,  16,17,18]
    b_idx = [1,2,3,  5,6,7,  9,10,11, 13,14,15,  17,18,19]

    a = v_norm[a_idx]   # (15,2)
    b = v_norm[b_idx]   # (15,2)

    # 내적 → arccos (clip으로 NaN 방지)
    dot = np.einsum('ij,ij->i', a, b)            # (15,)
    dot = np.clip(dot, -1.0, 1.0)
    angle = np.degrees(np.arccos(dot)).astype(np.float32)  # (15,)

    return v_norm, angle  # (20,2), (15,)
