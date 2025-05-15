# fall_detection_utils.py

import numpy as np

# 움직임과 낙하 감지 기준값 (필요시 조정)
DROP_THRESHOLD = 0.01          # 이전보다 y좌표 평균이 이만큼 내려가면 낙하로 간주
MOVEMENT_THRESHOLD = 0.01      # 좌표 변화 평균이 이 값보다 작으면 거의 안 움직인다고 판단
MOVEMENT_FRAMES = 5            # 몇 프레임을 기준으로 움직임 여부 판단할지

# Y좌표 기준 낙하 감지
def check_sudden_drop(buffer):
    if len(buffer) < 2:
        return False
    y_prev = np.mean([pt[1] for frame in buffer[:-1] for pt in frame])
    y_now = np.mean([pt[1] for pt in buffer[-1]])
    return (y_now - y_prev) > DROP_THRESHOLD

# 움직임 거의 없는지 확인
def check_low_movement(buffer):
    if len(buffer) < MOVEMENT_FRAMES:
        return False
    movement = calculate_movement(buffer[-MOVEMENT_FRAMES:])
    return movement < MOVEMENT_THRESHOLD

# 두 프레임 사이의 전체 관절 이동량 평균 계산
def calculate_movement(frames):
    diffs = []
    for i in range(1, len(frames)):
        prev = np.array(frames[i - 1])
        curr = np.array(frames[i])
        diff = np.linalg.norm(curr - prev, axis=1)
        diffs.append(np.mean(diff))
    return np.mean(diffs)

# 후처리: 모델 예측 + 조건 만족 시 fall 확정
def postprocess_label(raw_label, buffer, raw_score=None):
    if raw_score is not None and 0.4 <= raw_score <= 0.6:
        return "Unclear"
    
    if raw_label == "Fall":
        if check_sudden_drop(buffer) and check_low_movement(buffer):
            return "Fall"
        elif check_low_movement(buffer):
            return "Lying Still"
        else:
            return "Unclear"
    else:
        return "Normal"
