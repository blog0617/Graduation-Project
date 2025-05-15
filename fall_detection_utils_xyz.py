# fall_detection_utils_xyz.py

import numpy as np

# 기준값 (조정 가능)
DROP_THRESHOLD = 0.01
MOVEMENT_THRESHOLD = 0.01
MOVEMENT_FRAMES = 5

# z좌표 포함한 평균 이동량 계산
def calculate_movement(frames):
    diffs = []
    for i in range(1, len(frames)):
        prev = np.array(frames[i - 1])
        curr = np.array(frames[i])
        diff = np.linalg.norm(curr - prev, axis=1)
        diffs.append(np.mean(diff))
    return np.mean(diffs)

# Y좌표 평균 낙하 감지 (z는 제외)
def check_sudden_drop(buffer):
    if len(buffer) < 2:
        return False
    y_prev = np.mean([pt[1] for frame in buffer[:-1] for pt in frame])
    y_now = np.mean([pt[1] for pt in buffer[-1]])
    return (y_now - y_prev) > DROP_THRESHOLD

# 움직임 작음 여부
def check_low_movement(buffer):
    if len(buffer) < MOVEMENT_FRAMES:
        return False
    movement = calculate_movement(buffer[-MOVEMENT_FRAMES:])
    return movement < MOVEMENT_THRESHOLD

# 최종 라벨 처리
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
