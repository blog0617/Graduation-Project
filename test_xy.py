import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Mediapipe 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 모델 로드
model_xy = load_model("fall_detection_xy.h5")

# 테스트 동영상
test_video = r"C:\Users\blog0\OneDrive\문서\졸작_grok\videos\fall_5.mp4"  # 실제 경로로 수정
cap = cv2.VideoCapture(test_video)
window_size = 30
xy_buffer = []

if not cap.isOpened():
    print(f"Error: Could not open video file {test_video}")
    exit()

print(f"Processing {test_video}. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 좌표 추출
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    if results.pose_landmarks:
        frame_landmarks = [[lm.x, lm.y] for lm in results.pose_landmarks.landmark]
        xy_buffer.append(frame_landmarks)
        
        if len(xy_buffer) == window_size:
            X_test = np.array([xy_buffer])  # (1, 30, 33, 2)
            X_test_flat = X_test.reshape(1, 30, 33 * 2)
            prediction = model_xy.predict(X_test_flat, verbose=0)[0][0]
            label = "Fall" if prediction > 0.5 else "Normal"
            
            cv2.putText(frame, f"Prediction: {label} ({prediction:.2f})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            xy_buffer.pop(0)
    
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('Fall Detection (x, y) - Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Video test completed.")