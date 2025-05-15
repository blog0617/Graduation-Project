import cv2
import mediapipe as mp  # 이 줄 추가
import numpy as np
import os

# Mediapipe 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 동영상 폴더와 출력 폴더 설정
folder_path = r"C:\Users\blog0\OneDrive\문서\졸작_grok\videos"  # 동영상 파일이 있는 폴더 경로
output_folder = "xyz_landmarks/"  # x, y, z 좌표를 저장할 폴더
os.makedirs(output_folder, exist_ok=True)

# mp4 파일 목록 가져오기
video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]

# 각 동영상에서 좌표 추출
for video_file in video_files:
    video_path = os.path.join(folder_path, video_file)
    cap = cv2.VideoCapture(video_path)
    xyz_landmarks = []
    
    print(f"Processing {video_file} for x, y, z coordinates...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGR -> RGB 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        if results.pose_landmarks:
            frame_landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
            xyz_landmarks.append(frame_landmarks)
        
        # 시각화 (선택 사항)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Mediapipe Pose (x, y, z)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    
    # numpy 배열로 저장
    xyz_landmarks = np.array(xyz_landmarks)
    print(f"{video_file} - Shape:", xyz_landmarks.shape)
    output_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_xyz.npy")
    np.save(output_path, xyz_landmarks)

cv2.destroyAllWindows()
print("x, y, z 좌표 추출 완료!")