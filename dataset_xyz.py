import numpy as np
import os

def load_xyz_data(folder_path, window_size=30):
    X, y = [], []
    # 폴더 내 모든 .npy 파일 읽기
    for file in os.listdir(folder_path):
        if file.endswith('_xyz.npy'):
            data = np.load(os.path.join(folder_path, file))
            # 라벨 설정
            label = 1 if 'fall' in file.lower() else 0
            # window_size 단위로 나누기
            for i in range(0, len(data) - window_size, window_size):
                X.append(data[i:i + window_size])
                y.append(label)
    return np.array(X), np.array(y)

# 데이터 로드
xyz_folder = "xyz_landmarks/"
X_xyz, y_xyz = load_xyz_data(xyz_folder)
print("X_xyz shape:", X_xyz.shape)  # (샘플 수, 30, 33, 3)
print("y_xyz shape:", y_xyz.shape)  # (샘플 수,)
np.save("X_xyz.npy", X_xyz)
np.save("y_xyz.npy", y_xyz)