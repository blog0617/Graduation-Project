import numpy as np

# 데이터 불러오기
X_xy = np.load("X_xy.npy")
y_xy = np.load("y_xy.npy")

# 데이터 크기 확인
print("X_xy shape:", X_xy.shape)  # 예상: (샘플 수, 30, 66)
print("y_xy shape:", y_xy.shape)  # 예상: (샘플 수,)

# 고유 클래스 확인
unique, counts = np.unique(y_xy, return_counts=True)
print("클래스 분포:", dict(zip(unique, counts)))  # 0과 1이 적절히 분포하는지 확인

# 샘플 데이터 확인
print("첫 번째 X 데이터:", X_xy[0])
print("첫 번째 y 데이터:", y_xy[0])
