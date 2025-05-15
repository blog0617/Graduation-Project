import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. 데이터 로드
X_xyz = np.load("X_xyz.npy")
y_xyz = np.load("y_xyz.npy")

# 2. 데이터 평탄화 (3D → 2D)
X_xyz_flat = X_xyz.reshape(X_xyz.shape[0], 30, 33 * 3)  # 33 * 3 = 99

# 3. 훈련/검증 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_xyz_flat, y_xyz, test_size=0.2, stratify=y_xyz, random_state=42)

# 4. 모델 설계
model_xyz = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 99)),  # 99 = 33 * 3
    BatchNormalization(),
    Dropout(0.3),

    LSTM(32),
    BatchNormalization(),
    Dropout(0.3),

    Dense(16, activation='relu'),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

# 5. 모델 컴파일 및 학습 설정
model_xyz.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 6. 콜백 함수 설정 (EarlyStopping & Learning Rate 조정)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

# 7. 모델 학습
history = model_xyz.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)

# 8. 모델 저장
model_xyz.save("fall_detection_xyz.h5")
print("모델 저장 완료: fall_detection_xyz.h5")
