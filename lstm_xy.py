import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. 데이터 로드 및 Reshape
X_xy = np.load('X_xy.npy')
y_xy = np.load('y_xy.npy')

# 데이터 Reshape (4D → 3D)
X_xy = X_xy.reshape(X_xy.shape[0], X_xy.shape[1], -1)  # (20042, 30, 66)

# 2. 훈련/검증 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_xy, y_xy, test_size=0.2, stratify=y_xy, random_state=42)

# 3. LSTM 모델 정의
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 66)),  
    BatchNormalization(),
    Dropout(0.3),

    LSTM(32),
    BatchNormalization(),
    Dropout(0.3),

    Dense(16, activation='relu'),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

# 4. 모델 컴파일 및 학습 설정
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 5. 콜백 함수 설정 (EarlyStopping & Learning Rate 조정)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

# 6. 모델 학습
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)

# 7. 모델 저장
model.save('lstm_fall_detection.h5')
print("모델 저장 완료: lstm_fall_detection.h5")