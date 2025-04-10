import streamlit as st
import numpy as np
import pandas as pd
import joblib
import ast
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

# ─────────────────────────────────────────────
# 데이터 로딩
# ─────────────────────────────────────────────
train = np.load("dataset/data/data/train/P-3.npy")[:, 0].reshape(-1, 1)
test = np.load("dataset/data/data/test/P-3.npy")[:, 0].reshape(-1, 1)
labels = pd.read_csv("dataset/labeled_anomalies.csv")
channel_id = "P-3"
SEQ_LEN = 50

# 시퀀스 함수
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# 모델 및 데이터 불러오기 or 학습
if os.path.exists("saved_model/lstm_model_P3.h5"):
    model = load_model("saved_model/lstm_model_P3.h5", compile=False)
    data = joblib.load("saved_model/lstm_data_P3.pkl")
    X_test = data["X_test"]
    y_test = data["y_test"]
    errors = data["errors"]
    threshold = data["threshold"]
    scaler = data["scaler"]
else:
    # 정규화
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    # 시퀀스 생성
    X_train, y_train = create_sequences(train_scaled, SEQ_LEN)
    X_test, y_test = create_sequences(test_scaled, SEQ_LEN)

    # 모델 학습
    model = Sequential()
    model.add(LSTM(64, input_shape=(SEQ_LEN, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    # 오차 계산
    y_pred = model.predict(X_test)
    errors = np.abs(y_pred - y_test).flatten()
    train_pred = model.predict(X_train)
    train_errors = np.abs(train_pred - y_train).flatten()
    threshold = np.mean(train_errors) + 3 * np.std(train_errors)

    # 저장
    os.makedirs("saved_model", exist_ok=True)
    model.save("saved_model/lstm_model_P3.h5")
    joblib.dump({
        "X_test": X_test,
        "y_test": y_test,
        "errors": errors,
        "threshold": threshold,
        "scaler": scaler
    }, "saved_model/lstm_data_P3.pkl")

# 실제 라벨
anomaly_ranges = ast.literal_eval(labels[labels["chan_id"] == channel_id]["anomaly_sequences"].values[0])

# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────
st.title('이상치 판단 모델')
st.write('이상치를 판단하고 실제 결과와 비교해보세요!')

def check_point_status(index, errors, threshold, anomaly_ranges, seq_len=50, verbose=True):

    adjusted_index = index - seq_len

    # 범위 유효성 검사
    if adjusted_index < 0 or adjusted_index >= len(errors):
        if verbose:
            print(f" 검사 시점 {index}은 유효하지 않습니다.")
        return None, None

    # 판단
    model_is_anomaly = errors[adjusted_index] > threshold
    label_is_anomaly = any(start <= index <= end for (start, end) in anomaly_ranges)

    # 출력
    if verbose:
        print(f"▶ 검사 시점: {index} (SEQ_LEN: {seq_len})")
        print(f" - 모델 판단: {' 이상치' if model_is_anomaly else ' 정상'}")
        print(f" - 실제 라벨: {' 이상치' if label_is_anomaly else ' 정상'}")
        print(f" - 일치 여부 : {' 정답' if label_is_anomaly == model_is_anomaly else ' 오답'}")

    return model_is_anomaly, label_is_anomaly
# 사용자 입력: 0.0 ~ 1.0 사이 시간값
time = st.number_input('time(s)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# 시점 계산 (0.0 ~ 1.0 → 실제 시계열 인덱스)
total_len = len(test)
point_index = int(time * total_len)

model_anom, label_anom = check_point_status(point_index, errors, threshold, anomaly_ranges, seq_len=SEQ_LEN, verbose=False)

st.markdown(f"""
### 결과 (검사 시점: {point_index})
- 모델 판단: {' 이상치' if model_anom else ' 정상'}
- 실제 라벨: {' 이상치' if label_anom else ' 정상'}
- 일치 여부: {' 정답' if model_anom == label_anom else ' 오답'}
""")

import matplotlib.pyplot as plt

# 예측값
y_pred = model.predict(X_test)

# 시각화용 x축 생성
x_range = np.arange(SEQ_LEN, SEQ_LEN + len(y_test))  # 50 ~ 8043

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(x_range, test_scaled[SEQ_LEN:], label="Telemetry (scaled)")
ax.plot(x_range, y_pred, label="Prediction")
ax.scatter(
    x_range[errors > threshold],
    test_scaled[SEQ_LEN:][errors > threshold],
    color="red",
    label="Detected Anomaly",
    s=10
)

# 실제 이상 범위 표시 (원본 인덱스 기준 그대로 사용 가능)
for (start, end) in anomaly_ranges:
    ax.axvspan(start, end, color="red", alpha=0.2)

# 선택 시점
ax.axvline(point_index, color='gray', linestyle='--', label='Selected Time')

ax.set_title(f"LSTM Predictor Anomaly Detection - Channel {channel_id}")
ax.set_xlabel("Time")
ax.set_ylabel("Normalized Sensor Value")
ax.legend()
plt.tight_layout()
st.pyplot(fig)
