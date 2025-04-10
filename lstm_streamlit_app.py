
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import joblib
import os
from tensorflow.keras.models import load_model

# ───────────────
# 기본 설정
# ───────────────
st.set_page_config(layout="wide")
st.title(" LSTM 기반 이상치 판단 모델")
st.write("저장된 모델과 데이터를 활용하여 이상 탐지 결과를 확인할 수 있습니다.")

# ───────────────
# 설정값
# ───────────────
channel_id = "P-3"
SEQ_LEN = 50
shift_step = 0  # 예측 위치 보정용

# ───────────────
# 저장된 모델 및 데이터 불러오기
# ───────────────
data_path = "saved_model/lstm_data_P3.pkl"
model_path = "saved_model/lstm_model_P3.h5"
if not os.path.exists(data_path) or not os.path.exists(model_path):
    st.error(" 저장된 모델 파일이나 데이터가 없습니다. 먼저 학습을 진행해주세요.")
    st.stop()

data = joblib.load(data_path)
model = load_model(model_path, compile=False)

# 필요한 요소 불러오기
X_test = data["X_test"]
y_test = data["y_test"]
errors = data["errors"]
threshold = data["threshold"]
scaler = data["scaler"]

# 예측값 직접 계산
y_pred = model.predict(X_test)

# 원본 시계열 데이터
test = np.load(f"dataset/data/data/test/{channel_id}.npy")[:, 0].reshape(-1, 1)
test_scaled = scaler.transform(test)

# 라벨 정보
labels = pd.read_csv("dataset/labeled_anomalies.csv")
anomaly_row = labels[labels["chan_id"] == channel_id]
if anomaly_row.empty:
    st.error(" 해당 채널에 대한 라벨 정보가 없습니다.")
    st.stop()
anomaly_ranges = ast.literal_eval(anomaly_row["anomaly_sequences"].values[0])

# ───────────────
# Streamlit UI
# ───────────────
time = st.slider("검사 시점 (비율)", 0.0, 1.0, 0.5, 0.01)
point_index = int(time * len(test))
adjusted_index = point_index - SEQ_LEN

if 0 <= adjusted_index < len(errors):
    model_is_anomaly = errors[adjusted_index] > threshold
    label_is_anomaly = any(start <= point_index <= end for (start, end) in anomaly_ranges)

    st.markdown(f"""
    ### ✅ 결과 (검사 시점: {point_index})
    - 모델 판단: **{' 이상치' if model_is_anomaly else ' 정상'}**
    - 실제 라벨: **{' 이상치' if label_is_anomaly else ' 정상'}**
    - 일치 여부: **{' 정답' if model_is_anomaly == label_is_anomaly else ' 오답'}**
    """)
else:
    st.warning("선택한 시점이 유효하지 않습니다.")

# ───────────────
# 그래프 출력
# ───────────────
x_range = np.arange(SEQ_LEN, SEQ_LEN + len(y_test))
x_shifted = x_range + shift_step
anomaly_idx = np.where(errors > threshold)[0]

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(x_range, test_scaled[SEQ_LEN:SEQ_LEN + len(y_test)], label="Telemetry (scaled)")
ax.plot(x_shifted, y_pred, label="Prediction")
ax.scatter(x_shifted[anomaly_idx], test_scaled[SEQ_LEN:][anomaly_idx], color='red', label="Detected Anomaly", s=10)

for start, end in anomaly_ranges:
    ax.axvspan(start, end, color="red", alpha=0.2)

ax.axvline(point_index, color='gray', linestyle='--', label="Selected Time")
ax.set_title(f"LSTM Anomaly Detect Result Channel: {channel_id}")
ax.set_xlabel("시간")
ax.set_ylabel("센서값 (정규화)")
ax.legend()
plt.tight_layout()
st.pyplot(fig)
