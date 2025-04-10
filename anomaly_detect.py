

# In[33]:


import streamlit as st
import matplotlib as plt
import joblib
st.title('이상치 판단 모델')
st.write('이상치를 판단하고 실제 결과와 비교해보세요!')


# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
# In[ ]:



import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import joblib

# ───────────────
# 설정값
# ───────────────
channel_id = "P-3"

# 저장된 데이터 불러오기
data = joblib.load("saved_model/lstm_data_P3.pkl")
y_test = data["y_test"]
y_pred = data["y_pred"]
errors = data["errors"]
threshold = data["threshold"]
scaler = data["scaler"]
SEQ_LEN = data["SEQ_LEN"]

# 원본 데이터
test = np.load(f"dataset/data/data/test/{channel_id}.npy")[:, 0].reshape(-1, 1)
test_scaled = scaler.transform(test)

# 라벨 정보
labels = pd.read_csv("dataset/labeled_anomalies.csv")
anomaly_ranges = ast.literal_eval(labels[labels["chan_id"] == channel_id]["anomaly_sequences"].values[0])

# ───────────────
# Streamlit UI
# ───────────────
st.title("📦 저장된 데이터 기반 이상치 탐지 결과 확인")

time = st.slider("검사 시점 (비율)", 0.0, 1.0, 0.5, 0.01)
point_index = int(time * len(test))
adjusted_index = point_index - SEQ_LEN

if 0 <= adjusted_index < len(errors):
    model_is_anomaly = errors[adjusted_index] > threshold
    label_is_anomaly = any(start <= point_index <= end for (start, end) in anomaly_ranges)

    st.markdown(f"""
    ### ✅ 결과 (검사 시점: {point_index})
    - 모델 판단: **{'🛑 이상치' if model_is_anomaly else '✅ 정상'}**
    - 실제 라벨: **{'🛑 이상치' if label_is_anomaly else '✅ 정상'}**
    - 일치 여부: **{'🎯 정답' if model_is_anomaly == label_is_anomaly else '❌ 오답'}**
    """)
else:
    st.warning("선택한 시점이 유효하지 않습니다.")

# ───────────────
# 그래프 출력
# ───────────────
x_range = np.arange(SEQ_LEN, SEQ_LEN + len(y_test))
# 기존 x_range: SEQ_LEN ~ (SEQ_LEN + len(y_test))
# 이동량을 N만큼 주자
shift_step = 450  # 예: 100만큼 이동시키고 싶다면

# 이동된 x축 생성
x_shifted = x_range + shift_step





fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(x_range, test_scaled[SEQ_LEN:SEQ_LEN + len(y_test)], label="Telemetry (scaled)")
ax.plot(x_shifted, y_pred, label="Prediction")

# 이상치로 판단된 시점
anomaly_idx = np.where(errors > threshold)[0]
ax.scatter(x_shifted[anomaly_idx], test_scaled[SEQ_LEN:][anomaly_idx], color='red', label="Detected Anomaly", s=10)

# 실제 이상 구간
for start, end in anomaly_ranges:
    ax.axvspan(start, end, color="red", alpha=0.2)

# 선택 시점
ax.axvline(point_index, color='gray', linestyle='--', label="Selected Time")

ax.set_title(f"LSTM 이상치 탐지 결과 - 채널 {channel_id}")
ax.set_xlabel("시간")
ax.set_ylabel("센서값 (정규화)")
ax.legend()
st.pyplot(fig)

