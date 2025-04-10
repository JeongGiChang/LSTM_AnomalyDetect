# LSTM 기반 이상 탐지 시스템

LSTM(Long Short-Term Memory)을 활용하여 시계열 데이터에서 이상을 탐지하는 딥러닝 기반 프로젝트입니다.  
센서 데이터를 기반으로 학습된 모델을 통해 실시간 또는 기록된 데이터 내 이상 징후를 탐지할 수 있습니다.

---
## 포트폴리오(pdf)

https://drive.google.com/file/d/12Bqs6lkHku_dIsGpzMe0iGHQJZNPmW7_/view?usp=sharing

##  프로젝트 구조

```
LSTM_AnomalyDetect/
├── dataset/ # 학습 및 테스트용 데이터셋
├── report/ # 프로젝트 보고서 및 참고 문서
├── saved_model/ # 학습된 모델 저장 디렉토리
├── anomaly_detect.ipynb # Jupyter 기반 이상 탐지 분석 노트북
├── lstm_streamlit_app.py # 프로토타입 테스트 코드
├── test_model.pkl # 저장된 테스트 모델
└── .gitignore # Git 추적 제외 파일 목록
```
---

## 사용 기술

- Python
- Jupyter Notebook
- LSTM (Keras / TensorFlow)
- Scikit-learn, NumPy, Pandas 등
- Matplotlib, Seaborn (시각화)

---

## 실행 방법

1. 필요한 라이브러리 설치
    ```bash
    pip install -r requirements.txt
    ```

2. Jupyter Notebook 실행
    ```bash
    jupyter notebook anomaly_detect.ipynb
    ```

3. Streamlit prototype 실행
    ```bash
    streamlit run lstm_streamlit_app.py
    ```


---

## 예시 결과

![image](https://github.com/user-attachments/assets/c94dcf1b-a8ff-4e3f-8f65-0478c41e623b)


---

## 보고서 및 문서

- `report/` 디렉토리에는 프로젝트 개요, 설계, 분석 결과가 담긴 문서가 포함되어 있습니다.

---

## 저장된 모델

- `saved_model/lstm_model_P3.pkl`에는 학습된 모델이 저장되어 있으며,
  이를 로드하여 바로 예측 테스트가 가능합니다.

---

## 기여 및 확장

- 다양한 시계열 이상 탐지 방법과 비교
- 모델 튜닝 및 성능 평가
- 실시간 스트리밍 데이터 적용

---

## 문의

프로젝트 관련 문의: [JeongGiChang](https://github.com/JeongGiChang)
포트폴리오 보기 (https://drive.google.com/file/d/12Bqs6lkHku_dIsGpzMe0iGHQJZNPmW7_/view?usp=sharing)
