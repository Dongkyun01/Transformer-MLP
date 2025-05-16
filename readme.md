
---


## RetailRocket Session Time Prediction

## - 프로젝트 개요

본 프로젝트는 RetailRocket에서 제공하는 e-commerce clickstream 데이터를 활용하여, 사용자의  **다음 세션까지 걸리는 시간(Time to Next Session)** 을 예측하는 문제를 다룹니다. 

이 예측은 마케팅 자동화, 사용자 이탈 방지 전략, 추천 시스템의 타이밍 최적화 등에 활용될 수 있습니다.



##  - 사용 데이터

- **출처**: [RetailRocket Dataset - Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
- **파일**: `events.csv`
- **구성 정보**:
  - `timestamp`: 행동 발생 시점 (밀리초 단위)
  - `visitorid`: 방문자 ID
  - `event`: 행동 종류 (`view`, `addtocart`, `transaction`)
  - `itemid`, `transactionid`: 상품 및 거래 ID

---

## - 데이터 전처리 및 생성 Feature

### 시퀀스 데이터 생성
- 각 `visitorid`를 기준으로 세션을 정렬하여 행동 시퀀스 `[0, 0, 1, 0, 2]` 형태로 구성  
- 각 시퀀스는 Transformer의 입력으로 사용됨

### 타겟 값 생성
- 각 세션 종료 시점과 다음 세션 시작 시점 간의 시간 차(`seconds`)를 계산하여 타겟으로 설정  
- 예측 대상은 `log1p(Time to Next Session)`으로 변환하여 학습 후, 결과를 `expm1()`로 복원

### 추가 Feature (MLP 입력)
| Feature 이름         | 설명                                |
|----------------------|-------------------------------------|
| session_length       | 세션 내 행동 수                     |
| event_diversity      | 행동 유형의 다양성 (0~3)            |
| view_count           | `view` 행동 횟수                    |
| addtocart_count      | `addtocart` 행동 횟수               |
| transaction_count    | `transaction` 행동 횟수             |
| session_duration     | 세션 전체 지속 시간 (초)            |
| start_hour           | 세션 시작 시간 (0~23시)             |
| start_weekday        | 세션 시작 요일 (0:월 ~ 6:일)        |

---

## - 모델 구조

### Transformer + MLP

```text
(시퀀스 입력) → Embedding → Positional Encoding → Transformer Encoder → 시퀀스 표현
(추가 feature) → MLP (2-layer) → feature 표현
→ 두 표현을 concat → Linear → 예측값 출력
````

* 시퀀스 처리 : Transformer Encoder (2-layer, 4-head, d_model64)
* 시퀀스 길이 가변, 패딩 적용
* 추가 feature 처리: 2-layer MLP (64 units)
* 최종 출력은 회귀 예측값 (`Time to Next Session`)
* Loss: MSE Loss (로그 스케일 적용 후, 최종 역변환 실행)

---

## - 학습 설정

* **Train/Test Split**: 80:20
* **Target**: log1p(time\_to\_next\_session)
* **Optimizer**: Adam (lr=0.0005)
* **Loss Function**: MSELoss
* **Epochs**: 20
* **Batch Size**: 64

---

## - 성능 및 결과

* **Test MSE**: `180.96`
* **Test MAE**: `6.15`
* **MAE (Short session < 5 actions)**: `6.15`
* **MAE (Long session ≥ 5 actions)**: `6.92`

### Prediction Histogram
![image](https://github.com/user-attachments/assets/fb7da7ad-f4d6-4f8c-b66e-92053c2f7520)

![Prediction Histogram](./Prediction_Histogram.png)

* 실제값은 0\~100초 사이에 집중
* 예측값도 유사 범위에 분포하지만 전체적으로 **과소 추정**하는 경향
* 특히, 예측값이 **짧은 세션 위주로 몰림** ⇒ 모델이 평균값에 수렴하려는 경향을 보임

---

## - 문제 분석 및 결론

* **데이터 분포 편향**: 대부분의 세션이 매우 짧은 시간 안에 재방문하는 것으로 나타나며, long-tail 형태의 분포를 가짐
* **타겟 변환**: log 스케일 적용으로 일부 skew는 완화되었지만, 긴 세션 예측은 여전히 부정확함
* **MLP feature 활용**: 단순 시퀀스 기반 Transformer보다 더 많은 세션 특성을 반영했으나, 긴 세션 일반화에 한계

---

## - 향후 개선 방안

1. **데이터 리샘플링**

   * 긴 세션을 oversample하거나, 짧은 세션을 undersample하여 모델 학습 시 균형 유도

2. **Sequence Feature 강화**

   * `itemid`, `categoryid` 등을 embedding 형태로 포함
   * 행동 간 시간 간격(time gap)도 추가 feature로 고려

3. **Multi-Task Learning 확장**

   * 다음 행동 예측, 세션 유지 시간 등 다중 목적 예측을 동시에 수행하여 일반화 강화

4. **Attention 시각화 분석**

   * Transformer의 self-attention weight를 시각화하여, 모델이 어느 행동에 주목하는지 해석 가능

---

## - 프로젝트 구조

```bash
.
├── make_dataset.py         # 데이터 전처리 및 저장
├── dataset.py              # PyTorch Dataset 정의
├── model.py                # Transformer + MLP 모델 정의
├── train.py                # 학습 스크립트
├── eval.py                 # 평가 및 시각화
├── retailrocket_dataset.pkl  # 저장된 데이터셋
└── Prediction_Histogram.png  # 예측 결과 히스토그램


[Kaggle Dataset - RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

