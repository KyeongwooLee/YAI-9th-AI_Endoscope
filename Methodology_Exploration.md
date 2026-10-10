# Activation 기반 LLM 오류 진단 및 선택적 복구: 방법론 탐색

- 작성일: 2026-10-10
- 대상: GSM8K / Qwen2.5-1.5B-Instruct 파일럿
- 문서 성격: 방법론 후보와 실험 우선순위 제안. 구현 완료나 성능 검증 결과를 의미하지 않는다.
- 확인 기준: GitHub의 수집 코드·README와 아래 원 논문. 로컬 `Study_Plan.md`는 도구 오류로 확인하지 못했다.

## 1. 연구 목표와 중심 질문

이 문서는 프로젝트 목표를 다음과 같이 두고 방법론을 탐색한다.

> 모델의 내부 activation에서 오류 여부·오류 유형·첫 발생 위치를 진단하고, 그 진단을 이용한 선택적 복구가 같은 계산 예산에서 효과적인지 검증한다.

핵심 연구 질문은 다음과 같다.

1. Activation은 토큰 확률이나 출력 텍스트만으로 얻는 정보보다 추가적인 오류 진단 신호를 제공하는가?
2. 최종 답의 정오를 넘어, 오류 유형과 풀이의 첫 오류 위치를 예측할 수 있는가?
3. 진단에 맞는 피드백이나 내부 개입은 일반적인 재검토·재생성보다 복구에 도움이 되는가?
4. 발견한 신호는 문제·표현 방식이 달라져도 유지되며, 실제 출력에 영향을 미치는가?

오류 유형 분류 성능과 인과적 원인 규명은 별도 주장이다. Probe가 특정 유형을 예측한다는 사실만으로 해당 activation이 오류의 원인이라고 결론 내리지 않는다.

## 2. 현재 수집 구조와 적용 범위

확인한 저장소:

- [README.md](https://github.com/KyeongwooLee/YAI-9th-AI_Endoscope/blob/master/README.md)
- [collect.py](https://github.com/KyeongwooLee/YAI-9th-AI_Endoscope/blob/master/collect.py)

| 항목 | 확인한 수집 구조 | 후속 분석에서의 의미 |
| --- | --- | --- |
| 모델·데이터 | Qwen2.5-1.5B-Instruct / GSM8K | 우선 동일 모델·과제 안에서 파일럿을 수행한다. |
| 기본 activation | 전체 토큰의 embedding, 레이어별 residual, final normalization 출력 | 레이어별 probe와 단계별 집계를 먼저 시도할 수 있다. |
| 상세 activation | 예산 내 최대 20문항의 attention 출력, MLP 출력, MLP 중간 activation | 성분·뉴런 분석은 초기에는 사례 분석으로 사용한다. |
| 생성 확률 | 응답 토큰별 log probability, entropy, 상위 5개 후보 정보 | 추가 모델 실행 없이 확률 기반 기준선을 만들 수 있다. |
| 미수집 정보 | Q/K/V, 전체 attention 확률 행렬, 전체 vocabulary logits | 이 정보가 필요한 방법은 추가 수집이나 재실행이 필요하다. |
| 과정 라벨 | 9개 오류 유형과 첫 오류 위치는 미주석 상태 | 유형·위치 지도학습 전에 주석이 필요하다. |
| 구현 범위 | 수집·저장·복원·검증. Judge, probe, 복구 실험은 포함하지 않음 | 아래 방법들은 후속 실험 후보이며 구현된 기능이 아니다. |

현재 코드는 CA/CO/CV/HA/UC/OP/FC/MS/CS를 사용한다. 기존 정의를 유지하고, 복수 오류와 미분류를 허용한다. 이 문서에서 약어의 의미나 분류 경계를 새로 정의하지 않는다.

200문항은 파일럿 계획이며 실제 수집 완료를 확인한 수치가 아니다. 상세 activation 최대 20문항도 9개 유형의 일반화 성능을 주장하기 위한 충분한 표본으로 간주하지 않는다.

## 3. 방법론 후보와 우선순위

아래의 적용 방식·우선순위는 프로젝트에 대한 제안이다. 원 논문의 결과가 이 모델과 데이터에서 재현된다는 의미는 아니다.

| 번호 | 방법론 | 프로젝트에서 시도할 실험 | 우선순위 |
| --- | --- | --- | --- |
| 1 | 확률 기반 오류 탐지 | 저장된 entropy·log probability·상위 후보 확률 차이로 오류 점수를 만든다. | 필수 기준선 |
| 2 | 레이어별 linear probe | 각 레이어 residual에 로지스틱 회귀를 학습하고, 토큰 위치·집계 방식을 비교한다. | 가장 먼저 |
| 3 | 다중 라벨 오류 유형 probe | 9개 유형을 각각 예측한다. 선형 모델부터 시작하고 작은 MLP와 비교한다. | 핵심 |
| 4 | 단계별 오류 탐지·첫 오류 위치 추정 | 단계별 residual을 집계하고, 오류 점수가 처음 임계값을 넘는 위치를 예측한다. | 핵심 |
| 5 | Activation 궤적 분석 | 단계 사이 벡터 차이·cosine 변화·이차 차분을 사용한다. | 확장 |
| 6 | SAE 특징 기반 진단 | Residual을 희소 특징으로 분해하고 오류 유형과의 연관성을 조사한다. | 해석 확장 |
| 7 | Logit Lens / Tuned Lens | 중간 레이어를 토큰 예측 분포로 읽고 숫자·연산자 후보 변화를 분석한다. | 사례 분석 |
| 8 | Activation patching·ablation | 정렬 가능한 정상/오류 실행에서 activation을 교체하거나 제거하고 출력 변화를 측정한다. | 인과 검증 |
| 9 | Attribution patching / AtP* | Gradient로 개입 후보를 좁힌 뒤 상위 후보에 실제 patching을 수행한다. | 개입 비용 절감 |
| 10 | 진단에 따른 선택적 피드백·재생성 | 예측한 오류 유형·위치를 이용해 재검토·재계산·단계별 재생성을 유도한다. | 목표 달성 실험 |
| 11 | Contrastive Activation Addition | 대응하는 정상/오류 사례의 activation 차이 방향으로 생성 중 개입한다. | 후순위 |
| 12 | LoReFT | 모델 가중치를 고정하고 일부 레이어의 표현을 수정하는 작은 개입을 학습한다. | 후순위 |

### 3.1 확률 기준선과 레이어별 probe

**검증할 가설:** 내부 표현이 출력 확률만으로 설명되지 않는 오류 진단 정보를 제공한다.

확률 기준선은 평균·최대 entropy, 평균 log probability, 후보 확률 차이 등을 사용한다. 레이어별 probe는 프롬프트 마지막 토큰, 각 단계 마지막 토큰, 응답 평균 등 라벨 없이 정할 수 있는 위치·집계를 비교한다. Probe와 집계 방식은 validation에서 선택한다.

`LLMs Know More Than They Show`는 오류 탐지 신호가 토큰 위치에 따라 달라지며 오류 유형 예측에도 활용될 수 있음을 보고한다. 동시에 데이터셋 사이 일반화가 제한적이다. `Reasoning Models Know When They're Right`는 reasoning 모델의 중간 답 정오를 hidden-state probe로 평가한다. 두 연구의 결과는 현재 Qwen 파일럿에 대한 성능 보장이 아니다. [R1, R2]

### 3.2 다중 라벨 오류 유형 진단

**검증할 가설:** 내부 표현으로 오류의 존재뿐 아니라 유형을 구분할 수 있다.

기존 9개 유형에 대해 독립적인 이진 출력으로 시작한다. 단일 softmax로 한 유형만 강제하지 않는다. 정답 풀이, 미분류 오류, 판단 불가능 사례의 처리 규칙을 주석 지침에 명시한다. 작은 MLP는 선형 모델과 비교할 추가 후보로 둔다.

유형별 주석 수와 혼동을 먼저 확인한다. 드문 유형은 별도로 보고하며, 미주석을 정상으로 처리하지 않는다. 이 다중 라벨 설계는 프로젝트 분류 체계에 맞춘 제안이다.

### 3.3 단계별 오류 탐지와 첫 오류 위치

**검증할 가설:** 단계별 activation에서 오류가 처음 발생하는 위치를 예측할 수 있다.

풀이를 고정된 규칙으로 단계화하고, 각 단계의 마지막 토큰 또는 평균 residual을 사용한다. 단계별 분류 점수에서 첫 임계값 초과 위치를 예측하되, 오류 없음도 출력할 수 있어야 한다.

`Let's Verify Step by Step`는 최종 결과 감독과 과정 감독을 비교하고, `ProcessBench`는 첫 오류 단계 또는 모든 단계가 맞음을 판정하는 평가를 제공한다. 기존 process reward model은 비교 후보지만, 사용 가능한 체크포인트·대상 모델·추론 비용을 확인한 뒤 포함한다. [R3, R4]

최종 답이 맞아도 과정 오류가 있을 수 있다. 반대로 형식 실패·잘림을 논리적 과정 오류와 동일시하지 않는다. 첫 오류 이후 모든 단계를 같은 오류 유형으로 자동 라벨링하지 않는다.

### 3.4 Activation 궤적 분석

**검증할 가설:** 단일 상태보다 단계 사이 변화가 오류 시작점을 잘 드러낸다.

단계 벡터의 차이, cosine 변화, 변화량의 변화 등을 특징으로 계산하고 선형 분류기와 결합한다. 우선 복잡한 시퀀스 모델 없이 단일 단계 probe와 비교한다.

2026년 `Where Does Reasoning Break?`는 hidden-state 궤적의 기하학적 특징과 teacher–student 구조를 연구한다. 해당 teacher는 추론 시 단계 라벨을 요구하므로 oracle 분석이다. BiLSTM student는 전체 생성 풀이를 보는 사후 탐지기이며, 실시간 탐지기로 해석할 수 없다. 논문은 student의 분포 변화에 따른 성능 저하도 보고한다. [R5]

프로젝트에서 라벨 기반 정규화·PCA를 실사용 탐지 결과에 섞지 않는다. 일반적인 표준화·차원 축소도 train에서만 학습한다.

### 3.5 SAE 특징 기반 진단

**검증할 가설:** 희소 특징으로 오류 유형과 연관된 내부 신호를 더 쉽게 해석할 수 있다.

모델·레이어·수집 지점이 맞는 SAE로 activation을 인코딩하고, raw residual probe와 비교한다. 특징의 안정성과 특정 유형에서의 활성도를 확인하고, 필요하면 제거·증폭 개입으로 출력을 검증한다.

`Do I Know This Entity?`는 사실 회상에서 entity 인지와 관련된 SAE 특징을 발견하고 개입한다. 이 결과가 수학 오류 유형에도 성립하는지는 별도 실험이 필요하다. SAE 가중치는 오류 유형 분류기 가중치가 아니므로 후속 분류 학습을 대체하지 않는다. [R6]

### 3.6 Logit Lens / Tuned Lens

**검증할 가설:** 중간 레이어의 예측 변화가 오류 사례를 설명하는 데 도움이 된다.

숫자·연산자·답 후보가 레이어를 지나며 어떻게 달라지는지 조사한다. Tuned Lens는 중간 hidden state를 vocabulary 분포로 읽기 위한 학습된 변환을 사용한다. 단순 Logit Lens의 불안정성을 줄이기 위한 접근이다. [R7]

현재 저장 파일에는 전체 vocabulary logits가 없으므로 출력 head 적용 또는 추가 계산이 필요하다. 다중 토큰 숫자·답은 전체 문자열의 조건부 확률로 다뤄야 하며, 단일 숫자 토큰 순위만으로 답에 대한 지식을 단정하지 않는다. Lens 결과만으로 인과성을 주장하지 않는다.

### 3.7 Activation patching·ablation과 AtP*

**검증할 가설:** 진단과 연관된 성분을 바꾸면 모델의 실제 출력이 달라진다.

정렬 가능한 정상/오류 실행을 구성하고 레이어·토큰·성분 단위 교체를 수행한다. 무작위 위치·무관한 사례·반대 방향 개입 등을 대조 조건으로 둔다. 성분 제거는 정상 능력까지 손상시킬 수 있으므로 정상 사례도 평가한다.

Activation patching 결과는 교체 방식과 평가 지표에 민감하다. 정상 답의 activation을 제공하는 실험은 실사용 복구가 아니라 oracle 인과 분석으로 구분한다. [R8]

AtP*는 gradient 기반 근사로 후보 탐색 비용을 줄이는 방법이다. 후보를 좁히는 데 사용하고 상위 후보는 실제 개입으로 재검증한다. 저장된 detached activation만으로 gradient를 복원할 수 없으므로 모델 재실행이 필요하다. [R9]

### 3.8 진단에 따른 선택적 피드백·재생성

**검증할 가설:** 예측한 오류 유형·위치를 이용한 복구가 일반적인 재검토보다 유용하다.

오류 위치를 제공하면 교정에 도움이 될 수 있다는 연구를 참고하되, 사람의 실제 오류 위치를 제공하는 조건과 모델이 예측한 위치를 제공하는 조건을 구분한다. [R10]

권장 비교 조건은 다음과 같다.

1. 일반적인 재검토 피드백.
2. 예측한 오류 위치만 제공하는 피드백.
3. 예측한 오류 유형·위치를 제공하는 피드백.
4. 사람이 주석한 실제 유형·위치를 제공하는 oracle 피드백.

재계산, 조건 재확인, 특정 단계부터 재생성 등 복구 행동은 기존 유형 정의에 맞춰 train/validation에서 설계한다. Test 정답·오류 라벨은 실사용 복구 입력에 제공하지 않는다.

### 3.9 CAA와 LoReFT를 이용한 내부 개입

**검증할 가설:** 오류와 관련된 표현에 개입해 발생률을 줄일 수 있다.

CAA는 대조 사례의 activation 차이 방향을 생성 중 더하거나 빼는 방법이다. LoReFT는 모델 가중치를 고정한 상태에서 표현에 대한 작은 학습된 개입을 적용한다. 이 문서의 LoReFT는 `Representation Finetuning` 계열을 뜻한다. [R11, R12]

정상 평균과 오류 평균의 차이에는 문제 난도·길이·문장 형식이 섞일 수 있다. 가능한 대응 사례를 구성하고, 크기를 맞춘 무작위 방향과 반대 방향을 대조한다. 개입 레이어·강도는 validation에서 선택하고, 정답 사례의 손상과 출력 형식 변화도 확인한다.

## 4. 권장 실험 순서

### 단계 A: 데이터와 라벨 점검

- 수집 검증을 통과한 activation만 분석에 사용한다.
- 9개 유형·다중 라벨·미분류·첫 오류 위치의 주석 기준을 정한다.
- 일부 문항을 독립적으로 이중 주석하고 유형·위치의 불일치를 검토한다.
- 유형별 빈도, 정오 비율, 잘림·형식 실패·판단 불가능 사례를 집계한다.
- 문제 ID·템플릿 기준으로 train/validation/test를 고정한다.

### 단계 B: 최소 진단 실험

- 확률 기준선과 레이어별 선형 probe를 비교한다.
- 단계별 오류 여부·첫 오류 위치·다중 라벨 유형 예측을 평가한다.
- 텍스트만 사용하는 모델과 activation만 사용하는 모델을 비교한다.
- 필요하면 텍스트·확률·activation 결합 모델로 추가 정보의 가치를 확인한다.

이 단계는 방법론 1~4에 해당한다. 복잡한 모델보다 신호의 존재와 라벨 품질을 먼저 확인한다.

### 단계 C: 같은 예산에서 선택적 복구

- 무작위 선택, 확률 기반 선택, activation 진단 기반 선택을 비교한다.
- 복구 대상 선정과 피드백 내용을 분리해 비교한다.
- 재시도 수와 추가 토큰 예산을 맞추고 실제 시간도 기록한다.
- 진단만 수행한 경우와 진단 후 복구한 경우를 구분한다.

이 단계는 방법론 10에 해당한다. 오류 유형·위치 진단이 최종 성능에 기여하는지 검증한다.

### 단계 D: 설명과 인과 검증 확장

- 궤적·SAE·Lens로 반복적인 오류 사례를 분석한다.
- Patching과 ablation으로 후보 성분의 영향을 검증한다.
- 이후 CAA, 필요하면 LoReFT로 복구 개입을 확장한다.

## 5. 공통 평가 원칙

| 평가 대상 | 권장 지표·비교 | 주의점 |
| --- | --- | --- |
| 오류 여부 | AUROC, 오류 클래스 AUPRC, 임계값별 precision/recall | 클래스 비율과 표본 수를 함께 보고한다. |
| 오류 유형 | 유형별 precision/recall/F1, macro-F1, 유형별 표본 수 | 복수 오류를 허용한다. 미분류·미주석을 정상으로 취급하지 않는다. |
| 첫 오류 위치 | 정확한 위치 일치율, ±1단계 일치율, 오류 없음 판정 | 위치 평가의 분모와 오류 없음 처리 규칙을 명시한다. |
| 선택적 처리 | 처리 비율별 오류율·복구 후 정확도 | 임계값을 test에서 선택하지 않는다. |
| 복구 | 오답→정답, 정답→오답, 전체 정확도 변화 | 정답 사례 손상을 빠뜨리지 않는다. |
| 비용 | 추가 생성 토큰, 모델 호출 수, 진단·복구 시간 | 같은 재시도·토큰 예산과 실제 비용을 함께 보고한다. |
| 일반화 | 문제·템플릿·표현 변화에 따른 성능 | 모델 변경에 대한 일반화는 별도 검증한다. |

여러 단계가 있어도 독립 표본 수는 단순 단계 수와 같지 않다. 같은 문제의 단계·생성 변형·정상/오류 대응 사례는 같은 split에 둔다. 신뢰구간을 계산한다면 문제 단위 재표집을 사용한다.

표준화, PCA, 특징 선택, 레이어 선택, 임계값, 복구 프롬프트와 개입 강도는 train/validation에서만 결정한다.

## 6. 토큰 시점과 정보 누출 방지

현재 수집기의 activation은 저장된 생성 토큰을 causal forward로 다시 읽은 replay 값이다. 실제 생성 중 수집한 hook 값이라고 표현하지 않는다.

- 위치 `t`의 hidden state는 `t`까지 읽은 상태이며 `t+1`을 예측한다.
- 위치 `t`의 응답 log probability는 `t-1`의 logits에서 얻는다.
- 생성 전 예측은 프롬프트까지의 정보만 사용한다.
- 생성 중 단계 탐지는 해당 시점까지 생성된 정보만 사용한다.
- 전체 응답 평균·BiLSTM처럼 뒤의 단계를 사용하는 방법은 사후 진단으로 보고한다.
- 사람이 고른 첫 오류 토큰·실제 정답·실제 오류 유형을 사용하는 조건은 oracle로 분리한다.
- 최종 숫자 정오와 풀이 과정 오류를 같은 라벨로 간주하지 않는다.

## 7. 범위와 남은 불확실성

첫 구현 후보는 확률 기준선, residual 선형 probe, 단계별 위치·유형 진단, 선택적 피드백 복구다. SAE·전체 회로 탐색·학습된 내부 개입은 후속 후보로 둔다.

이 문서는 후보 탐색이며 구현 범위 확정이나 수집 정책 변경을 승인하는 문서가 아니다. 로컬 계획서와 기존 범위를 대조하고, 주석·실행 환경 등 미확정 사항을 정한 뒤 구현한다.

현재 확인하지 않은 사항:

- 실제 파일럿 수집 완료 여부와 유형별 라벨 분포.
- 원격 GPU에서의 각 방법 실행 시간·메모리·디스크 사용량.
- 현재 모델·레이어·수집 지점에 맞는 SAE와 Tuned Lens 자산의 가용성.
- 각 방법의 현재 데이터에서의 성능과 통계적 불확실성.

주요 위험은 희귀 유형의 표본 부족, 주석 불일치, 문제 중복에 따른 누출, 미래 정보·oracle 라벨 사용, 정상 사례를 손상시키는 복구다.

## 8. 참고 논문

- **[R1]** Orgad et al. *LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations*. https://arxiv.org/abs/2410.02707
- **[R2]** Zhang et al. *Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification*. https://arxiv.org/abs/2504.05419
- **[R3]** Lightman et al. *Let's Verify Step by Step*. https://arxiv.org/abs/2305.20050
- **[R4]** Zheng et al. *ProcessBench: Identifying Process Errors in Mathematical Reasoning*. https://arxiv.org/abs/2412.06559
- **[R5]** Alvarez and Baheri. *Where Does Reasoning Break? Step-Level Hallucination Detection via Hidden-State Transport Geometry*. 2026년 preprint. https://arxiv.org/abs/2605.13772
- **[R6]** Ferrando et al. *Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models*. https://arxiv.org/abs/2411.14257
- **[R7]** Belrose et al. *Eliciting Latent Predictions from Transformers with the Tuned Lens*. https://arxiv.org/abs/2303.08112
- **[R8]** Zhang and Nanda. *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods*. https://arxiv.org/abs/2309.16042
- **[R9]** Kramár et al. *AtP\*: An efficient and scalable method for localizing LLM behaviour to components*. https://arxiv.org/abs/2403.00745
- **[R10]** *LLMs cannot find reasoning errors, but can correct them given the error location*. https://arxiv.org/abs/2311.08516
- **[R11]** Panickssery et al. *Steering Llama 2 via Contrastive Activation Addition*. https://arxiv.org/abs/2312.06681
- **[R12]** Wu et al. *ReFT: Representation Finetuning for Language Models*. https://arxiv.org/abs/2404.03592

문헌 탐색은 프로젝트와 직접 관련된 대표 연구를 중심으로 수행했다. 전수 문헌 조사나 모든 결과의 독립 재현 검증은 수행하지 않았다.
