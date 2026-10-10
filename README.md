# GSM8K 연구 데이터 수집

`collect.py`는 `Study_Plan.md`의 Qwen2.5-1.5B 파일럿을 위한 수집 CLI다. 문제, 모델 풀이, 실제 정답, 최종 정오, 전체 토큰의 레이어별 activation을 저장한다. `analyze.py`는 검증된 수집 결과로 확률 기준선·레이어별 선형 probe·작은 MLP를 학습해 정답/오답 판별을 평가한다. 과정 오류 판정·Judge 호출·복구 실험은 포함하지 않는다.

## Linux RTX 3090 설치·실행

Python 3.11과 CUDA 12.4용 PyTorch를 사용할 수 있는 NVIDIA 드라이버가 필요하다. 아래 명령은 **연구 작업 공간 안에서** 실행한다. 예를 들어 `/work/yaicon/research_base`에 이 폴더를 복사한 뒤 해당 디렉터리로 이동한다.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
"$HOME/.local/bin/uv" venv --python 3.11 --seed .venv
source .venv/bin/activate
python -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install --no-cache-dir -r requirements.txt
python -m unittest -v test_collect.py

# 먼저 실제 모델로 3문항 생성·상세 activation 저장을 확인한다.
python collect.py --split train --limit 3 --seed 42 --output-dir smoke
python collect.py --split train --limit 3 --seed 42 --output-dir smoke --resume

# 3문항 결과의 replay_comparison.passed와 파일 복원을 확인한 뒤 파일럿을 실행한다.
python collect.py --split train --limit 200 --seed 42 --output-dir pilot
```

모델과 GSM8K는 공개 Hugging Face 자료를 처음 실행할 때 내려받는다. 별도 API key는 필요 없다. 모델·데이터의 실제 commit SHA를 고정해 manifest에 저장한다. [모델 원본](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [데이터 원본](https://huggingface.co/datasets/openai/gsm8k).

다시 시작하려면 **같은 명령에 `--resume`**를 추가한다. 완료된 응답은 다시 생성하지 않고, 완료된 activation은 파일 크기·SHA256·토큰 범위를 확인한다. 중간에 끊긴 문항의 activation은 같은 토큰으로 다시 수집한다. 코드·패키지·GPU·설정이 바뀌면 재개를 거절한다. SIGKILL/서버 종료로 `.lock`이 남았으면 기존 프로세스가 종료된 것을 확인한 뒤 해당 실행 디렉터리의 `.lock` 파일만 삭제한다.

전체 split은 `--limit 0`으로 선택한다. 용량 부족 시 전체 activation 수집 전에 중단할 수 있다. test는 별도 output 디렉터리를 사용하고, test 정답률로 프롬프트·분석 설정을 선택하지 않는다. 이 수집기는 probe용 train/validation 분할이나 템플릿 그룹 탐지를 수행하지 않는다. 후속 분석에서 문제 ID·템플릿 기준 분할을 고정해야 한다.

## 50GiB 저장 정책

`--output-dir`의 **부모 디렉터리**가 예산을 관리하는 작업 공간이다. `.venv`, 모델/데이터 캐시, 결과를 모두 그 안에 둔다. 캐시는 자동으로 `부모/cache`를 사용한다. 다른 위치의 사전 설치 환경은 이 디렉터리 크기에 포함되지 않으므로, 50GiB 전체 제한에 포함되는 환경은 위 설치 방법처럼 작업 공간 안에 설치한다.

- 작업 공간 사용량은 최대 45GiB, 실제 파일시스템 여유는 최소 5GiB를 유지한다. 파일 복제본·캐시를 임의로 삭제하지 않는다.
- 200개 응답을 먼저 생성하고 실제 토큰 길이로 필요한 BF16 tensor 크기와 파일 여유량을 계산한다.
- 전체 문항의 기본 activation을 먼저 예약하고, 남는 공간에서 seed 42의 고정 순서에 따라 최대 20개를 상세 수집한다. 정답 여부는 상세 문항 선택에 사용하지 않는다.
- 기본 문항 전부가 들어가지 않으면 activation을 시작하지 않는다. 응답 파일은 보존된다. 같은 50GiB 공간에서 train/test 전체 activation이 들어간다고 보장하지 않는다.
- 실행 중 다른 프로세스가 쓰는 공간도 실제 디스크 여유 검사에 반영한다. 이 CLI의 예산은 OS의 hard quota를 대체하지 않는다. 작업 공간 밖의 파일이나 다른 프로세스의 지속적인 쓰기까지 50GiB로 제한하려면 서버 quota도 설정한다.

3문항 점검과 파일럿은 같은 부모 디렉터리에 두어 모델 캐시를 공유한다. 3문항 점검 결과도 예산에 포함된다. 파일럿 저장 공간이 부족하면 점검 결과를 사용자가 별도로 보관·정리한 뒤 재개한다.

## 저장 내용과 토큰 의미

```text
pilot/
  manifest.json                # 원본 revision·환경·생성 설정·상세 문항·tensor 규격
  summary.json                 # 정확도·추출 실패·잘림·토큰·시간·용량
  items/train_00042/
    response.json              # 문제·정답 풀이·응답·숫자 정오·토큰·생성 확률
    activations/
      activation.json          # 검증 상태·재수집 확률·파일 목록·shape·SHA256
      embedding/chunk_0000.pt
      layer_00/chunk_0000.pt
      ...
      layer_27/chunk_0000.pt
      final_norm/chunk_0000.pt
```

`.pt`는 `token_start`, `token_end`, `tensors`를 가진 dict다. 각 tensor는 CPU BF16의 `[토큰 수, 특징 차원]`이며 범위는 `[token_start, token_end)`다.

| 수집 대상 | 전체 문항 | 상세 문항 | 차원 |
| --- | --- | --- | --- |
| 입력 embedding | O | O | 1536 |
| 각 레이어 residual: attention·MLP residual 합산 후, 마지막 norm 전 | O | O | 1536 |
| 모델의 최종 normalization 출력 | O | O | 1536 |
| attention 출력: `o_proj` 후, residual 합산 전 | — | O | 1536 |
| MLP 출력: `down_proj` 후, residual 합산 전 | — | O | 1536 |
| MLP 중간 activation: `SiLU(gate_proj) × up_proj`, `down_proj` 입력 | — | O | 8960 |

Q/K/V·전체 attention 확률 행렬·전체 vocabulary logits는 저장하지 않는다. 응답 토큰별 log probability, entropy, 상위 5개 후보 ID·log probability는 JSON으로 저장한다. 확률은 softmax 원본 분포 기준이며 greedy 선택 확률 1을 의미하지 않는다. 생성한 EOS도 토큰 수·확률·activation에 포함된다.

모든 토큰 ID와 `decoded_full_text`를 보존한다. `token_offsets`의 `text_offset`은 이 텍스트에서의 Python 문자열 위치, `byte_offset`은 tokenizer의 원래 byte stream 위치다. UTF-8 글자가 여러 토큰에 걸치면 text 범위가 겹칠 수 있다. `prompt_length - 1`은 생성 직전 프롬프트 마지막 토큰, `prompt_length`부터는 생성된 토큰이다. 토큰 위치 `t`의 hidden state는 **t까지 읽은 상태**이며 **t+1을 예측**한다. 위치 `t`의 응답 log probability는 **t-1의 logits**에서 얻는다. 이 차이를 분석에서 유지한다.

activation은 실제 생성 중 hook을 건 값이 아니라, **정확한 저장 토큰을 causal forward로 다시 읽은 값**이다. 프롬프트는 생성 때처럼 한 번에 처리하고 응답은 KV cache를 유지해 **한 토큰씩 계산**한다. CPU에는 최대 128토큰의 activation만 모아 파일로 저장한다. BF16에서 여러 토큰을 한 번에 계산하면 생성 때와 수치 차이가 생길 수 있어 계산 순서를 동일하게 유지한다. 마지막 EOS도 별도로 입력해 activation을 남긴다. 각 문항의 재수집 확률과 생성 확률을 비교한다. top-1 후보 불일치가 없고 최대 log probability 절대 차이가 0.10 이하일 때만 `complete=true`로 둔다. 검증 실패 시 파일은 남기되 실행을 중단한다. 허용 오차를 임의로 늘려 통과시키지 않는다.

생성은 배치 1·raw logits greedy·BF16·eager attention·PyTorch CPU thread 1로 고정한다. 입력 최대 1,024토큰과 응답 최대 1,024토큰을 사용한다. 긴 입력은 자르지 않고 `input_too_long` 오류로 저장하며 정확도 분모에 포함한다. 최대 길이로 끝난 응답은 `truncated=true`로 보존하고 실패 처리한다. `numeric_match`는 잘린 응답의 숫자 일치도 따로 보존한다. 정답표는 모델 입력에 포함되지 않는다.

마지막 `####` 뒤에는 숫자만 허용한다. 쉼표 천 단위·부호·소수는 정확한 Decimal 비교로 정규화하며 단위·분수·지수 표기나 뒤따르는 설명은 형식 실패로 둔다. `correct`는 최종 숫자 정답 여부다. 풀이가 맞다는 뜻은 아니다. `reasoning_annotation`은 미주석 상태로 두고 9개 오류 유형·다중 라벨·첫 오류 위치를 추후 별도로 기록한다.

## 결과 읽기

실제 저장된 문항 디렉터리를 지정한다.

```bash
python read_result.py pilot/items/train_00042 --layer 14
```

분석 코드에서는 한 레이어만 복원해 사용한다.

```python
from read_result import load_layer

record, residual = load_layer("pilot/items/train_00042", "layer_14", "residual")
before_generation = residual[record["prompt_length"] - 1].float()
after_response = residual[record["prompt_length"]:].float()
# 상세 문항에만 존재한다.
record, neurons = load_layer("pilot/items/train_00042", "layer_14", "mlp_intermediate")
```

분석은 `activation.json`의 검증 상태를 확인한 파일만 사용한다. 사람이 고른 첫 오류 위치는 oracle 분석으로 분리한다. 정답 라벨 없이 정한 위치·단계 집계와 섞지 않는다. 최종 정답과 풀이 과정 오류를 같은 라벨로 취급하지 않는다.

## 검증 범위

`test_collect.py`는 작은 임의 초기화 Qwen2 모델로 토큰 위치·마지막 토큰·상세 activation 값·저장/복원·재수집 비교·중단 후 재수집·손상 파일 검출·용량·숫자 판정을 검증한다. 학습된 1.5B 모델의 품질이나 원격 RTX 3090 실행 검증을 대신하지 않는다. 원격 설치 후 3문항 점검을 반드시 실행한다.

## 정답·오답 판별 분석

`Methodology_Exploration.md`의 방법론 1~3을 적용한다. 오류 유형 주석이 없으므로 3번은 동일 activation 특징에 대한 선형 probe와 이진 MLP 비교로 수행한다. 저장된 `correct`를 사용하고 **오답을 양성 클래스 1**로 둔다. 형식 실패·잘림도 오답에 포함하며, 풀이 과정의 오류나 오류 유형을 판정하지 않는다.

수집에 사용한 Python 환경에서 분석 의존성만 추가한다. 원본 수집 환경의 패키지를 재설치하지 않는다.

```bash
python -m pip install -r requirements-analysis.txt
python -m unittest -v test_analyze.py test_collect.py
python analyze.py --input-dir pilot --output-dir analysis/pilot --seed 42
```

분석은 CPU에서 실행하며 모델이나 데이터셋을 다시 다운로드하지 않는다. 시작 시점에 완료·replay 검증을 통과한 문항 목록을 고정한다. 생성 실패·activation 미완료 문항은 제외하고 이유와 수를 기록한다. 완료된 activation은 한 문항의 한 레이어씩 `load_layer`로 복원하며, checksum·토큰 범위 검증이 실패하면 경로를 출력하고 중단한다. 원본 파일이 분석 중 바뀌어도 중단한다. 입력 데이터는 수정하지 않는다.

문제 ID 또는 공백 정규화 후 같은 문제 텍스트는 전이적 관계까지 묶어 같은 분할에 둔다. 클래스 비율을 유지하도록 train/validation/test를 60/20/20에 가깝게 고정한다. 그룹 크기와 소수 클래스 수 때문에 정확한 비율을 만들 수 없으면 실제 분할 수를 보고한다. 각 클래스가 최소 세 개의 독립 문제 그룹에 존재하지 않으면 학습하지 않는다. 템플릿·의미적 유사성까지 자동으로 탐지하는 것은 아니다.

| 방법 | 특징 | 시점 |
| --- | --- | --- |
| 확률 기준선 | 응답 평균·최대 entropy, 평균·최소 log probability, top-1/top-2 확률 차이의 평균·최소; 개별 점수와 결합 로지스틱 회귀 | 사후 |
| 선형 probe | 모든 residual 레이어의 프롬프트 마지막 토큰 | 생성 전 |
| 선형 probe | 모든 residual 레이어의 응답 마지막 토큰·응답 평균 | 사후 |
| MLP | 생성 전·사후 각각 validation에서 선택한 선형 probe와 동일한 레이어·집계 특징 | 해당 특징과 동일 |

응답 특징은 EOS를 포함한다. top-1/top-2 확률 차이는 저장된 두 log probability를 지수화한 후 뺀 값이다. 개별 점수는 entropy가 클수록, log probability·확률 차이가 작을수록 오답 점수가 높다. 개별 점수는 보정된 확률이 아니므로 예측의 `error_probability`는 `null`이고 `error_score`를 사용한다. 학습된 로지스틱 회귀·MLP는 오답 확률을 출력한다.

표준화는 train에서만 학습한다. 로지스틱 회귀의 `C` 후보는 0.01, 0.1, 1, 10이다. 후보는 validation AUROC, AUPRC, 고정 후보 순서로 선택한다. 생성 전·사후 선형 probe는 각각 선택하고, MLP는 각 선형 probe의 특징을 그대로 사용한다. MLP는 은닉층 32, ReLU, Adam, 학습률 0.001, 배치 32, 최대 200 epoch이며 validation BCE loss가 20 epoch 동안 개선되지 않으면 중단하고 최선 상태를 복원한다. 임계값은 validation의 오답 F1이 최대인 값, 동률이면 가장 높은 값으로 고정한다. Test를 보고 모델이나 임계값을 다시 선택하지 않는다.

```text
analysis/pilot/
  analysis_manifest.json     # 입력 snapshot/hash, 제외 이유, 분할, 설정, 실제 패키지 버전
  candidates.json            # 모든 후보의 validation 지표
  metrics.json               # 선택 결과, test 지표, 숫자 추출 성공 보조 지표
  test_predictions.json      # 실제 정오, 오답 점수/확률, 임계값, 판정
  report.md                  # 비교 보고서
  models/*.pt                # 선택 모델·표준화 값·임계값 (CPU, weights_only로 읽기)
```

AUROC, 오답 AUPRC(average precision), precision/recall/F1, confusion matrix와 클래스별 문항 수를 보고한다. 숫자를 정상 추출한 비잘림 test 문항만의 보조 결과는 같은 모델·임계값을 사용한다. 단일 클래스·빈 보조 집합의 ranking 지표, 분모가 0인 분류 지표는 `null`과 사유를 기록한다. AUPRC는 오답 클래스 비율과 함께 읽어야 한다. 결과는 해당 고정 분할의 탐색 결과이며 통계적 유의성이나 다른 과제에서의 일반화를 입증하지 않는다.

저장 모델로 개별 문항을 다시 판별할 수 있다. `metrics.json`의 `artifact`에 있는 실제 모델 경로를 지정한다.

```python
from pathlib import Path
import json
from analyze import predict_item

output = Path("analysis/pilot")
metrics = json.loads((output / "metrics.json").read_text())
name = metrics["selections"]["post_generation_linear"]
model_path = output / metrics["models"][name]["artifact"]
result = predict_item(model_path, "pilot/items/train_00042")
print(result)  # error_score, error_probability, threshold, predicted_error
```

예측 입력에 실제 정답·`correct`·과정 주석을 사용하지 않는다. 학습 때의 모델·수집 규격과 맞는 파일을 사용한다. 출력 디렉터리가 비어 있지 않으면 덮어쓰지 않으며 새 경로를 지정해야 한다. `analysis/`는 생성 결과이므로 Git에서 제외한다.

`test_analyze.py`는 확률 계산·토큰 경계·그룹 분할·데이터 부족·train 전용 표준화·test 정보 누출 방지·validation 선택·단일 클래스 지표·손상 파일 중단·저장 모델 예측 재현을 작은 합성 수집 파일로 검증한다.
