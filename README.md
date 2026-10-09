# GSM8K 연구 데이터 수집

`Study_Plan.md`의 Qwen2.5-1.5B 파일럿을 위한 CLI다. 문제, 모델 풀이, 실제 정답, 최종 정오, 전체 토큰의 레이어별 activation을 저장한다. 과정 오류 판정·Judge 호출·probe 학습·복구 실험은 포함하지 않는다.

## Linux RTX 3090 설치·실행

Python 3.11과 CUDA 12.4용 PyTorch를 사용할 수 있는 NVIDIA 드라이버가 필요하다. 아래 명령은 **연구 작업 공간 안에서** 실행한다. 예를 들어 `/work/yaicon/research_base`에 이 폴더를 복사한 뒤 해당 디렉터리로 이동한다.

```bash
python3.11 -m venv .venv
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
