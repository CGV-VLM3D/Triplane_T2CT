# VLM3D ctgen 챌린지 제출 (`report2ct_wan`)

`report2ct_wan`을 forithmus `ct-volume-generation / main-2026`에 제출하기 위한 도커 빌드 컨텍스트와
런북. 로컬 채점은 [../docs/ctgen_local_eval.md](../docs/ctgen_local_eval.md), 플랫폼 일반 절차는
[../docs/ctgen_challenge_submission.md](../docs/ctgen_challenge_submission.md).

**연구 파이프라인은 이 디렉토리 밖을 건드리지 않는다.** `src/eval/samplers/`와
`scripts/decode_wan_latents.py`는 그대로 int16 `.mha`를 쓴다; float32 `.nii.gz`는 여기서만 쓴다.

## 제출 구성

| 항목 | 값 | 근거 |
|---|---|---|
| 모델 | `report2ct_wan` ep299 | mask 없이 텍스트만으로 동작 (아래 "왜 mask 모델이 아닌가") |
| spacing | `0.75 / 0.75 / 1.3` | [[spacing-optimum-final-n300]] |
| cfg_scale | `5.0` | FVD_CTCLIP proxy 최우수 (0.369) |
| n_steps | `50` | 100 대비 총비용 −24 %, CLIP은 스텝에 둔감 ([[report2ct-sampling-steps]]) |
| 출력 | `/output/<stem>.nii.gz`, 512×512×253 **float32** | 스키마가 dtype만 검사 (아래) |
| 케이스 수 | 3024 | |

### 왜 mask 모델(`report2ct_wan_mask_v2`)이 아닌가
헤드라인 모델은 생성에 GT CT로부터 TotalSegmentator로 만든 organ mask latent(`--mask-dir`)를
요구한다. 챌린지 입력은 `/input/prompts.json`(리포트 텍스트)뿐이고 CT를 주지 않으므로 그대로는
제출할 수 없다. 후속 옵션은 (a) 2-pass(생성→세그먼트→재생성), (b) 리포트 최근접 학습 케이스의
마스크 검색.

### 왜 float32인가
플랫폼 출력 스키마는 GenerateCT 베이스라인에서 자동 감지돼 `float32`를 기대한다. `forithmus validate`
결과 **거부 사유는 dtype 하나뿐**이고 shape 512×512×253 / spacing 0.75는 통과한다. `process.py`는
HU를 int16으로 반올림한 뒤 float32로 넓혀 저장하므로 **복셀 값은 로컬에서 검증한 `.mha`와 동일**하다.

## 파일

| 파일 | 역할 |
|---|---|
| `process.py` | 컨테이너 진입점. prompts.json → 생성 → Wan VAE 디코드 → `.nii.gz` + `/checkpoint` 재개 |
| `report_parse.py` | `"{age}-year-old {sex}: Findings: … Impression: …"` → `(findings, impression)` |
| `tests/test_report_parse.py` | validation_reports.csv 3039건 전수 라운드트립 |
| `stage_weights.sh` | `weights.zip`(4.0 GB) + `weights_stage/` 생성 |
| `Dockerfile` / `entrypoint.sh` / `requirements.txt` | thin 이미지 |
| `../.dockerignore` | 빌드 컨텍스트를 86파일 / 0.6 MiB로 제한 |

## 런북

### 1. 가중치 패키징 — 이 컨테이너
```bash
cd /workspace
bash vlm3d_challenge_eval/stage_weights.sh
# → data/vlm3d_submission/weights.zip        (4.0 GB, 제출용)
# → data/vlm3d_submission/weights_stage/     (드라이런 바인드마운트용)
```

### 2. 이미지 빌드 — 호스트 셸
컨테이너 안에는 docker 데몬이 없다. `/workspace`는 호스트의
`/user/woojin/Triplane_T2CT` 바인드마운트이므로 호스트에서 그대로 빌드된다.

⚠ **소유권**: 이 컨테이너는 root로 돌아서 여기서 만든 디렉토리는 `root:root`가 되고, 호스트
사용자(uid 1000)는 그 안에 파일을 못 만든다(`docker save … > …tar.gz`가 permission denied).
호스트가 쓸 디렉토리는 컨테이너에서 미리 `chown 1000:1000` 해둔다 — `data/vlm3d_submission`과
`dryrun_out`은 이미 해뒀다. 같은 이유로 드라이런의 `/output` 바인드마운트도 uid 1000이
쓸 수 있어야 한다(이미지가 `USER 1000`으로 돈다).
```bash
cd /user/woojin/Triplane_T2CT
docker build -f vlm3d_challenge_eval/Dockerfile -t vlm3d_challenge_eval_test:latest .
docker save vlm3d_challenge_eval_test:latest | gzip > data/vlm3d_submission/submission.tar.gz
```

### 3. 드라이런 — 호스트 셸
⚠ **`forithmus test`는 쓸 수 없다** — `--timeout`만 있고 weights를 마운트하는 옵션이 없어서
thin 이미지가 `/weights` 없이 ckpt 로드에서 죽는다. docker를 직접 돌린다.
```bash
cd /user/woojin/Triplane_T2CT
forithmus generate                       # .forithmus/test_data/{input,expected_output}
mkdir -p data/vlm3d_submission/dryrun_out
docker run --rm --gpus all \
  -v $PWD/.forithmus/test_data/input:/input:ro \
  -v $PWD/data/vlm3d_submission/weights_stage:/weights:ro \
  -v $PWD/data/vlm3d_submission/dryrun_out:/output \
  -v $PWD/data/vlm3d_submission/dryrun_ckpt:/checkpoint \
  vlm3d_challenge_eval_test:latest
```
`/checkpoint`를 마운트하지 않으면 컨테이너가 만들지 못해 재개가 비활성(경고 후 정상 진행)된다.
**재개를 실제로 확인하려면**: 위를 한 번 돌린 뒤 `rm -f dryrun_out/*.nii.gz`(플랫폼이 `/output`을
비우는 동작)하고 다시 돌려서 `Resuming from checkpoint: N recorded, N restored`가 뜨는지 본다.
mock 리포트는 `"sample_report"`라 마커가 없다 — 파서가 경고를 찍고 전체 텍스트를 findings로
쓰는 폴백이 정상 동작이다. 5볼륨 생성 후:
```bash
forithmus validate data/vlm3d_submission/dryrun_out --input .forithmus/test_data/input
# 기대: "Output looks perfect!"
```
확인할 것: 비-root(`USER 1000`)로 뜨는가, `torch.cuda.is_available()` 로그가 `True`인가,
볼륨당 초가 얼마인가.

### 4. 처리율 기준선 (캘리브레이션 런은 돌리지 않았다)
유휴 GPU에서 **15.3 s/vol** — generate 4.5 s + decode 10.8 s — 이고 컨테이너 드라이런 실측
14.75 s/vol과 일치한다(컨테이너 오버헤드 사실상 0). 3024개면 우리 하드웨어로 약 **12.6 h**.

L4가 몇 배 느릴지는 **모른다.** 디코드가 메모리 대역폭 바운드라(fp16으로도 전혀 빨라지지 않음)
연산 성능비로 외삽할 수 없다. 추정 3–6배 → 38–77 h.

별도의 1시간 캘리브레이션 런을 돌리려다 접었다. 미사용분이 환불되고 checkpoint 재개가 실제로
동작하므로, 예약을 넉넉히 잡는 것만으로 같은 리스크가 덮인다 — 측정만을 위한 런을 살 이유가 없다.

⚠ **성능을 잴 땐 GPU가 비어 있는지 먼저 확인할 것.** 경합 중인 GPU에서 잰 초기 수치는
2배 부풀려져 있었다(33 s/vol). `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv`.

### 5. 본 제출 — 웹 UI, spot
<https://research.forithmus.com/challenges/ct-volume-generation#submit>

**CLI가 아니라 웹 UI를 쓰는 이유**: CLI 0.1.10(PyPI 최신)에 `--spot` 플래그가 없다. L4 spot은
$0.417/hr로 on-demand $0.85 대비 **−51 %**(폼에 표시된 7200분 = $50에서 역산; CPU 티어에서 본
−61 %는 티어마다 다르다). 우리 워크로드는 볼륨 단위 산출물 + checkpoint 재개라 선점에 강하다.

| 폼 항목 | 값 |
|---|---|
| Compute tier | **L4** (24 GB VRAM) + **spot** 선택 |
| Time budget | **5400분 (90 h)** |
| Model weights | Upload new → `data/vlm3d_submission/weights.zip` (4.1 G) |
| Docker image | `data/vlm3d_submission/submission.tar.gz` (6.6 G) |
| Description | 리더보드에 표시될 알고리즘 이름 |

**예약 시간 산정.** 웹 폼 실측 단가는 L4 spot **$0.417/hr**(7200분 = $50). 예약분이 곧
선결제액이지만 미사용분은 자동 환불되므로, 실제 청구는 쓴 시간만큼이다(중심 추정 62 h ≈ $26).

| 예약 | 선결제 | 커버 배율 |
|---|---|---|
| 120 h | $50 | 9.3× |
| **90 h** | **$37** | 7.0× |
| 80 h | $33 | 6.2× |
| 65 h | $27 | 5.0× |

checkpoint 재개가 동작하므로 **예약이 최악의 경우를 커버할 필요는 없다** — 모자라면 Continue를
누르면 되고 진행분은 보존된다. 다만 재개가 공짜는 아니다(2000볼륨 시점이면 174 GB를 `/output`으로
되복사하느라 30분쯤 과금). 그래서 한 번에 끝날 확률이 높되 예약금이 과하지 않은 **90 h**를 택했다.
Continue는 계획이 아니라 보험이다.

재제출 시 가중치가 같으면 `--reuse-weights <성공한 이전 submission id>` / 웹의 "Reuse previous".

## T4 전환 옵션 (측정만 해둠, 미적용)

`gpu-t4`($0.65/hr, spot $0.333)는 L4($0.85, spot $0.417)보다 싸고 **VRAM도 충분하다**(피크
reserved 9.6 GiB < 16 GB). 막는 것은 T4가 Turing이라 **bf16 미지원**이라는 점 하나뿐이다.

**fp16은 안전하다 (실측, 같은 시드 / 유휴 GPU 0):**

| autocast | 시간 | non-finite | bf16 대비 |
|---|---|---|---|
| bf16 (현재) | 5.03 s | 0 | — |
| fp16 | 4.30 s | 0 | mean 6.4 HU, max 1362 HU |
| fp32 (off) | 5.34 s | 0 | mean 6.8 HU, max 1390 HU |

fp16의 bf16 대비 차이가 **fp32의 bf16 대비 차이보다 작다** → 이 차이는 fp16 불안정이 아니라
50-step 확산 샘플링의 경로 발산이다. 오버플로우 없음.

**그런데 이득이 작다.** 디코드가 대역폭 바운드인데 T4 320 GB/s > L4 300 GB/s라 디코드는 대등하고,
연산 바운드인 생성에서만 T4가 약 2배 뒤진다 → 볼륨당 T4 ~77 s vs L4 ~74 s(거친 외삽).
실제 청구 예상 T4 ~$22 vs L4 ~$26, **차이 약 $4**. 1.5배만 더 느려도 부호가 뒤집힌다.
재빌드 + 6.6 GB 재업로드 값어치가 아니라고 판단해 **L4로 갔다.**

전환한다면 dtype을 하드코딩하지 말고 런타임에 고를 것 — 같은 이미지가 모든 티어에서 돌고,
"dtype 바꾸는 걸 깜빡하고 제출" 사고가 구조적으로 사라진다:
```python
_AUTOCAST_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
```

## 운영 메모

- **⚠ `/output`은 런 사이에 비워진다. `/checkpoint/`만 살아남는다** (플랫폼 Checkpointing 문서).
  타임아웃(→ "Continue")이든 spot 선점 자동 재시도든 동일하다. 그래서 `process.py`의 `Checkpoint`가
  볼륨마다 `/checkpoint/outputs/`로 사본을 남기고 `progress.json`에 기록하며, 시작 시 `/output`으로
  되돌린다. **"출력 파일이 있으면 skip"만으로는 아무것도 재개되지 않는다** — 그 전제가 틀렸다.
- **checkpoint 백업은 best-effort다.** 출력 총량이 3024 × 87 MiB ≈ **257 GiB**인데 플랫폼은
  `/checkpoint`를 teardown 시 GCS로 올리는 로컬 디스크로 설명한다 — 그보다 작을 수 있다.
  용량이 차면 크게 로그를 남기고 **런은 계속한다**(볼륨은 여전히 `/output`에 쓰이고 채점된다;
  잃는 것은 그 이후의 재개 가능성뿐). 볼륨 2000번에서 디스크 풀로 죽는 쪽이 훨씬 손해다.
  시작 시 `/checkpoint` 여유 용량을 로그로 남기니 **첫 런에서 실제 용량을 알 수 있다.**
- 기록 순서는 `write /output → 백업 복사 → progress.json 갱신`이라, 복사가 중간에 끊긴 백업은
  절대 기록되지 않는다 → 복원이 반쪽 파일을 되살리는 일이 없다.
- **재개 방식**: 이미 있는 출력은 skip. 쓰기가 원자적이지 않아서(gcsfuse에 쓸 만한 동일-디렉토리
  스테이징이 없고 — 채점기 `_list_volumes`가 `rglob`이라 하위 디렉토리도 채점된다 — NiftiImageIO는
  비-NIfTI 임시 확장자를 거부한다) **시작 시 가장 최근 출력 1개를 버리고 재생성**한다. 중간에
  잘릴 수 있는 파일은 정의상 그것 하나뿐이다.
- **쓰기 스레드**: gzip 저장이 볼륨당 7.8 s의 CPU 작업이고 GPU 등급과 무관하게 고정이라,
  워커 1개로 다음 볼륨 생성과 겹친다 — 유휴 GPU 기준 직렬이면 15.3+7.8 = 23 s가 될 것을
  15 s로 만든다(GPU 시간이 쓰기보다 길어 쓰기가 완전히 숨는다). 항상 최대 1개의 쓰기만
  진행 중이므로 위의 "최신 1개 폐기" 전제가 유지된다.
- **로그 노이즈**: `third_party/maisi_bundle/scripts/rectified_flow.py:117`에 stray `print`가 있어
  볼륨마다 timestep 텐서를 찍는다. `third_party`는 read-only(P2)라 두었다.
- **출력량**: 볼륨당 87 MiB × 3024 ≈ 260 GB가 제출의 predictions/ prefix에 쌓인다(로컬 아님).

## 재검증 (코드 변경 시)

```bash
# 파서
HF_HOME=/workspace/data/checkpoints/hf_cache \
  python -m pytest vlm3d_challenge_eval/tests/ -q

# 최소 의존성으로 실제 생성 (Dockerfile이 하는 일과 동일한 구성)
/opt/conda/bin/python -m venv /workspace/data/vlm3d_submission/venv_probe
V=/workspace/data/vlm3d_submission/venv_probe/bin
$V/pip install --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
$V/pip install --no-cache-dir -r vlm3d_challenge_eval/requirements.txt
$V/pip install --no-cache-dir --no-deps monai==1.4.0        # numpy<2.0 핀이 낡음
CUDA_VISIBLE_DEVICES=3 \
  HF_HUB_CACHE=/workspace/data/vlm3d_submission/weights_stage/hf_cache/hub HF_HOME=/tmp/hf \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  FORITHMUS_INPUT=<prompts dir> FORITHMUS_OUTPUT=<out> \
  FORITHMUS_WEIGHTS=/workspace/data/vlm3d_submission/weights_stage \
  $V/python vlm3d_challenge_eval/process.py
```

`/weights`가 read-only이고 컨테이너가 비-root로 도는 것까지 재현하려면 위를
`su nobody -s /bin/bash -c "…"`로 감싸고 `weights_stage`를 `chmod -R a-w` 한다 —
이 조합에서만 드러나는 실패(HF 락 파일 쓰기 시도)를 잡는다.
