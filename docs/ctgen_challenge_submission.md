# ctgen 챌린지 제출 방법 (forithmus / VLM3D Task 4)

> forithmus 플랫폼에 ctgen(텍스트→3D CT) 도커를 제출하는 **방법 문서**.
> 로컬 채점은 [ctgen_local_eval.md](ctgen_local_eval.md) 참고.
> 제출 페이지: <https://research.forithmus.com/challenges/ct-volume-generation#submit>
> 참고 구현(베이스라인): `third_party/vlm3d_dockers/ct_challenges/ctgen_example_docker/` (GenerateCT cascade).
>
> ⚠️ **이 문서는 절차 정리용이다.** `forithmus login`/`submit`(과금) 실행과 **우리 자체 sampler
> (text2ct/report2ct) 제출 도커 빌드는 후속 작업**. 아래는 "어떻게 제출하는가"의 레퍼런스.

## 입출력 계약 (다른 트랙과 다름)

ctgen은 **CT 볼륨을 입력으로 받지 않는다.** 플랫폼이 텍스트 파일 하나만 마운트한다.

- 입력: `/input/prompts.json` — `{"input_image_name", "report"}` 객체의 JSON 배열
  ```json
  [ { "input_image_name": "<출력 볼륨 base name>", "report": "<생성 조건 텍스트(리포트)>" } ]
  ```
- 출력: `/output/<input_image_name>.nii.gz` — 프롬프트당 loose `.nii.gz` 1개. **zip 금지**
  (평가가 `.nii.gz`를 직접 읽음). 항상 `/output`을 먼저 생성.
- 체크포인트(선택): 시간초과 위험 시 `/checkpoint/`에 진행상황 저장 → 재시작 시 이어하기.

## 평가 & 랭킹 (서버측, 숨겨진 실제 CT와 비교)

- **1차 지표: `FVDCTNet`** (CT-Net 3D 백본 FVD, **낮을수록 좋음**).
- 함께 보고: `CT-CLIPScore`(I2T·I2I), `2.5D-FID`(XY/XZ/YZ).
- (로컬에선 FVD를 못 돌리므로 — [ctgen_local_eval.md](ctgen_local_eval.md)의 "FVD 주의" — 1차 지표는
  실제 제출 후 리더보드로만 확인됨을 유념.)

## 제출 절차

### 1. CLI 설치 & 로그인
```bash
pip install --upgrade forithmus      # >= 0.1.10 필요 (← 사용자가 로그인 수행)
forithmus login
```

### 2. 예제 + 로컬 테스트셋
```bash
git clone https://github.com/forithmus/VLM3D-Dockers.git && cd VLM3D-Dockers
forithmus init ct-volume-generation  # 현재 에디션(main-2026) 타깃
forithmus generate                   # 합성 prompts.json + 기대 출력 shape 생성(실제 리포트 아님, dry-run용)
```

### 3. 빌드 & 로컬 테스트
```bash
docker build -t ctgen-thin:latest ct_challenges/ctgen_example_docker/
forithmus test ctgen-thin:latest --timeout 1200   # 생성은 느림 → 타임아웃 넉넉히
```

### 4. 가중치 패키징 (thin-image 패턴)
제출 이미지는 **≤ 15 GB**. 모델 가중치는 분리 업로드한다. 파일은 `weights.zip` **루트에 평탄하게**
(부모 폴더 없이) 두고, 런타임에 `/weights`로 마운트 → entrypoint가 `/opt/app/models/`로 심링크.
```
weights.zip
 ├── ctvit_pretrained.pt
 ├── transformer_pretrained.pt
 └── superres_pretrained.pt
```
```bash
cd /path/to/raw_weights/ && zip -r0 ../weights.zip ./*   # -r0 = 무압축, 루트 평탄화
```

### 5. 제출
```bash
docker save ctgen-thin:latest | gzip > submission.tar.gz
forithmus submit submission.tar.gz \
    --phase main-2026 --tier gpu-a100-80 --time-budget 1200 \
    --weights weights.zip -d "알고리즘 이름(리더보드 표시)"
forithmus status
# 가중치 동일 재제출: --weights 대신 --reuse-weights <previous-submission-id>
```
- 생성이 무거우므로 큰 GPU tier(예 `gpu-a100-80`) + 넉넉한 `--time-budget` 권장.
- 예약된 시간만큼 선결제 후, 미사용분 환불.

## 컨테이너 요구사항 체크리스트
- **non-root**: Dockerfile에 `USER 1000`(예제 docker는 `useradd --uid 1000 user` 후 `USER user`). root 이미지는 거부됨.
- **CUDA base**: `FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` 류. plain `python:3.x`는 libcudart 부재로 GPU 비활성.
- **런타임 네트워크 없음**: HF/torch-hub 자산은 **빌드 타임에 pre-cache** + `TRANSFORMERS_OFFLINE=1`
  (`HF_HUB_OFFLINE=1`). 예제는 빌드 중 `google/t5-v1_1-base`·VGG16를 미리 받음.
- **fail loud**: 추론 루프를 `try/except`로 감싸 prompt를 조용히 건너뛰지 말 것. OOM은 크래시여야 함.
- **출력**: loose `.nii.gz`(zip 금지). 항상 `/output` 먼저 생성.
- `ENTRYPOINT`/`CMD` 필수, `linux/amd64`.

## 참고 구현 구조 (`ct_challenges/ctgen_example_docker/`)
- `Dockerfile` — thin(가중치 미포함). cu121 torch 먼저 설치 → `transformer_maskgit`/`super_resolution` editable → T5/VGG16 pre-cache.
- `entrypoint.sh` — `/weights/*.pt` → `/opt/app/models/`로 `ln -sfn` 후 `python process.py` 실행.
- `process.py` — `/input/prompts.json` 로드 → (Stage1 MaskGIT 저해상도 → Stage2 diffusion super-res) →
  `save` 시 HU clip→`SetSpacing` → `/output/<input_image_name>.nii.gz` loose 저장(zip 안 함).

## 우리 제출 (구현 완료 — `report2ct_wan`)
빌드 컨텍스트와 실행 런북은 **[../vlm3d_challenge_eval/README.md](../vlm3d_challenge_eval/README.md)**.
이 문서(절차 레퍼런스)와 달리 거기엔 실측치와 우리 구성이 들어 있다. 요약:

- `vlm3d_challenge_eval/process.py`가 `/input/prompts.json` → `report2ct_wan` RFlow(50 step,
  cfg 5.0, spacing 0.75/0.75/1.3) → Wan VAE decode → `/output/<stem>.nii.gz`를 한 프로세스로 처리한다
  (로컬의 3-step 플로우는 두 conda env의 diffusers 버전 차이 때문일 뿐이다).
- 출력은 **float32** `.nii.gz`다 — 스키마가 dtype만 검사하고 int16을 거부한다. shape 512×512×253과
  spacing 0.75는 통과한다. 값은 int16 반올림 후 float32로 넓혀 로컬 `.mha`와 동일하게 유지한다.
- 가중치는 `stage_weights.sh`가 만드는 `weights.zip`(4.0 GB: ckpt + HF 캐시)로 분리 업로드하고,
  `HF_HOME=/weights/hf_cache`로 가리켜 **entrypoint 심링크 없이** 오프라인 로드한다.
- ⚠ 헤드라인 `report2ct_wan_mask_v2`는 GT organ mask가 필요해 제출 불가 — README의 "왜 mask
  모델이 아닌가" 참고.
- ⚠ `forithmus test`는 weights 마운트 옵션이 없어 쓸 수 없다; 드라이런은 `docker run`으로 직접 한다.
