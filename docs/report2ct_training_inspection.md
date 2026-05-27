# Report2CT — 학습 코드 가이드 (읽기 + Inspect)

이 문서는 **Report2CT를 직접 학습시키기 전에 무엇을 읽고 무엇을 확인해야 하는지**를 정리합니다. 학습 launcher 자체는 아직 만들지 않았어요 (Phase B Day 1 [A] 작업). 이 문서는 그 작업 이전에 **현재 상태를 파악하는 길잡이**입니다.

---

## 1. 한 줄로 본 학습 구조

```
Step 1. Text encoders (3개)   → per-sample JSON에 findings + impression embedding 저장
Step 2. MAISI VAE (frozen)    → per-volume *_emb.nii.gz 저장
Step 3. torchrun training     → DiffusionModelUNetMaisi가 (image_emb, text_emb, spacing)로 RFlow noise prediction 학습
```

**중요**: 학습 코드 자체는 우리가 안 짭니다. submodule (`third_party/report2ct`)의 학습 스크립트를 그대로 torchrun으로 invoke. 우리가 채워야 할 건 (a) precompute 스크립트, (b) 우리 데이터 경로 반영한 env config, (c) launcher 쉘.

---

## 2. 어떤 파일을 읽어야 하나

### 2.1 ★ 우선순위 1 — 학습 진입점 (10분)

| 파일 | 분량 | 무엇을 알 수 있나 |
|---|---|---|
| `third_party/report2ct/train.sh` | 10줄 | 학습 명령 1줄 (torchrun + 3 config 인자) |
| `third_party/report2ct/vlm3D_work_dir/config_maisi_diff_model_vlm3D.json` | 8줄 | **학습 schedule**: batch=2, lr=1e-4, epochs=100, cache_rate=0 |
| `third_party/report2ct/vlm3D_work_dir/config_maisi_2560.json` | ~190줄 | **모델 아키텍처**: DiffusionModelUNetMaisi(channels [64,128,256,512], cross-attn dim 2560, attention at last 2 levels), RFlow scheduler (1000 steps, scale 1.4) |
| `third_party/report2ct/vlm3D_work_dir/environment_maisi_diff_model_vlm3D_FI_2560_multi.json` | (확인 필요) | **경로 설정** — 이게 우리 데이터 위치에 맞춰 수정돼야 함 |

### 2.2 ★ 우선순위 2 — 학습 루프 본체 (15분)

`third_party/report2ct/src/maisi/scripts/diff_model_train_vlm3D_2560_multi_text.py` (전체 ~500줄)

읽을 때 주목 포인트:
- L60-100: 데이터 transform 정의 (Lambdad로 `*multi_2560.json`에서 findings/impression embedding 로드)
- L270-300: forward + loss (context_f + context_i concat → UNet → RFlow noise prediction)
- L297: **CFG dropout p=0.15** (`context = torch.zeros_like(context)`)
- L435-480: 데이터 list 빌드 (filename → 이미지 emb + JSON metadata pair)

### 2.3 ★ 우선순위 3 — Precompute 두 단계

| Step | 파일 | 분량 | 의미 |
|---|---|---|---|
| Text encoders | `third_party/report2ct/vlm3d_inference.ipynb` **cell 0** | 30줄 | 3개 HF AutoModel 로드, 각 report → pooled embedding → 2560-d → JSON 저장 |
| Image embeddings | `third_party/report2ct/src/maisi/scripts/vlm3d_image_embedding.py` | ~250줄 | MAISI VAE encode + sliding window → `*_emb.nii.gz` 저장 |

### 2.4 우리가 추가/수정한 부분 (Phase A 산출물)

| 파일 | 역할 | 학습과의 관계 |
|---|---|---|
| `src/baselines/report2ct_adapter.py` | UNet LightningModule 래퍼 | **학습에 사용 X**. Phase B 진단(cross-attn hook)에서 forward 호출용 |
| `tests/test_report2ct_module.py` | UNet build + 1-step shape 검증 | 학습 전에 모델 정의 sanity 확인 |
| `docs/report2ct_external_components.md` | 3 text encoder HF id 핀 | precompute 스크립트 작성 시 참조 |
| `docs/report2ct_training_handoff.md` | 학습 runbook | 명령 + 결정 게이트 |
| `.omc/plans/report2ct_impl_spec.md` | paper read + submodule wrap strategy | 큰 그림 |
| `results/report2ct_envelope.json` | 학습 후 sanity 비교 target | 학습 결과 평가 기준 |

---

## 3. 원본 vs 추가 (모듈별 한눈에)

```
third_party/report2ct/              ← 원본, READ-ONLY (Principle P2)
├── src/maisi/scripts/
│   ├── diff_model_train_vlm3D_2560_multi_text.py   ← 학습 루프 (그대로 invoke)
│   ├── vlm3d_image_embedding.py                    ← 이미지 emb precompute (그대로 invoke)
│   ├── diff_model_setting.py                       ← config 로더 helper (그대로 사용)
│   └── utils.py                                    ← (그대로 사용)
├── vlm3D_work_dir/                                 ← config JSON 3종 (그대로 사용)
│   ├── config_maisi_2560.json                      ← 모델 아키텍처
│   ├── config_maisi_diff_model_vlm3D.json          ← 학습 schedule
│   └── environment_maisi_diff_model_vlm3D_FI_2560_multi.json  ← 경로 (※ 수정 필요)
├── train.sh                                        ← torchrun 1줄 (참고용; 우리는 own launcher 만듦)
└── vlm3d_inference.ipynb                           ← cell 0의 text encoder 로직 (그대로 차용)

[/workspace/src/] ours:
├── baselines/
│   ├── maisi.py                ★ Day 2 (FROZEN VAE loader, bundle 그대로)
│   ├── report2ct_adapter.py    ★ Day 4 (UNet wrapper, Phase B 진단용 — NOT 학습용)
│   └── generatect_adapter.py   ★ Day 3 (GenerateCT용 — Report2CT와 별개)
├── data/
│   └── ct_rate_datamodule.py   ★ Day 2 (메타데이터 join, mode='metadata' 만)
└── (학습 launcher / precompute scripts 아직 없음 ← Phase B Day 1)

[/workspace/scripts/] ours:
└── run_eda.py                  (EDA 실행, 학습과 무관)

[/workspace/configs/experiment/] ours:
└── example.yaml                (template; report2ct_repro.yaml 아직 없음 ← Phase B Day 1)
```

**한 줄 정리**: Phase A까지는 (1) MAISI VAE loader, (2) UNet adapter (진단용), (3) DataModule (metadata-only), (4) docs/configs/tests 만 우리 코드. 학습 launcher + precompute 스크립트는 Phase B Day 1에 들어옴.

---

## 4. 현재 학습 셋팅 inspect 명령어

### 4.1 학습 schedule (epochs, lr, batch) 확인

```bash
cat /workspace/third_party/report2ct/vlm3D_work_dir/config_maisi_diff_model_vlm3D.json
# → "diffusion_unet_train": {"batch_size": 2, "cache_rate": 0, "lr": 0.0001, "n_epochs": 100}
```

### 4.2 모델 아키텍처 정의 확인 (paper와 일치하는지)

```bash
python - <<'PY'
import json, textwrap
cfg = json.load(open('/workspace/third_party/report2ct/vlm3D_work_dir/config_maisi_2560.json'))
unet = cfg['diffusion_unet_def']
sched = cfg['noise_scheduler']
print("UNet target:", unet['_target_'])
print("  num_channels :", unet['num_channels'])
print("  attn_levels  :", unet['attention_levels'])
print("  cross_attn_dim:", unet['cross_attention_dim'])
print("  num_head_chan:", unet['num_head_channels'])
print("  flash_attn   :", unet['use_flash_attention'])
print("Scheduler:", sched['_target_'])
print("  steps        :", sched['num_train_timesteps'], "scale", sched['scale'])
PY
```

### 4.3 학습 launcher (torchrun) 명령 확인

```bash
cat /workspace/third_party/report2ct/train.sh
```

이게 우리가 만들 `scripts/run_report2ct_training.sh`의 base 예제.

### 4.4 학습 환경 config (경로) 확인 — ※ 가장 중요한 inspect

```bash
cat /workspace/third_party/report2ct/vlm3D_work_dir/environment_maisi_diff_model_vlm3D_FI_2560_multi.json
```

실제 내용:
```json
{
  "data_base_dir":               "./vlm3D_work_dir/sim_dataroot",
  "embedding_base_dir":          "./vlm3D_work_dir/embeddings",
  "json_data_list":              "./vlm3D_work_dir/sim_datalist.json",
  "model_dir":                   "./vlm3D_work_dir/models",
  "model_filename":              "diff_unet_ckpt_FI_2560_multi.pt",
  "output_dir":                  "./vlm3D_work_dir/predictions",
  "output_prefix":               "unet_3d",
  "trained_autoencoder_path":    "./vlm3D_work_dir/models/autoencoder_epoch273.pt",
  "existing_ckpt_filepath":      "./vlm3D_work_dir/models/diff_unet_ckpt_FI_2560_multi_best.pt"
}
```

**Phase B Day 1에서 우리가 갈아 끼워야 할 항목들**:
- `data_base_dir` → `/workspace/datasets/datasets/CT-RATE/dataset/train_fixed/` (CT-RATE 원본)
- `embedding_base_dir` → `/workspace/data/report2ct_embeddings/image/` (우리 precompute 출력)
- `json_data_list` → `/workspace/data/report2ct_embeddings/datalist.json` (우리가 만들 sample list)
- `model_dir` → `/workspace/data/checkpoints/report2ct/`
- `output_dir` → `/workspace/runs/report2ct_repro/`
- **`trained_autoencoder_path`** → `/workspace/third_party/maisi_bundle/models/autoencoder.pt`
  (※ 주의: submodule은 `autoencoder_epoch273.pt` 라는 다른 이름을 기대. **같은 파일인지 확인이 필요**. MAISI 공식 bundle의 `autoencoder.pt`가 paper에서 말하는 ckpt와 동일하면 path만 override하면 OK.)
- `existing_ckpt_filepath` → 무시 (없는 경로면 학습이 from-scratch로 시작)

### 4.5 UNet 자체가 정상 build / 1-step forward 되는지 (이미 통과한 테스트)

```bash
pytest /workspace/tests/test_report2ct_module.py -v
# 4 tests:
#   ✅ test_build_unet_from_submodule_config (233M params 검증)
#   ✅ test_forward_one_batch (1-step shape invariance)
#   ⏸  test_overfit_one_batch (skip — 실제 학습 launcher 필요)
#   ✅ test_adapter_lightning_module_forward (Report2CTAdapter LightningModule)
```

### 4.6 우리 adapter에서 UNet param 개수 + forward 시연

```bash
python - <<'PY'
from src.baselines.report2ct_adapter import build_unet, forward_one_step
import torch
unet = build_unet().eval()
n = sum(p.numel() for p in unet.parameters())/1e6
print(f"UNet params: {n:.1f}M")
print(f"trainable params: {sum(p.numel() for p in unet.parameters() if p.requires_grad)/1e6:.1f}M")
latent  = torch.randn(1, 4, 16, 16, 8)
context = torch.randn(1, 2, 2560)
spacing = torch.tensor([[1.0, 1.0, 1.5]])
with torch.no_grad():
    out = forward_one_step(unet, latent, context, spacing, timestep=500, class_label=1)
print(f"in: {tuple(latent.shape)} out: {tuple(out.shape)} match: {out.shape==latent.shape}")
PY
```

### 4.7 MAISI VAE가 frozen인지 (학습 시 안 깨지는지)

```bash
pytest /workspace/tests/test_maisi_frozen_load.py -v
# 4 tests, 모두 ✅. 핵심: requires_grad=False 모든 param.
```

### 4.8 학습용 GPU 가용성 + VRAM

```bash
nvidia-smi
# 3× A6000 Blackwell 96GB 확인. Report2CT는 2 GPU torchrun 권장.
```

### 4.9 학습 데이터 경로 (CT-RATE) 마운트 확인

```bash
ls /workspace/datasets/datasets/CT-RATE/dataset/train_fixed/ | wc -l
# 20,000 (대략) — 학습 train_fixed 환자 디렉토리 수
ls /workspace/datasets/datasets/CT-RATE/dataset/valid_fixed/ | wc -l
# 1,304
```

### 4.10 envelope (학습 후 비교 기준) 확인

```bash
cat /workspace/results/report2ct_envelope.json | python -m json.tool
# 어떤 metric에 무슨 anchor가 잡혀있는지 한눈에.
```

---

## 5. 정리: 학습 시작하려면 무엇이 더 필요한가

체크리스트 (Phase B Day 1 [A] 작업):

- [ ] `scripts/precompute_report2ct_text_embeddings.py` — `vlm3d_inference.ipynb` cell 0 로직을 .py로 옮겨서 CT-RATE 전체 report(train_reports.csv + validation_reports.csv) → `<image_name>multi_2560.json` 파일 생성. 출력 위치: `/workspace/data/report2ct_embeddings/text/`.
- [ ] `scripts/precompute_report2ct_image_embeddings.py` — submodule의 `vlm3d_image_embedding.py`를 우리 경로로 invoke. 출력 위치: `/workspace/data/report2ct_embeddings/image/`. 약 60 GB.
- [ ] `vlm3D_work_dir_custom/environment.json` — submodule env config 복사 후 경로를 우리 `/workspace/data/...`로 갈아끼움 (submodule은 read-only이므로 별도 위치에).
- [ ] `scripts/run_report2ct_training.sh` — torchrun 명령. 우리 env config + submodule의 model/training config 사용.
- [ ] `configs/experiment/report2ct_repro.yaml` — Hydra 측에서도 메타데이터 추적용 (학습 자체는 submodule script가 함).
- [ ] **6/1 compute-measurement gate**: 위 launcher로 100-sample subset 1-epoch 돌려서 wall-clock 측정 → `.omc/plans/phase_b_budget.md` 작성.

위 5개 파일을 만들면 `bash scripts/run_report2ct_training.sh` 한 줄로 학습 시작 가능. 작성에 반나절 정도 예상.

---

## 6. 더 깊이 알고 싶으면

- **Paper**: `paper_pdf/Report2CT.pdf` — Method 섹션 (3-text-encoder + voxel spacing + RFlow) 6-7페이지가 핵심.
- **Submodule 학습 루프**: `third_party/report2ct/src/maisi/scripts/diff_model_train_vlm3D_2560_multi_text.py` — 한 번 끝까지 훑으면 학습 흐름 완전 이해.
- **Our impl spec**: `.omc/plans/report2ct_impl_spec.md` — paper와 submodule 사이의 정합성, 미해결 open question 4개.
- **Training handoff**: `docs/report2ct_training_handoff.md` — TL;DR + 컴퓨트 현실 + sanity 보고 템플릿.
