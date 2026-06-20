# Experiment 3 — Saliency / Activation maps (global vs fine-grained encoder)

global encoder(**CT-CLIP**)와 fine-grained·anatomy-aware encoder(**fVLM**)가 주어진
abnormality 프롬프트에 어떻게 attend 하는지, 그리고 fVLM의 **anatomy-mask 의존성**을 시각화한다.
가설: text condition이 image space와 정렬될수록(global < fine-grained) 국소 병변에 더 잘 집중.

대비 점수 `S = cos(I, t_pos) − cos(I, t_neg)` 를 LayerCAM 으로 voxel 공간에 역투영 +
fVLM native token-text similarity 맵(보조). 격리 실험 — `tests/saliency_map/` 안에서 완결.

## 실행

```bash
# 빈 GPU 지정 (CTViT 는 cuda:0 하드코딩 → CUDA_VISIBLE_DEVICES 로 물리 GPU 선택)
CUDA_VISIBLE_DEVICES=0 python tests/saliency_map/run_saliency.py \
    --config tests/saliency_map/configs/fvlm_mask.yaml          # 본실행
CUDA_VISIBLE_DEVICES=0 python tests/saliency_map/run_saliency.py \
    --config tests/saliency_map/configs/ctclip.yaml --smoke     # 1 케이스 1 프롬프트
```

## arm (config 1개 = 1 arm)

| config | 의미 | 기대 |
|---|---|---|
| `ctclip.yaml` | CT-CLIP global | saliency 퍼짐 |
| `fvlm_mask.yaml` | fVLM organ center_crop + organ 토큰 (eval 충실, **default A**) | organ 집중 |
| `fvlm_nomask.yaml` | fVLM 전체볼륨 + 전체토큰 (마스크 제거) | 붕괴 |
| `fvlm_mask_whole.yaml` | fVLM 전체볼륨 + organ 토큰 (옵션) | crop 효과 격리 |

## config 키 (스키마: `config.py:SaliencyConfig`)

`exp_name, model(ctclip|fvlm), device, cam(layercam|gradcam|eigencam|eigengradcam|hirescam),
native_map, prompt_scheme(fvlm|ctclip), abnormalities, n_cases, positive_only, seed,
fvlm_crop(organ|whole), fvlm_mask(bool)`. 오타/미지의 키는 로드 시 에러.

## 출력 (`results/<exp_name>/`)

```
config.yaml              # 실행 config 스냅샷
run.log
index.json               # 케이스/프롬프트별 organ/score/pred_prob/grid
maps/<scan_id>/<name>.npy           # cam (D,H,W)
maps/<scan_id>/<name>_native.npy    # native (있으면)
overlays/<scan_id>/<name>.png       # CT + cam (axial+coronal)
overlays/<scan_id>/<name>_native.png
```

## 주의

- **CT-CLIP HU**: `valid_fixed` 는 HU 가 이미 구워져 있어 metadata slope/intercept 재적용 금지
  ([[ctrate-fixed-hu-no-rescale]]).
- **CT-CLIP VQ**: encode 토큰이 VQ 이후라 grad 차단 → straight-through estimator 로 우회.
- **fVLM eval 충실**: per-organ `center_crop_organ`+`divisible_pad_end`+`forward_test_win` 경로.
- `positive_only: true` + 여러 abnormality = 모두 양성(AND)인 케이스만. 보통 단일 abnormality 권장.
- `results/` 는 gitignore.
