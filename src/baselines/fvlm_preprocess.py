"""fVLM 입력 전처리 + 텍스트 조건 빌더.

CT-RATE 스캔(+ TotalSegmentator 마스크)과 영상의학 보고서를, anatomy-aware
fVLM(`FVLMBackbone.forward_test_win` / `prepare_text_feat`)이 기대하는
`(image, mask)` 텐서와 `(organ, prompt_id, neg, pos)` test_items로 변환한다.

- 이미지/마스크: `load_ct_and_mask_for_local` — upstream 오프라인 4-스크립트
  파이프라인(fix_data→generate_mask→resize→preprocess)을 인메모리로 재현.
- 보고서 조회: `load_ctrate_report`.
- 텍스트 조건:
    - `build_decomposed_test_items_as_local` — 장기별 분해 보고서(올바른 경로,
      `src.data.fvlm_organ_report.build_organ_text` 출력과 짝).
    - `build_local_test_items` — 수동 dict 입력.
    - `DEFAULT_FVLM_TEST_ITEMS` — 표준 zero-shot 이상치 프롬프트 16종.

이 모듈은 fVLM 실연구 입력 준비(이미지/마스크 + 텍스트 조건)만 담당한다 —
saliency·시각화·CT-CLIP 경로는 포함하지 않는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

CT_RATE_ROOT: Final[Path] = Path("/workspace/datasets/datasets/CT-RATE/dataset")

# 장기 ID 규약: {lung:1, heart:2, esophagus:3, aorta:4}.
# third_party/fvlm/eval.py:135-138 의 장기 순서와 일치.
# 리포트 라이브러리에서 단일 소스로 관리하여 목록 불일치를 방지.
from src.data.fvlm_organ_report import ORGANS  # noqa: E402

ORGAN_TO_ID: Final[dict[str, int]] = {o: i + 1 for i, o in enumerate(ORGANS)}

# 표준 fVLM test_items. third_party/fvlm/eval.py:143-160 에서 그대로 가져옴.
# (총 16개: 폐 11개 + 심장 3개 + 식도 1개 + 대동맥 1개)
# 각 튜플 형식: (organ, prompt_id, neg_prompt, pos_prompt) —
# `prepare_text_feat`이 소비하는 형식.
DEFAULT_FVLM_TEST_ITEMS: Final[list[tuple]] = [
    ("lung", "Emphysema", "Not Emphysema.", "Emphysema."),
    ("lung", "Atelectasis", "Not Atelectatic.", "Atelectatic."),
    ("lung", "Lung nodule", "Not Nodule.", "Nodule."),
    ("lung", "Lung opacity", "Not Opacity.", "Opacity."),
    (
        "lung",
        "Pulmonary fibrotic sequela",
        "Not Pulmonary fibrotic.",
        "Pulmonary fibrotic.",
    ),
    ("lung", "Pleural effusion", "Not Pleural effusion.", "Pleural effusion."),
    (
        "lung",
        "Mosaic attenuation pattern",
        "Not Mosaic attenuation pattern.",
        "Mosaic attenuation pattern.",
    ),
    (
        "lung",
        "Peribronchial thickening",
        "Not Peribronchial thickening.",
        "Peribronchial thickening.",
    ),
    ("lung", "Consolidation", "Not Consolidation.", "Consolidation."),
    ("lung", "Bronchiectasis", "Not Bronchiectasis.", "Bronchiectasis."),
    (
        "lung",
        "Interlobular septal thickening",
        "Not Interlobular septal thickening.",
        "Interlobular septal thickening.",
    ),
    ("heart", "Cardiomegaly", "Not Cardiomegaly.", "Cardiomegaly."),
    (
        "heart",
        "Pericardial effusion",
        "Not Pericardial effusion.",
        "Pericardial effusion.",
    ),
    (
        "heart",
        "Coronary artery wall calcification",
        "Not Coronary artery wall calcification.",
        "Coronary artery wall calcification.",
    ),
    ("esophagus", "Hiatal hernia", "Not Hiatal hernia.", "Hiatal hernia."),
    (
        "aorta",
        "Arterial wall calcification",
        "Not Arterial wall calcification.",
        "Arterial wall calcification.",
    ),
]

# 장기 병합 매핑. fVLM 오프라인 전처리에서 그대로 파생하여 레이블 누락을 방지.
# (과거에 직접 손으로 만든 int→int 맵은 atrial_appendage_left / TS-61을 빠트려
# 심장 마스크가 ~1.3% 축소된 버그가 있었음.)
#
# DUPLICATION INTENTIONAL — 아래 두 dict는 third_party/fvlm/data/resize.py에서
# 그대로 복사:
#   _MERGED_ORGAN_ID ← resize.py:141-151 (장기명 → 0-based 병합 id)
#   _TS_CLASS_MAP    ← resize.py:21-139 (class_map, TS 레이블 → 장기명); 단,
#                      _MERGED_ORGAN_ID에 등장하는 항목만 포함 —
#                      upstream이 `if organ_name not in merged: continue`로 나머지를 버림.
# 디스크 저장 마스크 값은 merged_id + 1 (resize.py:156-161):
# 배경 0, {lung:1, heart:2, esophagus:3, aorta:4}.
_MERGED_ORGAN_ID: Final[dict[str, int]] = {
    "lung_upper_lobe_left": 0,
    "lung_lower_lobe_left": 0,
    "lung_upper_lobe_right": 0,
    "lung_middle_lobe_right": 0,
    "lung_lower_lobe_right": 0,
    "heart": 1,
    "atrial_appendage_left": 1,
    "esophagus": 2,
    "aorta": 3,
}
_TS_CLASS_MAP: Final[dict[int, str]] = {
    10: "lung_upper_lobe_left",
    11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
    15: "esophagus",
    51: "heart",
    52: "aorta",
    61: "atrial_appendage_left",
}
# int → 최종 1-based id. 위 두 upstream dict에서 파생하므로 손으로 목록을 관리하지 않음.
TS_LABEL_TO_ORGAN: Final[dict[int, int]] = {
    ts_id: _MERGED_ORGAN_ID[name] + 1 for ts_id, name in _TS_CLASS_MAP.items()
}

# fVLM 입력 규약 (eval.py:213-226 와 일치).
LOCAL_IMG_SIZE: Final[tuple[int, int, int]] = (
    112,
    256,
    352,
)  # preprocess.py SpatialPad 최소
FVLM_ROI_SIZE: Final[tuple[int, int, int]] = (
    112,
    288,
    352,
)  # eval.py:274 roi_size (장기별 center_crop)
FVLM_PATCH_SIZE: Final[tuple[int, int, int]] = (16, 16, 32)


# ---------------------------------------------------------------------------
# 경로 헬퍼
# ---------------------------------------------------------------------------


def ts_mask_path(ct_path: str | Path) -> Path:
    """CT-RATE 볼륨에 대한 미리 계산된 TotalSegmentator 마스크 경로 반환.

    CT-RATE 파일 규약::
        dataset/{train,valid}_fixed/<patient>/<study>/<volume>.nii.gz
    대응 마스크 경로::
        dataset/ts_seg/ts_total/{train,valid}_fixed/<patient>/<study>/<volume>.nii.gz
    """
    ct_path = Path(ct_path)
    parts = ct_path.parts
    if "dataset" not in parts:
        raise ValueError(f"CT path is not under a CT-RATE 'dataset/' root: {ct_path}")
    idx = parts.index("dataset")
    rel = Path(*parts[idx + 1 :])
    mask = Path(*parts[: idx + 1]) / "ts_seg" / "ts_total" / rel
    if not mask.is_file():
        raise FileNotFoundError(
            f"Pre-computed TotalSegmentator mask not found at:\n  {mask}\n"
            "If this is an OOD CT (not in CT-RATE), run TotalSegmentator manually."
        )
    return mask


def load_ctrate_report(
    ct_path: str | Path,
    *,
    include: str = "findings_plus_impression",
) -> str:
    """CT-RATE 볼륨의 공식 영상의학과 보고서를 조회하여 반환.

    `dataset/radiology_text_reports/{train,validation}_reports.csv` 를 읽어
    `VolumeName` (ct_path의 마지막 파일명) 으로 매칭.

    Args:
        ct_path: `train_fixed/` 또는 `valid_fixed/` 하위 CT-RATE NIfTI 경로.
        include: 'findings' | 'impression' | 'findings_plus_impression' (기본값)

    Returns:
        연결된 보고서 텍스트. 해당 볼륨이 없으면 예외 발생.
    """
    ct_path = Path(ct_path)
    if "valid_fixed" in ct_path.parts:
        csv = CT_RATE_ROOT / "radiology_text_reports" / "validation_reports.csv"
    elif "train_fixed" in ct_path.parts:
        csv = CT_RATE_ROOT / "radiology_text_reports" / "train_reports.csv"
    else:
        raise ValueError(
            f"CT path is not under train_fixed/ or valid_fixed/: {ct_path}"
        )
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(csv)
    vol_name = ct_path.name
    row = df[df["VolumeName"] == vol_name]
    if len(row) == 0:
        raise KeyError(f"VolumeName {vol_name!r} not found in {csv} ({len(df)} rows)")
    row = row.iloc[0]
    findings = str(row.get("Findings_EN", "")).strip()
    impression = str(row.get("Impressions_EN", "")).strip()
    if include == "findings":
        return findings
    if include == "impression":
        return impression
    if include == "findings_plus_impression":
        return f"Findings: {findings}\nImpression: {impression}".strip()
    raise ValueError(
        f"include must be findings | impression | findings_plus_impression, got {include!r}"
    )


# ---------------------------------------------------------------------------
# CT + 장기 마스크 로더
# ---------------------------------------------------------------------------


def _center_pad_crop_3d(
    arr: np.ndarray, target_shape: tuple[int, int, int], pad_value=0
) -> np.ndarray:
    """3D ndarray를 `target_shape`으로 중앙 크롭 + 패딩."""
    out = np.full(target_shape, pad_value, dtype=arr.dtype)
    src_start = [max(0, (arr.shape[i] - target_shape[i]) // 2) for i in range(3)]
    src_end = [min(arr.shape[i], src_start[i] + target_shape[i]) for i in range(3)]
    dst_start = [max(0, (target_shape[i] - arr.shape[i]) // 2) for i in range(3)]
    dst_end = [dst_start[i] + (src_end[i] - src_start[i]) for i in range(3)]
    out[
        dst_start[0] : dst_end[0],
        dst_start[1] : dst_end[1],
        dst_start[2] : dst_end[2],
    ] = arr[
        src_start[0] : src_end[0],
        src_start[1] : src_end[1],
        src_start[2] : src_end[2],
    ]
    return out


def load_ct_and_mask_for_local(
    nifti_path: str | Path,
    ts_total_path: str | Path | None = None,
    *,
    pad_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fVLM `forward_test_win` 에 넣을 CT + 장기 마스크 로드.

    fVLM 오프라인 전처리 (resize.py + preprocess.py) — 모델이 학습된 순서 — 를 재현:
      1. ref_spacing (1.0, 1.0, 3.0) mm 으로 리샘플 — 이미지 trilinear, 마스크 nearest
         (resize.py:189-196);
      2. ScaleIntensityRange(-1150, 350 → [0,1]) (preprocess.py:131-134);
      3. 장기 바운딩박스로 크롭 (+D 방향 5복셀, H/W 방향 20복셀 확장)
         (preprocess.py:34-56);
      4. SpatialPadd((112, 256, 352), 대칭) — preprocess.py:77-90.
         ≥(112, 256, 352)로 대칭 패딩만; **절대 크롭하지 않음** (업스트림 동일).

    이 함수는 업스트림 오프라인 파이프라인(= 전체 흉부 볼륨)만 재현한다. 장기별
    윈도잉(eval.py: center_crop → DivisiblePad → forward_test_win)은 별도 단계로,
    `center_crop_organ` / `divisible_pad_end` 가 담당한다. (업스트림은 슬라이딩
    윈도우를 쓰지 않는다 — eval.py:301-307 의 dense_patch_slices 는 dead code.)

    NOTE: 리샘플-먼저 순서가 중요 — 이전 버전은 리샘플 + bbox-크롭을 건너뛰고
    고해상도 원본 그리드(예: 1024x1024 @ 0.45 mm)를 중앙 크롭했는데,
    흉부 주변 폐가 잘리는 문제가 있었음. 리샘플을 먼저 해야 흉부 전체가 들어감.

    pad_only:
      - False (기본값, preprocess.py 충실): ≥(112, 256, 352) 대칭 패딩만, 크롭 없음.
        업스트림 dataloader 가 내보내는 전체 볼륨과 동일.
      - True: 위에 더해 ViT 패치 (16, 16, 32) 배수까지 올림 (eval.py DivisiblePad 까지
        합친 형태). 전체 볼륨에서 ViT 토큰 그리드를 직접 유도하는 경로(saliency 등)용.

    반환값: (image, mask) 각각 (1, 1, D, H, W), 가변 크기 (≥112, 256, 352);
    image float32 [0,1]; mask int64 {0,1,2,3,4} = {배경, 폐, 심장, 식도, 대동맥}.
    """
    nifti_path = Path(nifti_path)
    if ts_total_path is None:
        ts_total_path = ts_mask_path(nifti_path)

    ct_nii = nib.load(str(nifti_path))
    img_arr = ct_nii.get_fdata().astype(np.float32)  # (X, Y, Z), HU 단위
    mask_arr = nib.load(str(ts_total_path)).get_fdata().astype(np.int32)
    sx, sy, sz = (abs(float(z)) for z in ct_nii.header.get_zooms()[:3])

    # nearest-interp가 id를 혼합하지 않도록, 리샘플 전에 TS 레이블 → {1,2,3,4} 리매핑.
    remapped = np.zeros_like(mask_arr, dtype=np.int32)
    for ts_id, our_id in TS_LABEL_TO_ORGAN.items():
        remapped[mask_arr == ts_id] = our_id
    mask_arr = remapped

    # (1) (X, Y, Z) 순서로 ref_spacing (1.0, 1.0, 3.0) mm 리샘플.
    ref = (1.0, 1.0, 3.0)
    # upstream(resize.py:198)은 `int(...)` 절삭(round 아님)으로 타겟 크기를 정한다.
    new_shape = [
        max(1, int(img_arr.shape[i] * s / ref[i])) for i, s in enumerate((sx, sy, sz))
    ]
    img_arr = F.interpolate(
        torch.from_numpy(img_arr)[None, None],
        size=new_shape,
        mode="trilinear",
        align_corners=False,
    )[0, 0].numpy()
    mask_arr = (
        F.interpolate(
            torch.from_numpy(mask_arr.astype(np.float32))[None, None],
            size=new_shape,
            mode="nearest",
        )[0, 0]
        .numpy()
        .astype(np.int32)
    )

    # fVLM의 (D, H, W) 입력 축 순서에 맞추어 (X, Y, Z) → (Z, Y, X) 변환.
    img_arr = img_arr.transpose(2, 1, 0)
    mask_arr = mask_arr.transpose(2, 1, 0)

    # (2) 강도 스케일링 — upstream은 ScaleIntensityRange (preprocess.py:131-134)를
    # bbox 크롭보다 먼저 적용. 크롭은 마스크 기반 순수 공간 선택이라
    # scale↔crop 교환 법칙이 성립(수치 동일)하지만, 충실도를 위해 upstream 순서 유지.
    img_arr = np.clip(img_arr, -1150, 350)
    img_arr = (img_arr - (-1150)) / (350 - (-1150))

    # (3) 장기 바운딩박스 크롭 (+D 5복셀, H/W 20복셀 확장) — preprocess.py:34-56.
    old_ids = np.unique(
        mask_arr
    )  # 크롭이 장기를 통째로 잘라냈는지 검증용 (preprocess.py:75)
    nz = np.nonzero(mask_arr)
    if nz[0].size > 0:
        ext = (5, 20, 20)
        # upstream(preprocess.py:54-62)은 `max_idx + extend`를 exclusive 슬라이스 끝으로 쓴다
        # (+1 없음) — 하단 마진 `extend`, 상단 마진 `extend-1`로 1복셀 비대칭.
        lo = [max(0, int(nz[a].min()) - ext[a]) for a in range(3)]
        hi = [min(mask_arr.shape[a], int(nz[a].max()) + ext[a]) for a in range(3)]
        img_arr = img_arr[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        mask_arr = mask_arr[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        # (3-검증) upstream preprocess.py:75 — bbox 크롭이 어떤 장기도 통째로
        # 제거하지 않았어야 한다. (경계-제로화는 뒤에서 일어나므로 여기 영향 없음.)
        assert np.array_equal(np.unique(mask_arr), old_ids), (
            f"bbox crop dropped an organ: {old_ids.tolist()} -> {np.unique(mask_arr).tolist()}"
        )

    # (4) SpatialPadd((112,256,352), 대칭) — preprocess.py:77-90.
    # ≥(112,256,352)로 대칭 패딩만; out_shape ≥ arr.shape 라 _center_pad_crop_3d 는
    # 절대 크롭하지 않는다 (업스트림 SpatialPad 와 동일).
    # pad_only=True: 추가로 ViT 패치(16,16,32) 배수까지 올림 (eval.py DivisiblePad 합산).
    if pad_only:
        import math  # noqa: PLC0415

        out_shape = tuple(
            max(
                LOCAL_IMG_SIZE[a],
                math.ceil(img_arr.shape[a] / FVLM_PATCH_SIZE[a]) * FVLM_PATCH_SIZE[a],
            )
            for a in range(3)
        )
    else:
        out_shape = tuple(max(LOCAL_IMG_SIZE[a], img_arr.shape[a]) for a in range(3))
    img_arr = _center_pad_crop_3d(img_arr, out_shape, pad_value=0.0)  # (D, H, W)
    mask_arr = _center_pad_crop_3d(mask_arr, out_shape, pad_value=0)  # (D, H, W)

    img_t = torch.from_numpy(img_arr).float().unsqueeze(0).unsqueeze(0)
    mask_t = torch.from_numpy(mask_arr).long().unsqueeze(0).unsqueeze(0)
    return img_t, mask_t


# ---------------------------------------------------------------------------
# 장기별 eval 윈도잉 — eval.py:99-125 center_crop + eval.py:239-245 DivisiblePad
# (업스트림은 슬라이딩 윈도우가 아니라, 장기마다 이 윈도우를 1회만 추론에 쓴다.
#  eval.py:301-307 의 dense_patch_slices 는 dead code.)
# ---------------------------------------------------------------------------


def center_crop_organ(
    image: torch.Tensor,
    mask: torch.Tensor,
    organ_idx: int,
    *,
    crop_size: tuple[int, int, int] = FVLM_ROI_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """타깃 장기를 중심으로 윈도우를 잘라낸다 — eval.py:99-125 `center_crop` 충실 이식.

    업스트림은 장기마다 그 장기를 중심에 둔 윈도우 하나만 추론에 쓴다(슬라이딩 X).
    크롭 크기는 축마다 max(crop_size, 장기 bbox extent) — 장기가 roi 보다 크면 윈도우가
    장기 전체를 담도록 커진다(폐가 흔히 그렇다). 그래서 타깃 장기는 절대 잘리지 않는다.

    Args:
        image: (1, 1, D, H, W) float — 전처리된 전체 흉부 볼륨.
        mask:  (1, 1, D, H, W) int  — multilabel {0,1,2,3,4}.
        organ_idx: 0-based 장기 인덱스 (eval.py 와 동일; lung=0, heart=1, …).
                   타깃 마스크 레이블 값은 organ_idx + 1.
        crop_size: (D, H, W) 최소 윈도우 = eval.py roi_size (112, 288, 352).

    Returns:
        (win_img, win_mask) 각각 (1, 1, d', h', w'); win_mask 는 단일 장기 {0, organ_idx+1}
        (eval.py:321-322 `window_mask[window_mask==1] = organ_id+1` 재현).
    """
    label = organ_idx + 1
    binary = mask == label  # (1, 1, D, H, W) bool — 타깃 장기만
    nz = torch.nonzero(binary[0, 0], as_tuple=False)  # (P, 3) = (z=D, y=H, x=W)
    if nz.numel() == 0:
        raise ValueError(f"organ_idx={organ_idx} (label {label}) absent from mask")
    lo = nz.min(0).values  # (z_min, y_min, x_min)
    hi = nz.max(0).values  # (z_max, y_max, x_max)
    dim = image.shape[-3:]  # (D, H, W)
    # 축별 crop = max(roi, 장기 extent), 장기 bbox 중심에 정렬. eval.py:102-106.
    crop = [max(crop_size[a], int(hi[a] - lo[a])) for a in range(3)]  # (cd, ch, cw)
    center = [(int(lo[a]) + int(hi[a])) // 2 for a in range(3)]  # (cz, cy, cx)
    spans = []
    for a in range(3):
        s = max(0, center[a] - crop[a] // 2)
        e = min(int(dim[a]), s + crop[a])
        if (
            e - s < crop[a]
        ):  # 끝에 밀려 모자라면 시작을 당겨 크기 확보. eval.py:112-113.
            s = max(0, e - crop[a])
        spans.append((s, e))
    (zs, ze), (ys, ye), (xs, xe) = spans
    win_img = image[..., zs:ze, ys:ye, xs:xe]  # (1, 1, d', h', w')
    win_mask = binary[..., zs:ze, ys:ye, xs:xe].to(mask.dtype) * label  # {0, label}
    return win_img, win_mask


def divisible_pad_end(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    k: tuple[int, int, int] = FVLM_PATCH_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """각 공간 축을 k 의 배수까지 **끝에만** 패딩 — eval.py:239-245 DivisiblePadd(method="end").

    image/mask: (1, 1, D, H, W). 반환은 (D, H, W) 가 각각 k 의 배수.
    """
    import math  # noqa: PLC0415

    d, h, w = image.shape[-3:]
    tgt = [int(math.ceil(s / ki) * ki) for s, ki in zip((d, h, w), k)]  # (td, th, tw)
    # F.pad 는 마지막 축부터: (W앞, W뒤, H앞, H뒤, D앞, D뒤); method="end" → 앞=0.
    pad = (0, tgt[2] - w, 0, tgt[1] - h, 0, tgt[0] - d)
    img = F.pad(image, pad, value=0.0)  # (1, 1, td, th, tw)
    msk = F.pad(mask, pad, value=0)  # (1, 1, td, th, tw)
    return img, msk


# ---------------------------------------------------------------------------
# 텍스트 입력 빌더 (fVLM `prepare_text_feat` 형식)
# ---------------------------------------------------------------------------


def build_local_test_items(
    reports_per_organ: dict[str, dict[str, str]],
) -> list[tuple]:
    """사용자 dict → fVLM `prepare_text_feat` test_items 형식 변환.

    Args:
        reports_per_organ: ::

            {"lung":      {"pos": "Emphysema.",   "neg": "Not Emphysema."},
             "heart":     {"pos": "Cardiomegaly.","neg": "Not Cardiomegaly."},
             "esophagus": {"pos": "...",          "neg": "..."},
             "aorta":     {"pos": "...",          "neg": "..."}}

        장기별 선택적 `id` 키가 있으면 기본 prompt_id를 덮어씀.

    Returns:
        (organ_name, prompt_id, neg_prompt, pos_prompt) 튜플 리스트 —
        third_party/fvlm/eval.py:143-160 에서 기대하는 정확한 형식.
    """
    items = []
    for organ, prompts in reports_per_organ.items():
        if organ not in ORGAN_TO_ID:
            raise ValueError(f"unknown organ '{organ}' (expected one of {ORGANS})")
        prompt_id = prompts.get("id", prompts["pos"])
        items.append((organ, prompt_id, prompts["neg"], prompts["pos"]))
    return items


def build_decomposed_test_items_as_local(
    organ_text: dict[str, str], *, neg: str = ""
) -> list[tuple]:
    """장기별 분해 보고서 → fVLM `prepare_text_feat` test_items 변환.

    anatomy-aware model에 환자 보고서를 입력하는 올바른 방법:
    각 장기에 보고서 전체가 아니라 그 장기의 분해 텍스트를 pos_prompt로 부여.
    `organ_text`는 `src.data.fvlm_organ_report.build_organ_text` 출력
    (train/split-valid용 저자 JSON, 또는 valid_fixed용 Qwen 생성 JSON — 호환 가능).

    `organ_text`에 없는 장기는 건너뜀. neg_prompt 기본값은 빈 문자열
    (모델은 여전히 대조 로짓을 계산함).

    Returns: (organ, prompt_id, neg_prompt, pos_prompt) 튜플 —
    third_party/fvlm/eval.py:143-160 에서 소비하는 형식.
    """
    return [
        (organ, f"report:{organ}", neg, organ_text[organ])
        for organ in ORGANS
        if organ in organ_text
    ]
