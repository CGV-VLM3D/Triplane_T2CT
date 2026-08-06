"""probe 공용 헬퍼 — CT를 wan 그리드로 올리고, teacher/마스크를 같은 격자에 맞춘다.

모델 쪽(로딩·windowing·token grid 재조립)은 전부
[src/baselines/spectre_adapter.py](../../src/baselines/spectre_adapter.py)에 있다.
여기 있는 건 probe에서만 쓰는 **데이터 접근 + 전처리 + 마스크 정렬**뿐이다.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import torch

from src.baselines.spectre_adapter import (
    CKPT_COMBINER,
    CKPT_VLA,
    DEFAULT_CKPT,
    SpectreBackbone,
)

CKPT_SSL: Final[Path] = DEFAULT_CKPT

CTRATE_ROOT: Final[Path] = Path("/workspace/datasets/datasets/CT-RATE/dataset")
TS_TOTAL_DIR: Final[Path] = CTRATE_ROOT / "ts_seg" / "ts_total"
FVLM_RESIZE_PY: Final[Path] = Path("/workspace/third_party/fvlm/data/resize.py")

# wan latent 그리드 (data/report2ct_wan/latents_512x512x253/meta.json)
IN_PLANE: Final[int] = 512
DEPTH_WAN: Final[int] = 253  # Wan causal ×4 temporal 압축이 무손실인 값 (253 → 64)
DEPTH_PAD: Final[int] = 256  # SPECTRE crop depth 64의 배수 — end-pad 필수
AIR_HU: Final[float] = -1000.0

TOKEN_GRID: Final[tuple[int, int, int]] = SpectreBackbone.token_grid(
    (IN_PLANE, IN_PLANE, DEPTH_PAD)
)  # (32, 32, 32)
PATCH_SIZE: Final[tuple[int, int, int]] = SpectreBackbone.patch_size
EMBED_DIM: Final[int] = SpectreBackbone.dense_dim


#: probe에서 SRSS를 재는 장기. TS v2에서 폐는 5-lobe라 prefix 매칭으로 합친다.
PROBE_ORGANS: Final[tuple[str, ...]] = ("lung_", "liver", "heart", "aorta")


def ts_labels_with_prefix(prefix: str) -> list[int]:
    """`third_party/fvlm/data/resize.py`의 class_map에서 이름이 `prefix`로 시작하는 레이블 id들.

    손으로 int를 나열하지 않는다 (`.claude/rules/fvlm.md`의 single-source-of-truth 규칙).
    `"lung_"` → 5-lobe 전체, `"liver"` → 단일 id.
    """
    body = re.search(
        r"class_map = \{(.*?)\n        \}", FVLM_RESIZE_PY.read_text(), re.S
    )
    if body is None:
        raise RuntimeError(
            "resize.py의 class_map 블록을 찾지 못했다 — 상류가 바뀌었는지 확인"
        )
    pairs = re.findall(r"(\d+)\s*:\s*['\"]([^'\"]+)['\"]", body.group(1))
    ids = [int(k) for k, name in pairs if name.startswith(prefix)]
    if not ids:
        raise ValueError(f"class_map에 '{prefix}'로 시작하는 장기가 없다")
    return ids


def ts_lung_labels() -> list[int]:
    """폐 5-lobe 레이블 id (하위호환 별칭)."""
    return ts_labels_with_prefix("lung_")


def _split_of(scan_id: str) -> str:
    return "valid_fixed" if scan_id.startswith("valid") else "train_fixed"


def ct_path(scan_id: str) -> Path:
    """`valid_1000_a_1` → `<root>/valid_fixed/valid_1000/valid_1000_a/valid_1000_a_1.nii.gz`."""
    parts = scan_id.split("_")
    return (
        CTRATE_ROOT / _split_of(scan_id) / "_".join(parts[:2]) / "_".join(parts[:3])
    ) / f"{scan_id}.nii.gz"


def mask_path(scan_id: str) -> Path:
    """ts_seg ts_total 마스크 경로 (raw CT와 같은 `<patient>/<scan>/` 레이아웃)."""
    parts = scan_id.split("_")
    return (
        TS_TOTAL_DIR / _split_of(scan_id) / "_".join(parts[:2]) / "_".join(parts[:3])
    ) / f"{scan_id}.nii.gz"


def mm_per_voxel(scan_id: str) -> tuple[float, float, float]:
    """공통 프레임 `(512, 512, 253→256)` 한 복셀이 덮는 **mm** (RAS 축 순서).

    `load_volume`은 원본을 RAS로 재정렬한 뒤 고정 격자로 리샘플하므로, 복셀 크기는 스캔마다
    다르다 (FOV / 512). SRSS 이웃 반경을 복셀 단위로 두면 프레임마다 물리적 크기가 달라져
    teacher 간 비교가 깨진다 — fVLM은 정확히 1×1×3 mm 인데 여기는 대략 0.7×0.7×1.4 mm 다.
    Manhattan 거리라 z 한 걸음이 fVLM에서 2배 넓고, 반경이 크면 negative를 더 멀리서 뽑아
    SRSSdm이 부풀어 오른다.

    Returns:
        `(mm_x, mm_y, mm_z)` — RAS 축 기준.
    """
    import nibabel as nib  # noqa: PLC0415

    nii = nib.load(str(ct_path(scan_id)))
    zooms, shape = nii.header.get_zooms()[:3], nii.shape[:3]
    axis = {"R": 0, "L": 0, "A": 1, "P": 1, "S": 2, "I": 2}
    extent = [0.0, 0.0, 0.0]  # RAS 축별 원본 물리 길이 (mm)
    for i, code in enumerate(nib.aff2axcodes(nii.affine)):
        extent[axis[code]] = float(shape[i]) * float(zooms[i])
    return (extent[0] / IN_PLANE, extent[1] / IN_PLANE, extent[2] / DEPTH_WAN)


def build_transforms(nearest: bool = False):
    """wan precompute와 **동일한** resample-only 파이프라인.

    `scripts/precompute_wan_image_embeddings.py:82-93`을 1:1로 따른다 — intensity scaling 없음
    (raw HU), Orientation RAS, `Resized((512, 512, 253))`. teacher grid가 wan latent grid와
    voxel 단위로 맞으려면 이게 정확히 같아야 한다.

    Args:
        nearest: 마스크용. `Resized` 보간을 nearest로 바꾼다.
    """
    import monai  # noqa: PLC0415
    from monai.transforms import Compose  # noqa: PLC0415

    return Compose(
        [
            monai.transforms.LoadImaged(keys="image"),
            monai.transforms.EnsureChannelFirstd(keys="image"),
            monai.transforms.Orientationd(keys="image", axcodes="RAS"),
            monai.transforms.EnsureTyped(keys="image", dtype=torch.float32),
            monai.transforms.Resized(
                keys="image",
                spatial_size=(IN_PLANE, IN_PLANE, DEPTH_WAN),  # (X, Y, Z)
                mode="nearest" if nearest else "trilinear",
            ),
        ]
    )


def load_volume(scan_id: str, mask: bool = False, pad: bool = True) -> torch.Tensor:
    """CT(또는 ts_seg 마스크)를 wan 그리드로 올려 `(1, 512, 512, 256)`으로 반환한다.

    Args:
        scan_id: 예 `valid_1000_a_1`.
        mask: True면 ts_seg 마스크를 nearest 보간으로 읽는다.
        pad: z를 253 → 256으로 end-pad한다 (CT는 air, 마스크는 0). **끄지 말 것** —
            안 하면 `window_scan`이 192로 center-crop해 61 slice가 조용히 사라진다.
    """
    path = mask_path(scan_id) if mask else ct_path(scan_id)
    out = build_transforms(nearest=mask)({"image": str(path)})["image"]
    vol = out.as_tensor() if hasattr(out, "as_tensor") else out
    return pad_depth(vol, 0.0 if mask else AIR_HU) if pad else vol


def pad_depth(x: torch.Tensor, pad_value: float) -> torch.Tensor:
    """z를 253 → 256으로 **end-pad**한다 (`(C, H, W, D)` → `(C, H, W, 256)`)."""
    pad = DEPTH_PAD - x.shape[-1]
    if pad < 0:
        raise ValueError(f"depth {x.shape[-1]} > {DEPTH_PAD}")
    return x if pad == 0 else torch.nn.functional.pad(x, (0, pad), value=pad_value)


def lung_occupancy(scan_id: str) -> torch.Tensor:
    """ts_seg 폐 마스크를 teacher token grid로 평균 pool한다 → `(32, 32, 32)` 점유율 [0, 1]."""
    return organ_occupancy(scan_id, ("lung_",))["lung_"]


def organ_occupancy(
    scan_id: str, organs: tuple[str, ...] = PROBE_ORGANS
) -> dict[str, torch.Tensor]:
    """장기별 ts_seg 마스크를 teacher token grid로 평균 pool한다.

    마스크는 CT와 **동일한 transform**(nearest 보간 + 같은 end-pad)으로 올라오므로 teacher
    토큰과 voxel 단위로 정렬된다.

    Returns:
        이름 → `(32, 32, 32)` 점유율 [0, 1].
    """
    mask = load_volume(scan_id, mask=True)[0].round().long()  # (512, 512, 256)
    return {
        organ: pool_to_token_grid(
            torch.isin(mask, torch.tensor(ts_labels_with_prefix(organ))).float()
        )
        for organ in organs
    }


def body_occupancy(scan_id: str, hu_threshold: float = -500.0) -> torch.Tensor:
    """CT에서 **몸통** 토큰 점유율을 만든다 → `(32, 32, 32)`.

    CT는 절반 가까이가 공기라, 공기 토큰끼리는 서로 매우 유사해서 공간 구조 지표를 전부
    낙관적으로 부풀린다. 그래서 U3는 모든 지표를 "전체 토큰"과 "몸통 토큰만" 두 축에서 보고한다.
    """
    ct = load_volume(scan_id)[0]  # (512, 512, 256) raw HU
    return pool_to_token_grid((ct > hu_threshold).float())


def pool_to_token_grid(vol: torch.Tensor) -> torch.Tensor:
    """`(H, W, D)` voxel 볼륨을 token grid `(32, 32, 32)`로 평균 pool한다."""
    h, w, d = vol.shape
    p_h, p_w, p_d = PATCH_SIZE
    return vol.reshape(h // p_h, p_h, w // p_w, p_w, d // p_d, p_d).mean(dim=(1, 3, 5))


def build_backbone(
    ckpt: Path = CKPT_SSL, device: str = "cuda", with_combiner: bool = False
):
    """probe용 frozen SPECTRE 인코더 (어댑터 그대로)."""
    return SpectreBackbone(
        ckpt_path=ckpt, device_str=device, with_combiner=with_combiner
    )


__all__ = [
    "AIR_HU",
    "CKPT_COMBINER",
    "CKPT_SSL",
    "CKPT_VLA",
    "DEPTH_PAD",
    "DEPTH_WAN",
    "EMBED_DIM",
    "IN_PLANE",
    "PATCH_SIZE",
    "PROBE_ORGANS",
    "TOKEN_GRID",
    "SpectreBackbone",
    "body_occupancy",
    "build_backbone",
    "build_transforms",
    "ct_path",
    "load_volume",
    "lung_occupancy",
    "mm_per_voxel",
    "mask_path",
    "organ_occupancy",
    "pad_depth",
    "pool_to_token_grid",
    "ts_labels_with_prefix",
    "ts_lung_labels",
]
