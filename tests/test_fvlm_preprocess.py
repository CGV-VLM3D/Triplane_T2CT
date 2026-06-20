"""fVLM 장기별 eval 윈도잉 헬퍼 단위 테스트 (CPU 전용 — GPU/가중치 불필요).

`center_crop_organ` / `divisible_pad_end` 가 업스트림 third_party/fvlm/eval.py 의
`center_crop`(:99-125) + `DivisiblePadd(method="end")`(:239-245)와 일치하는지,
인라인 오라클로 bit-exact 대조한다. (업스트림은 슬라이딩 윈도우가 아니라 장기별
center-crop 윈도우 1개를 쓴다 — dense_patch_slices 는 dead code.)
"""

from __future__ import annotations

import torch

from src.baselines.fvlm_preprocess import (
    FVLM_PATCH_SIZE,
    FVLM_ROI_SIZE,
    center_crop_organ,
    divisible_pad_end,
)


# --- 업스트림 오라클 (eval.py:18-55, 99-125 verbatim, torch-only) -------------
def _masks_to_boxes_3d(masks: torch.Tensor) -> torch.Tensor:
    d, h, w = masks.shape[-3:]
    z = torch.arange(0, d).float()
    y = torch.arange(0, h).float()
    x = torch.arange(0, w).float()
    z, y, x = torch.meshgrid(z, y, x, indexing="ij")
    xm = masks * x.unsqueeze(0)
    x_max = xm.flatten(1).max(-1).values
    x_min = xm.masked_fill(~masks.bool(), float("inf")).flatten(1).min(-1).values
    ym = masks * y.unsqueeze(0)
    y_max = ym.flatten(1).max(-1).values
    y_min = ym.masked_fill(~masks.bool(), float("inf")).flatten(1).min(-1).values
    zm = masks * z.unsqueeze(0)
    z_max = zm.flatten(1).max(-1).values
    z_min = zm.masked_fill(~masks.bool(), float("inf")).flatten(1).min(-1).values
    return torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], dim=1)


def _up_center_crop(image, mask, crop_size):
    x_min, y_min, z_min, x_max, y_max, z_max = _masks_to_boxes_3d(mask)[0].long()
    crop_d = max(crop_size[0], z_max - z_min)
    crop_h = max(crop_size[1], y_max - y_min)
    crop_w = max(crop_size[2], x_max - x_min)
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    cz = (z_min + z_max) // 2
    d, h, w = image.shape[-3:]
    x_start = max(0, cx - crop_w // 2)
    x_end = min(w, x_start + crop_w)
    if x_end - x_start < crop_w:
        x_start = max(0, x_end - crop_w)
    y_start = max(0, cy - crop_h // 2)
    y_end = min(h, y_start + crop_h)
    if y_end - y_start < crop_h:
        y_start = max(0, y_end - crop_h)
    z_start = max(0, cz - crop_d // 2)
    z_end = min(d, z_start + crop_d)
    if z_end - z_start < crop_d:
        z_start = max(0, z_end - crop_d)
    return (
        image[..., z_start:z_end, y_start:y_end, x_start:x_end],
        mask[..., z_start:z_end, y_start:y_end, x_start:x_end],
    )


def _make_volume() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    D, H, W = 120, 400, 360  # (1, 1, D, H, W)
    img = torch.rand(1, 1, D, H, W)
    mask = torch.zeros(1, 1, D, H, W, dtype=torch.long)
    mask[0, 0, 30:90, 50:380, 40:320] = 1  # lung — H extent 330 > roi 288 → crop 확장
    mask[0, 0, 40:70, 100:160, 80:140] = 2  # heart — roi 보다 작음
    return img, mask


def test_center_crop_organ_matches_upstream_oracle() -> None:
    """center_crop_organ 이 eval.py center_crop 과 img/mask 모두 bit-exact 일치."""
    img, mask = _make_volume()
    for organ_idx, label in [(0, 1), (1, 2)]:
        wi, wm = center_crop_organ(img, mask, organ_idx)  # (1, 1, d', h', w')
        ui, um = _up_center_crop(img, (mask == label).long(), FVLM_ROI_SIZE)
        um = um.clone()
        um[um == 1] = label  # eval.py:321-322 재라벨
        assert torch.equal(wi, ui), f"organ{organ_idx}: image crop mismatch"
        assert torch.equal(wm, um), f"organ{organ_idx}: mask crop mismatch"
        assert wm.unique().tolist() == [0, label]  # 단일 장기만


def test_center_crop_organ_grows_to_contain_large_organ() -> None:
    """장기가 roi 보다 크면 윈도우가 그 축으로 커진다 (eval.py:102 max())."""
    img, mask = _make_volume()
    wi, _ = center_crop_organ(img, mask, 0)  # lung
    # H extent(330) > roi H(288) → 윈도우 H 가 roi 보다 큼; D/W 는 roi 로 고정.
    assert wi.shape[3] > FVLM_ROI_SIZE[1]
    assert wi.shape[2] == FVLM_ROI_SIZE[0]
    assert wi.shape[4] == FVLM_ROI_SIZE[2]


def test_divisible_pad_end_pads_only_at_end() -> None:
    """divisible_pad_end: 각 축 패치 배수 + 끝에만 패딩(앞부분 불변)."""
    img, mask = _make_volume()
    wi, wm = center_crop_organ(img, mask, 0)
    d, h, w = wi.shape[-3:]
    pi, pm = divisible_pad_end(wi, wm)
    # 패치 배수
    for s, k in zip(pi.shape[2:], FVLM_PATCH_SIZE):
        assert s % k == 0
    assert pi.shape[2] >= d and pi.shape[3] >= h and pi.shape[4] >= w
    # 끝-패딩 → 원본은 앞쪽 슬라이스에 그대로 보존
    assert torch.equal(pi[..., :d, :h, :w], wi)
    assert torch.equal(pm[..., :d, :h, :w], wm)
    # 패딩 영역은 0
    if pi.shape[4] > w:
        assert torch.all(pi[..., :, :, w:] == 0)
