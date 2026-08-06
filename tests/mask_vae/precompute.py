"""학습된 마스크 VAE로 seg latent precompute.

전체 `(480,480,256)` 마스크를 인코딩 → seg latent `(4,120,120,64)` → `{id}_maskemb.nii.gz`(HWDC) 저장.
[src/baselines/report2ct_image_encoder.py](../../src/baselines/report2ct_image_encoder.py)의 encode/save 규약을
그대로 따라 이미지 latent과 격자·affine이 정렬되게 한다(Stage 3 concat용).

체크포인트는 train.py의 `_ckpt()` 형식: ``{"ae", "embed", "input_mode", "embed_dim"}``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from tests.mask_vae.dataset import list_mask_paths
from tests.mask_vae.model import build_mask_vae
from tests.mask_vae.preprocess import NUM_CLASSES, build_mask_transform, to_vae_input

_IN_CH = {"raw": 1, "onehot": NUM_CLASSES}  # embed는 embed_dim


def load_mask_vae(
    ckpt_path: str | Path, device: torch.device, num_splits: int = 2
) -> tuple[torch.nn.Module, torch.nn.Module | None, str]:
    """train.py `_ckpt()` 체크포인트에서 (ae, embed_layer, input_mode) 복원(frozen/eval).

    Args:
        ckpt_path: `results/<exp>/best.pt`.
        device: 디바이스.
        num_splits: 480³ 전체 인코딩 메모리용 conv 분할(가중치와 무관, 크게 잡아도 됨).

    Returns:
        (ae, embed_layer 또는 None, input_mode).
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    mode, dim = ckpt["input_mode"], ckpt["embed_dim"]
    in_ch = dim if mode == "embed" else _IN_CH[mode]
    ae = build_mask_vae(in_channels=in_ch, num_splits=num_splits).to(device)
    ae.load_state_dict(ckpt["ae"])
    ae.eval()
    embed = None
    if mode == "embed":
        embed = torch.nn.Embedding(NUM_CLASSES, dim).to(device)
        embed.load_state_dict(ckpt["embed"])
        embed.eval()
    return ae, embed, mode


def _prep(mask: torch.Tensor, mode: str, embed: torch.nn.Module | None) -> torch.Tensor:
    """정수 라벨 ``(B,1,·)`` → VAE 입력 ``(B,in_channels,·)`` (train.py prep과 동일)."""
    if mode == "raw":
        return to_vae_input(mask)
    idx = mask.long().squeeze(1)
    if mode == "onehot":
        return F.one_hot(idx, NUM_CLASSES).movedim(-1, 1).float()
    return embed(idx).movedim(-1, 1)


@torch.inference_mode()
def encode_mask(
    ae: torch.nn.Module,
    embed: torch.nn.Module | None,
    mode: str,
    mask_path: str,
    tfm,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """마스크 1개 → seg latent ``(120,120,64,4)`` (HWDC) + affine.

    Returns:
        (latent HWDC ``(120,120,64,4)`` float32, affine ``(4,4)``).
    """
    out = tfm({"mask": mask_path})["mask"]  # (1,480,480,256) long
    affine = out.meta["affine"].numpy() if hasattr(out, "meta") else np.eye(4)
    x = _prep(out.unsqueeze(0).to(device), mode, embed)  # (1,C,480,480,256)
    with autocast("cuda", dtype=torch.bfloat16):
        z = ae.encode_stage_2_inputs(x)  # (1,4,120,120,64) z_mu (deterministic)
    lat = z.squeeze(0).float().cpu().numpy().transpose(1, 2, 3, 0)  # (120,120,64,4)
    return np.float32(lat), affine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="results/<exp>/best.pt")
    ap.add_argument(
        "--split",
        type=str,
        default="valid_fixed",
        choices=["train_fixed", "valid_fixed"],
    )
    ap.add_argument("--limit", type=int, default=None, help="앞 N개만 (테스트용)")
    ap.add_argument("--out-dir", type=str, required=True, help="seg latent 저장 폴더")
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    dev = torch.device(args.device)
    ae, embed, mode = load_mask_vae(args.ckpt, dev)
    tfm = build_mask_transform(is_train=False)  # 전체 480³
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = list_mask_paths(args.split)
    if args.limit is not None:
        paths = paths[: args.limit]
    for i, p in enumerate(paths):
        scan_id = Path(p).name[: -len(".nii.gz")]
        lat, affine = encode_mask(ae, embed, mode, p, tfm, dev)
        nib.save(
            nib.Nifti1Image(lat, affine), str(out_dir / f"{scan_id}_maskemb.nii.gz")
        )
        if i % 50 == 0:
            print(f"[{i + 1}/{len(paths)}] {scan_id} → {tuple(lat.shape)}", flush=True)
    print(f"done: {len(paths)} seg latents → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
