from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.dataloader import build_dataloader, CTVolumeDataset  # noqa: E402
from monai.data import DataLoader  # noqa: E402


def add_bundle_to_syspath(bundle_dir: Path) -> None:
    bundle_dir = bundle_dir.resolve()
    if str(bundle_dir) not in sys.path:
        sys.path.insert(0, str(bundle_dir))


def find_ae_ckpt(bundle_dir: Path, override: str | None) -> Path:
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = bundle_dir / p
        if not p.exists():
            raise FileNotFoundError(f"--ae_ckpt does not exist: {p}")
        return p

    candidates = sorted((bundle_dir / "models").glob("autoencoder*.pt"))
    candidates += sorted((bundle_dir / "models").glob("autoencoder*.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"No autoencoder checkpoint found under {bundle_dir / 'models'}. "
            f"Pass --ae_ckpt explicitly."
        )
    return candidates[0]


def build_autoencoder(bundle_dir: Path, network_key: str):
    from monai.bundle import ConfigParser

    cfg_path = bundle_dir / "configs" / "inference.json"
    if not cfg_path.exists():
        cfg_path = bundle_dir / "configs" / "inference.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No inference config at {bundle_dir / 'configs'}")

    parser = ConfigParser()
    parser.read_config(str(cfg_path))
    parser.parse()

    if network_key not in parser.config:
        raise KeyError(
            f"Key '{network_key}' not found in {cfg_path}. "
            f"Available top-level keys: {list(parser.config.keys())[:20]}"
        )

    net = parser.get_parsed_content(network_key)
    return net


def load_state(net: torch.nn.Module, ckpt_path: Path) -> None:
    sd = torch.load(str(ckpt_path), map_location="cpu")
    # Bundles may save raw state_dict or wrap inside dict.
    if isinstance(sd, dict) and "state_dict" in sd and not any(
        k.startswith(("encoder", "decoder", "quant")) for k in sd
    ):
        sd = sd["state_dict"]

    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"[load_state] missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        print(f"  first missing: {missing[:5]}")
    if len(unexpected) > 0:
        print(f"  first unexpected: {unexpected[:5]}")


class EncodeWrapper(nn.Module):
    """
    Wraps the MAISI autoencoder so `forward(x)` returns the encoder output
    (mu, or mu||sigma along the channel dim). Required for `nn.DataParallel`
    since DP only proxies `forward`, not `encode`.
    """

    def __init__(self, net: nn.Module, return_sigma: bool):
        super().__init__()
        self.net = net
        self.return_sigma = return_sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, sigma = self.net.encode(x)
        if self.return_sigma:
            return torch.cat([mu, sigma], dim=1)
        return mu


@torch.no_grad()
def encode_volume(
    predictor,
    x: torch.Tensor,
    roi_size: tuple[int, int, int],
    overlap: float,
    sw_batch_size: int,
    return_sigma: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Sliding-window encode. `predictor` is the wrapped (possibly DP-parallelized)
    encoder. If return_sigma is True, the predictor output has 2C channels
    (mu||sigma) and we split it back.
    """
    from monai.inferers import sliding_window_inference

    out = sliding_window_inference(
        inputs=x,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=predictor,
        overlap=overlap,
        mode="gaussian",
        progress=False,
    )
    if return_sigma:
        c = out.shape[1] // 2
        return out[:, :c], out[:, c:]
    return out, None


def derive_save_stem(image_path: str, idx: int) -> str:
    base = os.path.basename(image_path)
    if base.endswith(".nii.gz"):
        base = base[: -len(".nii.gz")]
    if not base:
        base = f"sample_{idx:06d}"
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle_dir", type=str, required=True,
                        help="Path to the extracted MAISI bundle root.")
    parser.add_argument("--ae_ckpt", type=str, default=None,
                        help="Autoencoder weight path (optional, autodetect under models/).")
    parser.add_argument("--network_key", type=str, default="autoencoder_def",
                        help="Key of the autoencoder network_def in inference.json. "
                             "MAISI v0.4.x typically uses 'autoencoder_def'.")

    parser.add_argument("--data_root", type=str, default="/workspace/dataset/train")
    parser.add_argument("--split_json", type=str, default=None,
                        help="If set, use train/valid file lists from this JSON "
                             "instead of scanning --data_root. Paths in JSON are "
                             "resolved relative to its 'root' field. Latents are "
                             "written under <out_dir>/train and <out_dir>/valid; "
                             "stats are computed on the train split only.")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[480, 480, 256])
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--out_dir", type=str, default="/workspace/datasets/datasets/latents")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="DataLoader batch size (volumes per forward). "
                             "Volumes are Resized to a fixed spatial_size so B>1 is safe.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--save_dtype", type=str, default="float16",
                        choices=["float16", "float32"])
    parser.add_argument("--limit", type=int, default=-1,
                        help="Cap number of volumes (debug). -1 = all.")

    parser.add_argument("--sw_roi", type=int, nargs=3, default=None,
                        help="Sliding-window ROI for MAISI encoder. Default = "
                             "value from bundle's inference.json (autoencoder_sliding_window_infer_size).")
    parser.add_argument("--sw_overlap", type=float, default=None,
                        help="Sliding-window overlap (default from bundle inference.json).")
    parser.add_argument("--sw_batch_size", type=int, default=4)

    parser.add_argument("--save_sigma", action="store_true",
                        help="Save both mu and sigma so eps can be re-sampled at "
                             "training time (z = mu + sigma*eps). Roughly doubles "
                             "disk usage. Default: save mu only.")

    parser.add_argument("--num_shards", type=int, default=1,
                        help="Split the file list across N parallel processes.")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="0-indexed shard for this process (0..num_shards-1).")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs to use via nn.DataParallel. "
                             "When >1, set --batch_size to a multiple of --gpus.")
    args = parser.parse_args()

    if args.gpus > 1 and args.batch_size % args.gpus != 0:
        raise ValueError(f"--batch_size ({args.batch_size}) must be a multiple of --gpus ({args.gpus})")

    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(f"shard_id {args.shard_id} out of range for num_shards {args.num_shards}")

    bundle_dir = Path(args.bundle_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    add_bundle_to_syspath(bundle_dir)
    net = build_autoencoder(bundle_dir, args.network_key)

    # Resolve sliding-window settings from the bundle if not provided.
    import json as _json
    with open(bundle_dir / "configs" / "inference.json", "r") as f:
        bundle_cfg = _json.load(f)
    sw_roi = tuple(args.sw_roi) if args.sw_roi else tuple(
        bundle_cfg.get("autoencoder_sliding_window_infer_size", [80, 80, 80])
    )
    sw_overlap = args.sw_overlap if args.sw_overlap is not None else float(
        bundle_cfg.get("autoencoder_sliding_window_infer_overlap", 0.4)
    )
    print(f"[main] sliding-window roi={sw_roi} overlap={sw_overlap} sw_batch={args.sw_batch_size}")

    ckpt_path = find_ae_ckpt(bundle_dir, args.ae_ckpt)
    print(f"[main] loading weights: {ckpt_path}")
    load_state(net, ckpt_path)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "float32": torch.float32,
             "bfloat16": torch.bfloat16}[args.dtype]
    save_dtype = {"float16": torch.float16, "float32": torch.float32}[args.save_dtype]

    net = net.to(device).eval()

    # Build encode wrapper; wrap in DataParallel for multi-GPU.
    encoder = EncodeWrapper(net, return_sigma=args.save_sigma).to(device).eval()
    if args.gpus > 1:
        gpu_ids = list(range(args.gpus))
        encoder = nn.DataParallel(encoder, device_ids=gpu_ids)
        print(f"[main] DataParallel on GPUs {gpu_ids}")

    # ---- Build (split_name, files) pairs ----------------------------------
    splits: list[tuple[str, list[str]]] = []
    if args.split_json:
        split_path = Path(args.split_json).resolve()
        with open(split_path, "r") as f:
            split_obj = json.load(f)
        split_root = Path(split_obj.get("root", split_path.parent))
        for name in ("train", "valid"):
            entries = split_obj.get(name, [])
            files = [str((split_root / e["path"]).resolve()) for e in entries]
            splits.append((name, files))
        print(f"[main] split.json: {split_path}")
        for name, files in splits:
            print(f"  {name}: {len(files)} volumes (total)")

        if args.num_shards > 1:
            sharded: list[tuple[str, list[str]]] = []
            for name, files in splits:
                start = (len(files) * args.shard_id) // args.num_shards
                end = (len(files) * (args.shard_id + 1)) // args.num_shards
                sharded.append((name, files[start:end]))
            splits = sharded
            print(f"[main] shard {args.shard_id}/{args.num_shards}")
            for name, files in splits:
                print(f"  {name}: {len(files)} volumes (this shard)")
    else:
        loader = build_dataloader(
            data_root=args.data_root,
            batch_size=1,
            num_workers=args.num_workers,
            spatial_size=tuple(args.spatial_size),
            train=False,
            cache_rate=0.0,
            val_split=0.0,
        )
        ds = loader.dataset
        if hasattr(ds, "files"):
            files = list(ds.files)
        elif hasattr(ds, "data"):
            files = [d["image"] if isinstance(d, dict) else d for d in ds.data]
        else:
            raise RuntimeError(f"Cannot extract file paths from dataset of type {type(ds)}")
        splits.append(("", files))

    # ---- Welford accumulators (train split only) ---------------------------
    n_voxels = 0
    mean: torch.Tensor | None = None
    M2: torch.Tensor | None = None
    per_split_saved: dict[str, list[str]] = {}

    for split_name, split_files in splits:
        split_out = out_dir / split_name if split_name else out_dir
        split_out.mkdir(parents=True, exist_ok=True)

        ds = CTVolumeDataset(
            spatial_size=tuple(args.spatial_size),
            train=False,
            files=split_files,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        saved_paths: list[str] = []
        accumulate_stats = (split_name == "train") or not args.split_json

        desc = f"extract[{split_name}]" if split_name else "extract"
        vol_idx = 0
        pbar = tqdm(total=len(split_files), desc=desc, unit="vol")
        for batch in loader:
            if args.limit > 0 and vol_idx >= args.limit:
                break

            x = batch["image"]
            if hasattr(x, "as_tensor"):
                x = x.as_tensor()
            x = x.to(device=device, dtype=dtype, non_blocking=True)

            with torch.autocast(device_type=device.type,
                                dtype=dtype if dtype != torch.float32 else torch.float32,
                                enabled=(dtype != torch.float32)):
                mu, sigma = encode_volume(
                    predictor=encoder,
                    x=x,
                    roi_size=sw_roi,
                    overlap=sw_overlap,
                    sw_batch_size=args.sw_batch_size,
                    return_sigma=args.save_sigma,
                )

            mu_cpu = mu.detach().to(torch.float32).cpu()  # [B, C, D', H', W']
            sigma_cpu = sigma.detach().to(torch.float32).cpu() if sigma is not None else None
            assert mu_cpu.dim() == 5, f"unexpected mu shape: {mu_cpu.shape}"
            B = mu_cpu.shape[0]

            for b in range(B):
                mu_single = mu_cpu[b]  # [C, D', H', W']

                if accumulate_stats:
                    C = mu_single.shape[0]
                    flat = mu_single.reshape(C, -1)
                    n_new = flat.shape[1]
                    if mean is None:
                        mean = torch.zeros(C, dtype=torch.float64)
                        M2 = torch.zeros(C, dtype=torch.float64)
                    batch_mean = flat.mean(dim=1).to(torch.float64)
                    batch_var = flat.var(dim=1, unbiased=False).to(torch.float64)
                    delta = batch_mean - mean
                    new_n = n_voxels + n_new
                    mean = mean + delta * (n_new / new_n)
                    M2 = M2 + batch_var * n_new + (delta ** 2) * (n_voxels * n_new / new_n)
                    n_voxels = new_n

                src = split_files[vol_idx]
                stem = derive_save_stem(src, vol_idx)
                patient_dir = split_out / stem
                patient_dir.mkdir(parents=True, exist_ok=True)
                torch.save(mu_single.to(save_dtype), str(patient_dir / "mu.pt"))
                if sigma_cpu is not None:
                    torch.save(sigma_cpu[b].to(save_dtype),
                               str(patient_dir / "sigma.pt"))
                with open(patient_dir / "src.txt", "w") as f:
                    f.write(src + "\n")
                saved_paths.append(str(patient_dir))
                vol_idx += 1
                pbar.update(1)
                if args.limit > 0 and vol_idx >= args.limit:
                    break

        pbar.close()
        per_split_saved[split_name or "all"] = saved_paths

    if mean is None:
        raise RuntimeError("No volumes were processed.")

    var = M2 / max(1, n_voxels)
    std = var.clamp(min=1e-12).sqrt()

    sharded = args.num_shards > 1
    shard_tag = f"_shard{args.shard_id:02d}" if sharded else ""

    stats = {
        "num_volumes": {k: len(v) for k, v in per_split_saved.items()},
        "stats_split": "train" if args.split_json else "all",
        "num_voxels_per_channel": int(n_voxels),
        "channel_mean": mean.tolist(),
        "channel_M2": M2.tolist(),     # raw Welford sum-of-sq-dev, for shard merging
        "channel_std": std.tolist(),
        "spatial_size": list(args.spatial_size),
        "latent_channels": int(mean.shape[0]),
        "save_dtype": args.save_dtype,
        "ae_ckpt": str(ckpt_path),
        "bundle_dir": str(bundle_dir),
        "split_json": str(args.split_json) if args.split_json else None,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
    }

    stats_name = f"stats_partial{shard_tag}.json" if sharded else "stats.json"
    stats_path = out_dir / stats_name
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[main] wrote {stats_path}")
    print(f"  channel_mean: {[f'{m:.4f}' for m in stats['channel_mean']]}")
    print(f"  channel_std:  {[f'{s:.4f}' for s in stats['channel_std']]}")

if __name__ == "__main__":
    main()
