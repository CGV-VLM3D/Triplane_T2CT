"""Unit 2 — z② supervised 18-label 3D CNN (MAISI latent) 학습 + penultimate 추출.

작은 3D conv 분류기를 train(toy_v2) latent로 학습 → penultimate(256-d)을 Probe B의
semantic image feature ``z②``로 쓴다. 동시에 valid_v2에서의 18-label mean-AUC를
보고(=Probe C, image-side content / z② 천장). 게이트: mean-AUC ≥ ~0.7 이면 z② 채택.

5개 text 인코더와 **독립**(random init, image+label만 사용) → Probe B 순환논증 방지.

사용:
    CUDA_VISIBLE_DEVICES=2 python -m tests.alignment_probe.classifier --device cuda:0 --epochs 20
출력:
    tests/alignment_probe/z_classifier.pt                 (가중치)
    tests/alignment_probe/embeddings/<split>/z_classifier.npz   (emb (N,256), scan_ids)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tests.alignment_probe.cases import ABNORMALITY_LABELS, AlignmentCase, load_cases
from tests.alignment_probe.latents import has_latent, load_latent

_CKPT = Path("tests/alignment_probe/z_classifier.pt")
_OUT_ROOT = Path("tests/alignment_probe/embeddings")
_FEAT_DIM = 256


class Latent3DCNN(nn.Module):
    """MAISI latent (4,120,120,64) → 18-label. penultimate = 256-d global feature."""

    def __init__(self, in_ch: int = 4, feat_dim: int = _FEAT_DIM, n_labels: int = 18):
        super().__init__()

        def block(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv3d(i, o, 3, padding=1),
                nn.BatchNorm3d(o),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2),
            )

        self.body = nn.Sequential(
            block(in_ch, 32),  # (32, 60, 60, 32)
            block(32, 64),  # (64, 30, 30, 16)
            block(64, 128),  # (128, 15, 15, 8)
            nn.Conv3d(128, feat_dim, 3, padding=1),
            nn.BatchNorm3d(feat_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),  # (feat_dim, 1, 1, 1)
        )
        self.head = nn.Linear(feat_dim, n_labels)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 4, 120, 120, 64) → (B, feat_dim, 15, 15, 8) pre-pool 공간 feature map.

        per-organ pooling용 — `body`의 마지막 AdaptiveAvgPool3d 직전 출력. state_dict는
        그대로(슬라이스는 동일 child 모듈 참조).
        """
        return self.body[:-1](x)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 4, 120, 120, 64) → (B, feat_dim) penultimate."""
        return self.body(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 4, 120, 120, 64) → (B, 18) logits."""
        return self.head(self.features(x))


class LatentDataset(Dataset):
    """latent이 디스크에 있는 케이스만. __getitem__ → (latent (4,120,120,64), labels (18,), scan_id)."""

    def __init__(self, cases: list[AlignmentCase], split: str):
        self.split = split
        self.items = [c for c in cases if has_latent(c.scan_id, split)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        c = self.items[i]
        y = torch.tensor([c.labels[a] for a in ABNORMALITY_LABELS], dtype=torch.float32)
        return load_latent(c.scan_id, self.split), y, c.scan_id


def _mean_auc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict]:
    """라벨별 ROC-AUC의 평균(단일 클래스뿐인 라벨은 제외)."""
    per = {}
    for j, name in enumerate(ABNORMALITY_LABELS):
        if len(np.unique(y_true[:, j])) < 2:
            continue
        per[name] = roc_auc_score(y_true[:, j], y_score[:, j])
    return float(np.mean(list(per.values()))), per


@torch.no_grad()
def _predict(model: Latent3DCNN, loader: DataLoader, device: str):
    """→ (logits (N,18), labels (N,18), scan_ids (N,))."""
    model.eval()
    logits, labels, ids = [], [], []
    for x, y, sid in loader:
        logits.append(model(x.to(device)).cpu())
        labels.append(y)
        ids.extend(sid)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy(), np.array(ids)


@torch.no_grad()
def _extract_features(model: Latent3DCNN, loader: DataLoader, device: str):
    """→ (feats (N,feat_dim), scan_ids (N,))."""
    model.eval()
    feats, ids = [], []
    for x, _y, sid in loader:
        feats.append(model.features(x.to(device)).cpu())
        ids.extend(sid)
    return torch.cat(feats).numpy(), np.array(ids)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()
    device = args.device

    train_ds = LatentDataset(load_cases("train"), "train")
    proxy_ds = LatentDataset(load_cases("valid_v2"), "valid_v2")
    print(
        f"train latents={len(train_ds)}  valid_v2 latents={len(proxy_ds)}", flush=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        proxy_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = Latent3DCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y, _ in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            running += loss.item()
        logits, labels, _ = _predict(model, eval_loader, device)
        auc, _per = _mean_auc(labels, logits)
        print(
            f"epoch {ep:02d}/{args.epochs}  train_loss={running / len(train_loader):.4f}"
            f"  proxy mean-AUC={auc:.4f}",
            flush=True,
        )

    _CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), _CKPT)

    # Probe C 보고 + z② 추출(train + valid_v2).
    logits, labels, _ = _predict(model, eval_loader, device)
    auc, per = _mean_auc(labels, logits)
    print(
        f"\n[Probe C] valid_v2 mean-AUC={auc:.4f}  ({'PASS' if auc >= 0.7 else 'BELOW 0.7'})"
    )
    for name, v in sorted(per.items(), key=lambda kv: -kv[1]):
        print(f"  {name:32s} {v:.3f}")

    for split, ds in (("train", train_ds), ("valid_v2", proxy_ds)):
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        feats, ids = _extract_features(model, loader, device)
        out = _OUT_ROOT / split
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "z_classifier.npz", emb=feats, scan_ids=ids)
        print(f"saved z② {feats.shape} → {out / 'z_classifier.npz'}", flush=True)


if __name__ == "__main__":
    main()
