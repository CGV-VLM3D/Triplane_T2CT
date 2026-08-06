"""LatentClassifier 학습 — train(toy_v2) latent로 학습 → latent_classifier.pt + img_feat(npz) 추출.

valid_v2에서 18-label mean-AUC 보고(=Probe C, img_feat 천장; 게이트 ≥0.7).
사용: CUDA_VISIBLE_DEVICES=2 python -m tests.alignment_probe.main stage=train
출력: latent_classifier.pt + embeddings/<split>/img_feat.npz
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tests.alignment_probe.metric.metrics import mean_auc
from tests.alignment_probe.metric.plots import (
    mean_pairwise_cosine,
    save_auc_curve,
    save_confusion_matrices,
    save_loss_curve,
    save_metrics_json,
    save_per_label_auc,
    save_roc_curve,
)
from tests.alignment_probe.model.classifier import LatentClassifier, LatentDataset
from tests.alignment_probe.utils.cases import load_cases
from tests.alignment_probe.utils.io import EMB_ROOT, save_npz

_RESULTS = Path("tests/alignment_probe/results")


@torch.no_grad()
def _predict(model: LatentClassifier, loader: DataLoader, device: str):
    """→ (logits (N,18), labels (N,18), scan_ids (N,))."""
    model.eval()
    logits, labels, ids = [], [], []
    for x, y, sid in loader:
        logits.append(model(x.to(device)).cpu())
        labels.append(y)
        ids.extend(sid)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy(), np.array(ids)


@torch.no_grad()
def _extract_features(model: LatentClassifier, loader: DataLoader, device: str):
    """→ (feats (N,feat_dim), scan_ids (N,))."""
    model.eval()
    feats, ids = [], []
    for x, _y, sid in loader:
        feats.append(model.features(x.to(device)).cpu())
        ids.extend(sid)
    return torch.cat(feats).numpy(), np.array(ids)


def run(cfg) -> None:
    """LatentClassifier을 학습하고 img_feat 임베딩을 추출한다.

    train split으로 BCEWithLogitsLoss 분류기를 학습한 뒤 valid_v2로 Probe C mean-AUC를 보고한다.
    완료 후 train·valid_v2 두 split의 feature ``(N, feat_dim)``을 img_feat.npz로 저장한다.
    """
    device, epochs = cfg.device, cfg.epochs
    batch_size, lr, num_workers = cfg.batch_size, cfg.lr, cfg.num_workers
    out_dir = _RESULTS / cfg.exp_name
    last_ckpt = out_dir / "latent_classifier.pt"
    best_ckpt = out_dir / "latent_classifier_best.pt"
    train_ds = LatentDataset(load_cases("train"), "train")
    eval_ds = LatentDataset(load_cases("valid_v2"), "valid_v2")
    print(f"train latents={len(train_ds)}  valid_v2 latents={len(eval_ds)}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = LatentClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    best_auc = -1.0
    best = None  # (state_dict, logits, labels, per) — val_mean_auc 최대 시점 스냅샷
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y, _ in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            running += loss.item()
        train_loss = running / len(train_loader)
        logits, labels, _ = _predict(model, eval_loader, device)
        auc, per = mean_auc(labels, logits)
        val_loss = loss_fn(
            torch.from_numpy(logits), torch.from_numpy(labels).float()
        ).item()
        history.append(
            {
                "epoch": ep,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mean_auc": auc,
            }
        )
        print(
            f"epoch {ep:02d}/{epochs}  train_loss={train_loss:.4f}  valid_loss={val_loss:.4f}"
            f"  valid_v2 mean-AUC={auc:.4f}",
            flush=True,
        )
        if auc > best_auc:
            best_auc = auc
            best = (
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                logits,
                labels,
                per,
            )
            print(f"  ↑ new best mean-AUC={auc:.4f}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), last_ckpt)  # last 에폭

    # 이후 img_feat 추출·Probe C 리포트는 best 가중치/지표로 통일.
    best_state, logits, labels, per = best
    torch.save(best_state, best_ckpt)
    model.load_state_dict(best_state)
    auc = best_auc

    # Probe C 보고 + img_feat 추출(train + valid_v2). best 에폭의 logits/labels/auc/per 재사용.
    print(
        f"\n[Probe C] valid_v2 mean-AUC={auc:.4f}  ({'PASS' if auc >= 0.7 else 'BELOW 0.7'})"
    )
    for name, v in sorted(per.items(), key=lambda kv: -kv[1]):
        print(f"  {name:32s} {v:.3f}")

    eval_feats = None
    for split, ds in (("train", train_ds), ("valid_v2", eval_ds)):
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        feats, ids = _extract_features(model, loader, device)
        if split == "valid_v2":
            eval_feats = feats
        save_npz(EMB_ROOT / split / "img_feat.npz", emb=feats, scan_ids=ids)
        print(
            f"saved img_feat {feats.shape} → {EMB_ROOT / split / 'img_feat.npz'}",
            flush=True,
        )

    # results/<exp_name>/ 산출물: 곡선·label별 AUC·confusion matrix·metrics(img_feat cosine 포함).
    cosine = mean_pairwise_cosine(eval_feats)
    save_loss_curve(history, out_dir)
    save_auc_curve(history, out_dir)
    save_per_label_auc(per, out_dir)
    save_confusion_matrices(labels, logits, out_dir)
    save_metrics_json(out_dir, auc, per, history, cosine)
    save_roc_curve(labels, logits, out_dir)
    print(
        f"saved results → {out_dir}/  (img_feat cosine raw={cosine['raw']:.3f} "
        f"centered={cosine['centered']:.3f})",
        flush=True,
    )
