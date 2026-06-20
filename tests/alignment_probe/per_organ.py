"""Unit 4b — per-organ faithful fVLM alignment (plan Part B).

global Probe B(concat-4-organ)는 fVLM의 anatomy-local 강점을 평균내 과소평가한다. 여기서는
**neutral per-organ image feature** `z_organ`을 만들어 fVLM의 per-organ text가 그걸 잘 예측하는지
(locality에 공정한 평가) 본다.

z_organ (neutral, 256-d/organ): z② 분류기의 pre-pool feature map `(256,15,15,8)`을 organ
occupancy로 가중 풀링. 재학습/OOD 없음, 5개 text 인코더와 독립.
  organ mask는 MAISI latent과 동일 프레임(RAS→Resize(120,120,64) nearest)으로 정렬.

평가는 **proxy 내부 5-fold CV**(TS mask가 1024³ 거대 파일이라 scan당 ~3.6s; train 5000은 비현실적
→ proxy 425만; latent 공간 동일하므로 cross-split 불필요). out-of-fold(OOF) 예측으로 leakage 없음.
  측정1 (core): per-organ RidgeCV  fVLM organ-text → z_organ_o, R²/Pearson.
  측정2 (locality): organ-routing top-1 acc — organ-o text가 같은 scan의 4개 organ 중 o를 맞히는가.

사용: CUDA_VISIBLE_DEVICES=1 python -m tests.alignment_probe.per_organ --device cuda:0
출력: results/per_organ.json + embeddings/valid_v2/z_organ.npz 캐시.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import monai
import monai.data
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from torch.utils.data import Dataset

from src.baselines.fvlm_preprocess import TS_LABEL_TO_ORGAN, ts_mask_path
from src.data.fvlm_organ_report import ORGANS  # (lung, heart, esophagus, aorta)
from tests.alignment_probe.cases import load_cases
from tests.alignment_probe.classifier import Latent3DCNN
from tests.alignment_probe.latents import has_latent, load_latent
from tests.alignment_probe.probe import _ALPHAS, _emb_dict, _mean_pearson

_CT_ROOT = Path("/workspace/datasets/datasets/CT-RATE/dataset")
_CKPT = Path("tests/alignment_probe/z_classifier.pt")
_EMB_ROOT = Path("tests/alignment_probe/embeddings")
_OUT = Path("tests/alignment_probe/results/per_organ.json")
_SPLIT = "valid_v2"
_GRID = (120, 120, 64)  # MAISI latent 공간 = feature map의 8×
_POOL = 8  # 120/8=15, 64/8=8 → feature map (256,15,15,8) 그리드와 일치
_SEED = 0

_MASK_TF = (
    monai.transforms.Compose(  # report2ct_image_encoder와 동일 프레임(mask는 nearest)
        [
            monai.transforms.LoadImaged(keys="mask"),
            monai.transforms.EnsureChannelFirstd(keys="mask"),
            monai.transforms.Orientationd(keys="mask", axcodes="RAS"),
            monai.transforms.Resized(keys="mask", spatial_size=_GRID, mode="nearest"),
        ]
    )
)


def _ct_path(scan_id: str) -> Path:
    """'valid_1000_a_1' → valid_fixed/valid_1000/valid_1000_a/valid_1000_a_1.nii.gz."""
    parts = scan_id.split("_")  # [split, patient#, series, recon]
    return (
        _CT_ROOT
        / f"{parts[0]}_fixed"
        / "_".join(parts[:2])
        / "_".join(parts[:3])
        / f"{scan_id}.nii.gz"
    )


def _organ_occupancy(scan_id: str) -> torch.Tensor | None:
    """scan의 TS mask → organ occupancy ``(4, 15, 15, 8)`` (0~1 분율), mask 없으면 None."""
    mpath = ts_mask_path(_ct_path(scan_id))
    if not mpath.is_file():
        return None
    arr = (
        _MASK_TF({"mask": str(mpath)})["mask"][0].numpy().astype(np.int32)
    )  # (120,120,64)
    organ = np.zeros_like(arr)
    for ts_id, oid in TS_LABEL_TO_ORGAN.items():
        organ[arr == ts_id] = oid  # TS label → {1:lung,2:heart,3:esophagus,4:aorta}
    t = torch.from_numpy(organ)
    occ = [
        F.avg_pool3d((t == oid).float()[None, None], _POOL)[0, 0]  # (15,15,8)
        for oid in (1, 2, 3, 4)
    ]
    return torch.stack(occ)  # (4,15,15,8)


class _OrganDataset(Dataset):
    """병렬 로딩용 — __getitem__ → {latent (4,120,120,64), occ (4,15,15,8), scan_id}.

    거대 TS mask(1024³, scan당 ~3.6s)가 병목이므로 ``monai.data.DataLoader``의 worker로 병렬 로딩.
    """

    def __init__(self) -> None:
        self.cases = [
            c
            for c in load_cases(_SPLIT)
            if has_latent(c.scan_id, _SPLIT)
            and ts_mask_path(_ct_path(c.scan_id)).is_file()
        ]

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, i: int) -> dict:
        c = self.cases[i]
        return {
            "latent": load_latent(c.scan_id, _SPLIT),
            "occ": _organ_occupancy(c.scan_id),  # (4,15,15,8)
            "scan_id": c.scan_id,
        }


@torch.no_grad()
def build_z_organ(model: Latent3DCNN, device: str, num_workers: int = 8):
    """proxy z_organ → (z (N,4,256), scan_ids, present). monai.data.DataLoader로 병렬 로딩."""
    loader = monai.data.DataLoader(
        _OrganDataset(), batch_size=1, num_workers=num_workers
    )
    Z, ids, present = [], [], []
    for n, b in enumerate(loader):
        fm = model.feature_map(b["latent"].to(device))[0].cpu()  # (256,15,15,8)
        occ = b["occ"][0]  # (4,15,15,8)
        z_scan, pres = [], []
        for o in range(4):
            s = float(occ[o].sum())
            z_scan.append(
                (fm * occ[o]).flatten(1).sum(1) / s if s > 0 else torch.zeros(256)
            )
            pres.append(s > 0)
        Z.append(torch.stack(z_scan).numpy())
        ids.append(b["scan_id"][0])
        present.append(pres)
        if (n + 1) % 100 == 0:
            print(f"  z_organ {n + 1}/{len(loader)}", flush=True)
    return np.stack(Z), np.array(ids), np.array(present)


def _cv_predict(X: np.ndarray, Z: np.ndarray):
    """5-fold OOF: 표준화 RidgeCV → (fold R²/Pearson 평균, OOF raw 예측 (N,256))."""
    oof = np.zeros_like(Z)
    r2s, prs = [], []
    for tr, te in KFold(5, shuffle=True, random_state=_SEED).split(X):
        xsc, zsc = StandardScaler().fit(X[tr]), StandardScaler().fit(Z[tr])
        ridge = RidgeCV(alphas=_ALPHAS).fit(xsc.transform(X[tr]), zsc.transform(Z[tr]))
        pred = ridge.predict(xsc.transform(X[te]))
        true = zsc.transform(Z[te])
        r2s.append(r2_score(true, pred, multioutput="variance_weighted"))
        prs.append(_mean_pearson(true, pred))
        oof[te] = zsc.inverse_transform(pred)  # raw z² 공간(routing용)
    return float(np.mean(r2s)), float(np.mean(prs)), oof


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    model = Latent3DCNN().eval()
    model.load_state_dict(torch.load(_CKPT, map_location="cpu", weights_only=True))
    model.to(args.device)

    cache = _EMB_ROOT / _SPLIT / "z_organ.npz"
    if cache.is_file():
        d = np.load(cache, allow_pickle=True)
        Z, ids, present = d["z"], d["scan_ids"], d["present"]
    else:
        Z, ids, present = build_z_organ(model, args.device)
        np.savez(cache, z=Z, scan_ids=ids, present=present)
        print(f"  built z_organ {Z.shape} → {cache}", flush=True)

    fvlm = np.load(_EMB_ROOT / _SPLIT / "fvlm.npz", allow_pickle=True)
    ftext = dict(zip(fvlm["scan_ids"], fvlm["emb_organs"]))  # scan → (4,256)
    keep = [i for i, s in enumerate(ids) if s in ftext]  # fvlm 임베딩 있는 scan
    Z, ids, present = Z[keep], ids[keep], present[keep]
    Xall = np.stack([ftext[s] for s in ids])  # (N,4,256)

    results: dict = {"n": int(len(ids))}

    # V-PO1 coverage
    cov = {ORGANS[o]: round(float(present[:, o].mean()), 3) for o in range(4)}
    results["coverage"] = cov
    print(f"V-PO1 organ coverage (n={len(ids)}): {cov}")

    # V-PO2 inter-organ cosine (organ 간 z_organ이 구별되는가)
    nz = Z / (np.linalg.norm(Z, axis=2, keepdims=True) + 1e-8)
    io = float(
        np.mean(
            [
                np.sum(nz[:, a] * nz[:, b], 1).mean()
                for a in range(4)
                for b in range(a + 1, 4)
            ]
        )
    )
    results["inter_organ_cosine"] = round(io, 3)
    print(f"V-PO2 mean inter-organ cosine: {io:.3f}")

    # 측정1: per-organ predictability (5-fold OOF) + OOF raw 예측 보관
    per_organ, oof_raw = {}, {}
    for o in range(4):
        sel = present[:, o]
        r2, pear, oof = _cv_predict(Xall[sel, o], Z[sel, o])
        per_organ[ORGANS[o]] = {"r2": r2, "pearson": pear, "n": int(sel.sum())}
        oof_raw[o] = dict(zip(ids[sel], oof))
    results["per_organ"] = per_organ
    mean_r2 = float(np.mean([v["r2"] for v in per_organ.values()]))
    results["per_organ_mean_r2"] = round(mean_r2, 3)
    print("\n측정1 per-organ predictability (fVLM organ-text → z_organ):")
    for name, v in per_organ.items():
        print(
            f"  {name:10s} R²={v['r2']:+.3f}  Pearson r={v['pearson']:+.3f}  (n={v['n']})"
        )
    print(f"  mean R²={mean_r2:.3f}  (vs fVLM global Probe B = 0.282)")

    # V-PO5: global encoder baseline — whole-report text → 같은 per-organ target.
    # global은 organ-specific 입력이 없어(scan당 벡터 1개) per-organ을 구별 못 함 → fVLM organ-text의
    # localized predictability 우위 여부를 가늠하는 기준선.
    compare = {
        "fvlm_organ": {
            ORGANS[o]: round(per_organ[ORGANS[o]]["r2"], 3) for o in range(4)
        }
    }
    for enc in ("ctclip", "text2ct", "t5", "report2ct"):
        emb = _emb_dict(enc, _SPLIT)  # scan → whole-report vec
        row = {}
        for o in range(4):
            sel = present[:, o]
            X = np.stack([emb[s] for s in ids[sel]])
            r2, _, _ = _cv_predict(X, Z[sel, o])
            row[ORGANS[o]] = round(r2, 3)
        compare[enc] = row
    for row in compare.values():
        row["mean"] = round(float(np.mean([row[ORGANS[o]] for o in range(4)])), 3)
    results["compare_per_organ_r2"] = compare
    print("\nV-PO5 per-organ R² — fVLM organ-text vs global whole-report:")
    print(
        "  "
        + "".join(f"{ORGANS[o][:5]:>8s}" for o in range(4))
        + f"{'mean':>8s}   encoder"
    )
    for name, row in compare.items():
        print(
            "  "
            + "".join(f"{row[ORGANS[o]]:+8.3f}" for o in range(4))
            + f"{row['mean']:+8.3f}   {name}"
        )

    # 측정2: organ-routing top-1 acc — 4 organ 모두 present + 모두 OOF 있는 scan
    routable = [
        s for s in ids[present.all(1)] if all(s in oof_raw[o] for o in range(4))
    ]
    ztrue = dict(zip(ids, Z))
    hits = total = 0
    for s in routable:
        for o in range(4):
            zhat = oof_raw[o][s]  # raw z² 공간
            sims = [
                float(
                    zhat
                    @ ztrue[s][op]
                    / (np.linalg.norm(zhat) * np.linalg.norm(ztrue[s][op]) + 1e-8)
                )
                for op in range(4)
            ]
            hits += int(np.argmax(sims) == o)
            total += 1
    acc = hits / max(total, 1)
    results["routing_top1_acc"] = round(acc, 3)
    results["routing_n_scans"] = len(routable)
    print(
        f"\n측정2 organ-routing top-1 acc = {acc:.3f}  (chance 0.25, n_scans={len(routable)})"
    )

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(results, indent=2))
    print(f"\nsaved → {_OUT}")


if __name__ == "__main__":
    main()
