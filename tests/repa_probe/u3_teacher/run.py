"""U3 — teacher 적합성 판정: 어떤 SPECTRE 체크포인트/레이어를, spatial norm 유무로 쓸 것인가.

REPA가 실제로 전달하는 신호는 **공간 구조**(토큰 간 상대 유사도)라는 게 iREPA의 결론이다
(27개 인코더에서 spatial metric ↔ gFID |r| > 0.85, linear probing은 0.26). 그래서 학습을 한 번도
돌리지 않고 teacher 후보를 같은 축에서 비교한다.

축:
    teacher      ssl (DINO+iBOT+KoLeo) vs vla (+SigLIP report alignment)
    layer        11 / 17 / 23 (depth 24)
    spatial norm off / on (iREPA — teacher 토큰에서 global 성분 제거)
    토큰 범위    전체 vs **몸통만** (CT는 절반이 공기고, 공기끼리는 서로 비슷해서 지표를 부풀린다)

지표: LDS · CDS · RMSC · SRSS(lung/liver/heart/aorta, ts_seg 마스크) · crop seam gap
그림: anchor cos-sim map (ssl/vla × norm on/off)

실행:
    CUDA_VISIBLE_DEVICES=3 python -m tests.repa_probe.u3_teacher.run [--n-volumes 24]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tests.repa_probe._metrics import all_metrics, spatial_norm
from tests.repa_probe._spectre import (
    CKPT_SSL,
    CKPT_VLA,
    EMBED_DIM,
    PROBE_ORGANS,
    TOKEN_GRID,
    body_occupancy,
    build_backbone,
    load_volume,
    organ_occupancy,
)

IDS_FILE = Path("/workspace/data/ctrate_toy_v2/valid_v2/ids.json")
TEACHERS = {"ssl": CKPT_SSL, "vla": CKPT_VLA}
LAYERS = (
    11,
    17,
    23,
)  # 마지막(23)은 forward_features와 동치 — test_spectre_adapter가 고정
OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figs"
RESULT_DIR = OUT_DIR / "results"
OCCUPANCY_THRESHOLD = 0.5


def volume_ids(n: int) -> list[str]:
    data = json.loads(IDS_FILE.read_text())
    return list(data["ids"] if isinstance(data, dict) else data)[:n]


def cossim_map_figure(
    dense_by_arm: dict[str, torch.Tensor], lung_occ: torch.Tensor, scan_id: str
) -> str:
    """4개 arm(ssl/vla × norm off/on)의 anchor cos-sim map을 한 장에 겹쳐 그린다.

    anchor는 폐 내부에서 고정으로 잡아 arm 간 비교가 가능하게 한다.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    lung_flat = lung_occ.reshape(-1)
    anchor = int(
        torch.nonzero(lung_flat >= 0.999).flatten()[
            len(torch.nonzero(lung_flat >= 0.999)) // 2
        ]
    )
    ai, aj, ak = (int(v) for v in np.unravel_index(anchor, TOKEN_GRID))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    arms = list(dense_by_arm)
    fig, axes = plt.subplots(len(arms), 3, figsize=(11, 2.6 * len(arms)))
    for row, arm in enumerate(arms):
        feat = torch.nn.functional.normalize(dense_by_arm[arm].float(), dim=-1)
        sim = (feat @ feat[anchor]).reshape(*TOKEN_GRID)
        for col, (title, plane) in enumerate(
            [
                (f"axial D={ak}", sim[:, :, ak]),
                (f"coronal W={aj}", sim[:, aj, :]),
                (f"sagittal H={ai}", sim[ai, :, :]),
            ]
        ):
            im = axes[row, col].imshow(
                plane.T.cpu().numpy(), cmap="turbo", vmin=-0.5, vmax=1.0
            )
            axes[row, col].set_title(f"{arm} | {title}", fontsize=8)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if col == 2:
                plt.colorbar(im, ax=axes[row, col], fraction=0.046)
    fig.suptitle(
        f"anchor cos-sim maps, layer 23, anchor in lung (h,w,d)=({ai},{aj},{ak}) — {scan_id}",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG_DIR / "U3_cossim_maps.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return str(out.relative_to(Path("/workspace")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-volumes", type=int, default=24)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True  # U2b 실측: cos_min 0.99998, 3.7× 빠름
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids = volume_ids(args.n_volumes)
    backbones = {t: build_backbone(ckpt, device=device) for t, ckpt in TEACHERS.items()}

    per_arm: dict[str, list[dict]] = {}
    fig_path = None
    for i, scan_id in enumerate(ids):
        vol = load_volume(scan_id)
        crops, grid = backbones["ssl"].window(vol)
        crops = crops.to(device)

        organs = {
            name: (occ.reshape(-1) > OCCUPANCY_THRESHOLD).to(device)
            for name, occ in organ_occupancy(scan_id, PROBE_ORGANS).items()
        }
        body = (body_occupancy(scan_id).reshape(-1) > OCCUPANCY_THRESHOLD).to(device)

        fig_arms: dict[str, torch.Tensor] = {}
        for teacher in TEACHERS:
            for layer in LAYERS:
                dense, _ = backbones[teacher].encode_crops(crops, grid, layer=layer)
                flat = dense.reshape(-1, EMBED_DIM).float()  # (32768, 1080)
                for norm in (False, True):
                    tokens = spatial_norm(flat) if norm else flat
                    arm = f"{teacher}_L{layer}_{'norm' if norm else 'raw'}"
                    for scope, subset in (("all", None), ("body", body)):
                        row = all_metrics(
                            tokens,
                            TOKEN_GRID,
                            organ_masks=organs,
                            subset=subset,
                            seed=i,
                        )
                        per_arm.setdefault(f"{arm}|{scope}", []).append(row)
                    if layer == LAYERS[-1] and i == 0:
                        fig_arms[f"{teacher}_{'norm' if norm else 'raw'}"] = (
                            tokens.cpu()
                        )
                del dense, flat
        if i == 0:
            fig_path = cossim_map_figure(
                fig_arms, organ_occupancy(scan_id, ("lung_",))["lung_"], scan_id
            )
            del fig_arms
        print(f"[{i + 1}/{len(ids)}] {scan_id}", flush=True)

    summary = {}
    for key, rows in per_arm.items():
        keys = [k for k in rows[0] if k != "n_tokens"]
        summary[key] = {
            k: {
                "mean": float(np.nanmean([r[k] for r in rows])),
                "std": float(np.nanstd([r[k] for r in rows])),
            }
            for k in keys
        }
        summary[key]["n_tokens_mean"] = float(np.mean([r["n_tokens"] for r in rows]))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / "teacher_spatial.json"
    out.write_text(
        json.dumps(
            {"n_volumes": len(ids), "ids": ids, "figure": fig_path, "arms": summary},
            indent=2,
        )
    )
    print(f"\n[done] {out}")

    # 콘솔 요약 — body scope, LDS 내림차순
    print(
        f"\n{'arm':28s} {'LDS':>8s} {'CDS':>8s} {'RMSC':>8s} {'SRSS_lung':>10s} {'seam_w':>8s} {'seam_a':>8s} {'seam_drop':>10s} {'cos_mean':>9s}"
    )
    body_rows = {k: v for k, v in summary.items() if k.endswith("|body")}
    for key in sorted(body_rows, key=lambda k: -body_rows[k]["LDS"]["mean"]):
        s = body_rows[key]
        print(
            f"{key.split('|')[0]:28s} {s['LDS']['mean']:8.4f} {s['CDS']['mean']:8.4f} "
            f"{s['RMSC']['mean']:8.4f} {s['SRSS_lung_']['mean']:10.4f} "
            f"{s['seam_within']['mean']:8.4f} {s['seam_across']['mean']:8.4f} "
            f"{s['seam_drop']['mean']:10.4f} {s['cos_mean']['mean']:9.4f}"
        )


if __name__ == "__main__":
    main()
