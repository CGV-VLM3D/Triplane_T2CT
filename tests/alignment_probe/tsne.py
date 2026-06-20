"""Unit 5 (보조) — 5개 text 임베딩 + visual z② t-SNE, abnormality별 present/absent 색칠.

단독 근거 아님(가설 판정은 Probe B). "같은 abnormality가 더 잘 뭉치는가"를 나란히 보고,
perplexity 여러 개로 robustness를 확인하며, silhouette score로 정량 동반한다.
fVLM Fig 12(visual 기준)를 그대로 복제하지 않는다 — text와 visual을 같은 점 집합에서 비교.

모든 rep을 z②가 있는 공통 scan(=visual과 동일 점)으로 제한해 공정 비교.
사용: python -m tests.alignment_probe.tsne
출력: results/tsne_p<perp>.png (primary) + results/tsne_silhouette.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

from tests.alignment_probe.cases import ABNORMALITY_LABELS  # noqa: E402
from tests.alignment_probe.probe import (  # noqa: E402
    _emb_dict,
    _encoders,
    _label_dict,
    _z_dict,
)

_OUT = Path("tests/alignment_probe/results")
_PERPLEXITIES = (5, 30, 50)
_PRIMARY = 30
_TOP_K_ABN = 4  # valid_v2에서 가장 흔한 abnormality 몇 개만 패널로
_SEED = 0


def _common_reps() -> tuple[dict[str, np.ndarray], list[str]]:
    """{rep_name: emb_matrix}, scan_ids — z②가 있는 공통 scan으로 정렬·제한."""
    zte = _z_dict("valid_v2")
    if zte is None:
        raise SystemExit("z②(z_classifier.npz) 미존재 — classifier.py 먼저 실행.")
    reps_raw = {enc: _emb_dict(enc, "valid_v2") for enc in _encoders()}
    reps_raw["visual_z"] = zte
    ids = sorted(set.intersection(*(set(d) for d in reps_raw.values())))
    reps = {name: np.stack([d[i] for i in ids]) for name, d in reps_raw.items()}
    return reps, ids


def _top_abnormalities(ids: list[str]) -> list[str]:
    labd = _label_dict("valid_v2")
    Y = np.stack([labd[i] for i in ids])  # (N, 18)
    prev = Y.sum(0)
    order = np.argsort(-prev)
    return [ABNORMALITY_LABELS[j] for j in order[:_TOP_K_ABN]]


def main() -> None:
    reps, ids = _common_reps()
    abns = _top_abnormalities(ids)
    labd = _label_dict("valid_v2")
    Y = {a: np.array([labd[i][ABNORMALITY_LABELS.index(a)] for i in ids]) for a in abns}
    print(f"t-SNE on {len(ids)} common scans; reps={list(reps)}; abns={abns}\n")

    # rep × perplexity → 2D 좌표 + abnormality별 silhouette
    coords: dict[tuple[str, int], np.ndarray] = {}
    sil: dict = {}
    for name, X in reps.items():
        sil[name] = {}
        for perp in _PERPLEXITIES:
            emb2d = TSNE(
                n_components=2, perplexity=perp, init="pca", random_state=_SEED
            ).fit_transform(X)
            coords[(name, perp)] = emb2d
            sil[name][perp] = {
                a: (
                    float(silhouette_score(emb2d, Y[a]))
                    if len(np.unique(Y[a])) == 2
                    else None
                )
                for a in abns
            }
        sline = "  ".join(
            f"{a[:12]}:"
            + "/".join(
                f"{sil[name][p][a]:.2f}" if sil[name][p][a] is not None else "NA"
                for p in _PERPLEXITIES
            )
            for a in abns
        )
        print(f"  {name:10s} silhouette(p={_PERPLEXITIES}) {sline}")

    # primary perplexity 패널: rows=rep, cols=abnormality, present/absent 2색
    names = list(reps)
    fig, axes = plt.subplots(
        len(names), len(abns), figsize=(3 * len(abns), 3 * len(names)), squeeze=False
    )
    for r, name in enumerate(names):
        xy = coords[(name, _PRIMARY)]
        for c, a in enumerate(abns):
            ax = axes[r][c]
            m = Y[a] == 1
            ax.scatter(xy[~m, 0], xy[~m, 1], s=3, c="lightgray", label="absent")
            ax.scatter(xy[m, 0], xy[m, 1], s=3, c="crimson", label="present")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name, fontsize=9)
            if r == 0:
                ax.set_title(a, fontsize=8)
    fig.suptitle(f"t-SNE (perplexity={_PRIMARY}) — present(red) vs absent(gray)")
    fig.tight_layout()
    _OUT.mkdir(parents=True, exist_ok=True)
    png = _OUT / f"tsne_p{_PRIMARY}.png"
    fig.savefig(png, dpi=120)
    (_OUT / "tsne_silhouette.json").write_text(json.dumps(sil, indent=2))
    print(f"\nsaved → {png}\nsaved → {_OUT / 'tsne_silhouette.json'}")


if __name__ == "__main__":
    main()
