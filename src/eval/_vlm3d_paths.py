"""Single source of truth for the VLM3D-Dockers ctgen-evaluation directory.

Upstream commit a945900 ("Reorganize into ct_challenges") moved every challenge
dir under ``ct_challenges/`` — ``ctgen_evaluation/`` became
``ct_challenges/ctgen_evaluation/``. Rather than hard-code that path in several
modules (and break again on the next reorg), every consumer imports
``ctgen_eval_dir()`` from here. It prefers the new ``ct_challenges/`` layout and
falls back to the old top-level path, so the code works against either pin.
"""

from __future__ import annotations

from pathlib import Path

_SUBMODULE_ROOT = Path("/workspace/third_party/vlm3d_dockers")


def ctgen_eval_dir() -> Path:
    """Return the ctgen evaluation dir, preferring the ct_challenges layout.

    Falls back to the pre-a945900 top-level path if the new one is absent, so a
    stale submodule pin still resolves correctly.
    """
    new = _SUBMODULE_ROOT / "ct_challenges" / "ctgen_evaluation"
    old = _SUBMODULE_ROOT / "ctgen_evaluation"
    return new if new.is_dir() else old


def ctclip_pkg_parents() -> list[Path]:
    """Parent dirs to put on PYTHONPATH so `import transformer_maskgit` / `ct_clip`
    resolve, for evaluate_clip.py's CLIPScore.

    The upstream Dockerfile `pip install -e`s these from a CT-CLIP/ tree; locally
    that editable mapping breaks whenever the submodule moves (the a945900 reorg).
    We instead discover the package *parent* dirs under the submodule and add them
    to PYTHONPATH — the parent (not the package dir itself) so `import ct_clip.mlm`
    still works. ctgen_evaluation/CT-CLIP/ is empty in-repo (populated at docker
    build), so any sibling docker's vendored CT-CLIP copy is used.
    """
    parents: list[Path] = []
    for pkg in ("transformer_maskgit", "ct_clip"):
        for cand in sorted(_SUBMODULE_ROOT.rglob(f"{pkg}/__init__.py")):
            parent = (
                cand.parent.parent
            )  # .../<X>/<pkg>/__init__.py → put .../<X> on path
            if parent not in parents:
                parents.append(parent)
                break
    return parents


__all__ = ["ctgen_eval_dir", "ctclip_pkg_parents"]
