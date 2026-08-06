"""04_report — CT-RATE Report/Text EDA (GPT section 5 verify + extend).

Unit = SCAN (report + 18 labels are constant per scan; we drop reconstruction
duplicates). Population = CLEAN (exclude no_chest + unencodable), taken from
/workspace/data/ctrate_full/{split}/ids.json (train 46,393 vol / valid 3,001 vol).
A scan is kept if it has >=1 clean volume -> train 24,128 scans / valid 1,564 scans.

Read-only on datasets. Self-contained + re-runnable.
Outputs under /workspace/tests/ctrate_eda/ (tables/, figures/, samples/).
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/workspace")

import json
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA_ROOT = Path("/workspace/datasets/datasets/CT-RATE/dataset")
FULL_CENSUS = Path("/workspace/data/ctrate_full")
OUT = Path("/workspace/tests/ctrate_eda")
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)
(OUT / "samples").mkdir(parents=True, exist_ok=True)

TRAIN_BLUE = "#1f77b4"
VALID_ORANGE = "#ff7f0e"

FIELDS = ["ClinicalInformation_EN", "Technique_EN", "Findings_EN", "Impressions_EN"]

LABELS_18 = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]

# Synonym / surface-term sets per label (lowercased regex-ish substrings).
LABEL_SYNONYMS = {
    "Medical material": [
        "stent",
        "catheter",
        "pacemaker",
        "port-a-cath",
        "port a cath",
        "chest tube",
        "drainage tube",
        "surgical material",
        "sternotomy wire",
        "sternal wire",
        "prosthesis",
        "prosthetic",
        "surgical clip",
        "metallic density",
        "foreign material",
        "vascular graft",
    ],
    "Arterial wall calcification": [
        "aortic calcification",
        "aorta calcific",
        "calcific plaque",
        "arterial wall calcification",
        "vascular calcification",
        "atherosclerotic calcific",
        "calcified plaque",
        "calcific atheroma",
        "wall calcification in the aorta",
    ],
    "Cardiomegaly": [
        "cardiomegaly",
        "enlarged heart",
        "cardiac enlargement",
        "increased heart size",
        "heart size is increased",
        "enlarged cardiac silhouette",
        "heart contour size increased",
        "cardiothoracic ratio increased",
    ],
    "Pericardial effusion": [
        "pericardial effusion",
        "pericardial fluid",
        "fluid in the pericardium",
    ],
    "Coronary artery wall calcification": [
        "coronary artery calcification",
        "coronary calcification",
        "coronary artery wall calcification",
        "coronary calcific",
        "calcification in the coronary",
        "calcified coronary",
    ],
    "Hiatal hernia": ["hiatal hernia", "hiatus hernia", "diaphragmatic hernia"],
    "Lymphadenopathy": [
        "lymphadenopathy",
        "enlarged lymph node",
        "lymph node enlargement",
        "pathological lymph node",
        "significant lymph node",
        "lymphadenomegaly",
        "in pathological size",
        "pathological dimension",
    ],
    "Emphysema": ["emphysema", "emphysematous"],
    "Atelectasis": ["atelectasis", "atelectatic", "atelectas"],
    "Lung nodule": ["nodule", "nodular densit", "nodular lesion"],
    "Lung opacity": [
        "opacity",
        "opacities",
        "ground-glass",
        "ground glass",
        "infiltrat",
        "reticulonodular densit",
        "consolidat",
    ],
    "Pulmonary fibrotic sequela": [
        "fibrotic",
        "fibrosis",
        "fibrotic sequela",
        "sequela",
        "scarring",
        "scar",
        "traction bronchiectasis",
        "parenchymal band",
    ],
    "Pleural effusion": [
        "pleural effusion",
        "pleural fluid",
        "effusion in the pleural",
    ],
    "Mosaic attenuation pattern": [
        "mosaic attenuation",
        "mosaic pattern",
        "mosaic perfusion",
        "mosaic densit",
    ],
    "Peribronchial thickening": [
        "peribronchial thickening",
        "bronchial wall thickening",
        "peribronchial",
        "thickening of the bronchial wall",
        "wall thickening in the bronch",
    ],
    "Consolidation": ["consolidation", "consolidations", "consolidative"],
    "Bronchiectasis": ["bronchiectasis", "bronchiectatic", "bronchiectas"],
    "Interlobular septal thickening": [
        "interlobular septal thickening",
        "septal thickening",
        "interlobular septal",
        "interlobular thickening",
    ],
}

# ---- NegEx-style cue lexicons (sentence-scoped, window-bounded) ----
PRE_NEG = [
    "no ",
    "no significant",
    "no evidence of",
    "without",
    "absence of",
    "free of",
    "negative for",
    "not ",
]
POST_NEG = [
    "was not observed",
    "were not observed",
    "not observed",
    "not detected",
    "was not detected",
    "were not detected",
    "not seen",
    "was not seen",
    "were not seen",
    "is not present",
    "are not present",
    "was not present",
    "were not present",
    "not identified",
    "ruled out",
    "excluded",
    "could not be detected",
    "was not found",
    "were not found",
]
UNCERTAIN = [
    "suspicious",
    "suspected",
    "possible",
    "possibly",
    "probable",
    "probably",
    "may represent",
    "may be",
    "cannot be excluded",
    "can not be excluded",
    "suggestive of",
    "likely",
    "concerning for",
    "questionable",
    "?",
    "in favor of",
    "primarily in favor",
    "consistent with",
    "consider",
]
TERMINATION = [" but ", " however ", "; ", " although ", " otherwise ", " except "]

NORMAL_WORDS = [
    "normal",
    "natural",
    "unremarkable",
    "preserved",
    "are open",
    "is open",
    "patent",
    "no significant",
    "within normal",
    "not observed",
    "not detected",
    "not seen",
]
COMPARISON_WORDS = [
    "compared",
    "comparison",
    "previous",
    "prior",
    "interval",
    "unchanged",
    "stable",
    "increased in size compared",
    "decreased compared",
    "regressed",
    "progression",
    "when compared",
]
RECOMMENDATION_WORDS = [
    "recommend",
    "advised",
    "suggest follow",
    "follow-up",
    "follow up",
    "correlation",
    "correlate clinically",
    "further evaluation",
    "control",
    "should be",
    "is recommended",
]

SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+")
NOT_GIVEN = re.compile(r"^\s*(not given|none|n/?a|-)\.?\s*$", re.IGNORECASE)


def load_clean_scan_reports(split: str) -> pd.DataFrame:
    """Return scan-deduped clean report+label DataFrame for a split.

    Columns: scan_key + the 4 report fields + 18 labels + n_labels.
    """
    census = json.load(open(FULL_CENSUS / split / "ids.json"))
    clean_vols = set(census["ids"])  # e.g. 'train_10000_a_1' (no extension)
    clean_scans = set("_".join(v.split("_")[:3]) for v in clean_vols)

    rep_name = "train_reports.csv" if split == "train" else "validation_reports.csv"
    lab_name = (
        "train_predicted_labels.csv"
        if split == "train"
        else "valid_predicted_labels.csv"
    )
    reports = pd.read_csv(DATA_ROOT / "radiology_text_reports" / rep_name)
    labels = pd.read_csv(DATA_ROOT / "multi_abnormality_labels" / lab_name)

    df = reports.merge(labels, on="VolumeName", how="inner")
    df["scan_key"] = (
        df["VolumeName"]
        .str.replace(".nii.gz", "", regex=False)
        .apply(lambda v: "_".join(v.split("_")[:3]))
    )
    df = df[df["scan_key"].isin(clean_scans)].copy()
    # dedup to scan (report + labels constant per scan) -> keep first reconstruction row
    df = (
        df.sort_values("VolumeName")
        .drop_duplicates("scan_key", keep="first")
        .reset_index(drop=True)
    )
    for f in FIELDS:
        df[f] = df[f].fillna("").astype(str)
    df["n_labels"] = df[LABELS_18].sum(axis=1).astype(int)
    return df


# ---------- text primitives ----------
def is_missing(text: str) -> bool:
    t = (text or "").strip()
    return t == "" or bool(NOT_GIVEN.match(t))


def sentences(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    return [s.strip() for s in SENT_SPLIT.split(t) if s.strip()]


def word_count(text: str) -> int:
    return len((text or "").split())


def norm_sentence(s: str) -> str:
    """Punct-stripped, lowercased, whitespace-collapsed sentence for near-dup clustering."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------- NegEx-style assertion ----------
def assertion_status(sentence: str, term_span: tuple[int, int]) -> str:
    """Classify a term mention in a sentence as affirmed / negated / uncertain.

    Window-bounded NegEx: scan the sentence around the matched span, but do not
    cross a termination cue (but/however/;...) into the term's scope.
    """
    s = sentence.lower()
    start, end = term_span
    # bound scope by nearest termination cue on each side
    left = 0
    right = len(s)
    for cue in TERMINATION:
        li = s.rfind(cue, 0, start)
        if li != -1:
            left = max(left, li + len(cue))
        ri = s.find(cue, end)
        if ri != -1:
            right = min(right, ri)
    scope = s[left:right]
    scope_before = s[left:start]
    scope_after = s[end:right]

    # uncertainty anywhere in scope
    if any(u in scope for u in UNCERTAIN):
        return "uncertain"
    # post-negation (term ... was not observed) — common in CT-RATE
    if any(neg in scope_after for neg in POST_NEG):
        return "negated"
    # pre-negation (no / not / without ... term) within ~6 words before
    pre_window = " ".join(scope_before.split()[-7:])
    if any(neg.strip() and neg in (" " + pre_window + " ") for neg in PRE_NEG):
        return "negated"
    if any(neg in scope for neg in POST_NEG):
        return "negated"
    return "affirmed"


def find_term_mentions(sentence: str, terms: list[str]) -> list[tuple[int, int]]:
    s = sentence.lower()
    spans = []
    for term in terms:
        idx = s.find(term)
        while idx != -1:
            spans.append((idx, idx + len(term)))
            idx = s.find(term, idx + 1)
    return spans


# ============================================================
# SECTION 1: per-field statistics
# ============================================================
def field_statistics(dfs: dict[str, pd.DataFrame]) -> tuple[list[dict], dict]:
    rows = []
    raw = {}
    for split, df in dfs.items():
        n = len(df)
        for f in FIELDS:
            texts = df[f].tolist()
            missing = sum(is_missing(t) for t in texts)
            nonmissing = [t for t in texts if not is_missing(t)]
            words = np.array([word_count(t) for t in nonmissing], dtype=float)
            sents = np.array([len(sentences(t)) for t in nonmissing], dtype=float)
            chars = np.array([len(t) for t in nonmissing], dtype=float)
            # exact-unique ratio over non-missing (normalized whitespace)
            norm = [re.sub(r"\s+", " ", t.strip().lower()) for t in nonmissing]
            uniq_ratio = len(set(norm)) / max(len(norm), 1) * 100.0

            def pct(a):
                if a.size == 0:
                    return {
                        k: 0.0
                        for k in [
                            "mean",
                            "median",
                            "std",
                            "p05",
                            "p25",
                            "p75",
                            "p95",
                            "min",
                            "max",
                        ]
                    }
                return {
                    "mean": float(np.mean(a)),
                    "median": float(np.median(a)),
                    "std": float(np.std(a)),
                    "p05": float(np.percentile(a, 5)),
                    "p25": float(np.percentile(a, 25)),
                    "p75": float(np.percentile(a, 75)),
                    "p95": float(np.percentile(a, 95)),
                    "min": float(a.min()),
                    "max": float(a.max()),
                }

            wstat, sstat, cstat = pct(words), pct(sents), pct(chars)
            rows.append(
                {
                    "split": split,
                    "field": f,
                    "n_scans": n,
                    "missing_rate_pct": round(missing / n * 100, 4),
                    "missing_count": missing,
                    "word_mean": round(wstat["mean"], 2),
                    "word_median": wstat["median"],
                    "word_p25": wstat["p25"],
                    "word_p75": wstat["p75"],
                    "word_p05": wstat["p05"],
                    "word_p95": wstat["p95"],
                    "word_max": wstat["max"],
                    "sent_median": sstat["median"],
                    "sent_p25": sstat["p25"],
                    "sent_p75": sstat["p75"],
                    "char_median": cstat["median"],
                    "char_p25": cstat["p25"],
                    "char_p75": cstat["p75"],
                    "exact_unique_ratio_pct": round(uniq_ratio, 4),
                }
            )
            raw[f"{split}__{f}"] = {
                "word": wstat,
                "sent": sstat,
                "char": cstat,
                "missing_rate_pct": missing / n * 100,
                "exact_unique_ratio_pct": uniq_ratio,
                "n": n,
            }
    # Impression/Findings word ratio (per scan, non-missing both)
    for split, df in dfs.items():
        ratios = []
        for _, r in df.iterrows():
            wf, wi = word_count(r["Findings_EN"]), word_count(r["Impressions_EN"])
            if wf > 0:
                ratios.append(wi / wf)
        ratios = np.array(ratios)
        raw[f"{split}__imp_find_word_ratio_median"] = (
            float(np.median(ratios)) if ratios.size else 0.0
        )
    return rows, raw


# ============================================================
# SECTION 2: tokenizer truncation (CT-CLIP text encoder)
# ============================================================
def tokenizer_truncation(dfs: dict[str, pd.DataFrame]) -> dict:
    from transformers import AutoTokenizer

    tok_name = "microsoft/BiomedVLP-CXR-BERT-specialized"
    used = tok_name
    try:
        tok = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        used = "bert-base-uncased (FALLBACK, no net)"
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    maxlens = [77, 128, 256, 512]
    out = {"tokenizer": used, "splits": {}}
    for split, df in dfs.items():
        find = df["Findings_EN"].tolist()
        imp = df["Impressions_EN"].tolist()
        concat = [(f + " " + i).strip() for f, i in zip(find, imp)]
        res = {}
        for name, texts in [
            ("findings", find),
            ("impression", imp),
            ("findings+impression", concat),
        ]:
            enc = tok(texts, add_special_tokens=True)
            lens = np.array([len(x) for x in enc["input_ids"]], dtype=float)
            res[name] = {
                "token_mean": float(lens.mean()),
                "token_median": float(np.median(lens)),
                "token_p95": float(np.percentile(lens, 95)),
                "token_max": float(lens.max()),
                "trunc_rate_pct": {
                    str(m): round(float((lens > m).mean() * 100), 3) for m in maxlens
                },
            }
        out["splits"][split] = res
    return out


# ============================================================
# SECTION 3: boilerplate
# ============================================================
def boilerplate_analysis(dfs: dict[str, pd.DataFrame]) -> tuple[list[dict], dict]:
    rows = []
    raw = {}
    for split, df in dfs.items():
        for field in ["Findings_EN", "Impressions_EN"]:
            all_sents = []
            per_report_sents = []
            for t in df[field].tolist():
                if is_missing(t):
                    per_report_sents.append([])
                    continue
                ss = sentences(t)
                per_report_sents.append(ss)
                all_sents.extend(ss)
            exact_counter = Counter(all_sents)
            norm_counter = Counter(norm_sentence(s) for s in all_sents)
            n_reports = sum(1 for x in per_report_sents if x)
            total_tokens = sum(word_count(s) for s in all_sents)

            # boilerplate = normalized sentences appearing in > 1% of reports
            thresh = max(2, int(0.01 * n_reports))
            boiler_norms = {k for k, v in norm_counter.items() if v >= thresh}
            boiler_tokens = sum(
                word_count(s) for s in all_sents if norm_sentence(s) in boiler_norms
            )
            boiler_frac = boiler_tokens / max(total_tokens, 1)

            top_exact = exact_counter.most_common(20)
            for rank, (sent, cnt) in enumerate(top_exact, 1):
                rows.append(
                    {
                        "split": split,
                        "field": field,
                        "rank": rank,
                        "count": cnt,
                        "pct_of_reports": round(cnt / max(n_reports, 1) * 100, 2),
                        "sentence": sent[:200],
                    }
                )
            raw[f"{split}__{field}"] = {
                "n_reports": n_reports,
                "total_sentences": len(all_sents),
                "unique_exact_sentences": len(exact_counter),
                "unique_norm_sentences": len(norm_counter),
                "boilerplate_token_fraction": round(boiler_frac, 4),
                "n_boilerplate_sentence_types": len(boiler_norms),
                "top5": [(s[:120], c) for s, c in top_exact[:5]],
            }
        # near-duplicate WHOLE-report rate (normalized findings)
        norm_reports = [
            norm_sentence(t) for t in df["Findings_EN"].tolist() if not is_missing(t)
        ]
        c = Counter(norm_reports)
        dup = sum(v for v in c.values() if v > 1) - sum(1 for v in c.values() if v > 1)
        raw[f"{split}__near_dup_whole_report_rate_pct"] = round(
            dup / max(len(norm_reports), 1) * 100, 3
        )
    return rows, raw


# ============================================================
# SECTION 4: language all-zero vs positive + assertion
# ============================================================
def language_patterns(dfs: dict[str, pd.DataFrame]) -> dict:
    out = {}
    for split, df in dfs.items():
        groups = {
            "all_zero": df[df["n_labels"] == 0],
            "positive": df[df["n_labels"] > 0],
        }
        res = {}
        for gname, gdf in groups.items():
            n = len(gdf)
            neg = norm = unc = comp = rec = 0
            for t in gdf["Findings_EN"].tolist():
                tl = " " + t.lower() + " "
                if any(k in tl for k in (POST_NEG + [p for p in PRE_NEG])):
                    neg += 1
                if any(k in tl for k in NORMAL_WORDS):
                    norm += 1
                if any(k in tl for k in UNCERTAIN):
                    unc += 1
                if any(k in tl for k in COMPARISON_WORDS):
                    comp += 1
                if any(k in tl for k in RECOMMENDATION_WORDS):
                    rec += 1
            res[gname] = {
                "n": n,
                "negation_rate_pct": round(neg / max(n, 1) * 100, 2),
                "normal_wording_rate_pct": round(norm / max(n, 1) * 100, 2),
                "uncertainty_rate_pct": round(unc / max(n, 1) * 100, 2),
                "comparison_rate_pct": round(comp / max(n, 1) * 100, 2),
                "recommendation_rate_pct": round(rec / max(n, 1) * 100, 2),
            }
        out[split] = res
    return out


# ============================================================
# SECTION 5: disease-term <-> label agreement
# ============================================================
def term_label_agreement(
    dfs: dict[str, pd.DataFrame],
) -> tuple[list[dict], list[dict], dict]:
    rows = []
    disagreements = []  # collected on TRAIN
    assertion_totals = Counter()
    df_tr = dfs["train"]
    df_va = dfs["valid"]

    for split, df in [("train", df_tr), ("valid", df_va)]:
        for label in LABELS_18:
            terms = LABEL_SYNONYMS[label]
            tp = fp = fn = tn = 0
            raw_mention = 0  # any surface mention regardless of assertion
            raw_in_neg_reports = 0
            neg_examples, pos_examples = [], []
            for _, r in df.iterrows():
                text = r["Findings_EN"] + " " + r["Impressions_EN"]
                y = int(r[label])
                affirmed = False
                any_mention = False
                for sent in sentences(text):
                    spans = find_term_mentions(sent, terms)
                    for sp in spans:
                        any_mention = True
                        st = assertion_status(sent, sp)
                        assertion_totals[st] += 1
                        if st == "affirmed":
                            affirmed = True
                pred = affirmed
                if any_mention:
                    raw_mention += 1
                    if y == 0:
                        raw_in_neg_reports += 1
                if pred and y == 1:
                    tp += 1
                elif pred and y == 0:
                    fp += 1
                    if len(neg_examples) < 12:
                        neg_examples.append((r["scan_key"], text))
                elif not pred and y == 1:
                    fn += 1
                    if len(pos_examples) < 12:
                        pos_examples.append((r["scan_key"], text))
                else:
                    tn += 1
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            npos = int(df[label].sum())
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "n_positive_labels": npos,
                    "affirmed_pred_precision": round(prec, 4),
                    "affirmed_pred_recall": round(rec, 4),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "raw_mention_count": raw_mention,
                    "raw_mention_in_label_negative_pct": round(
                        raw_in_neg_reports / max(raw_mention, 1) * 100, 2
                    ),
                }
            )
            if split == "train":
                # disagreement cases (>=10 where possible): FP then FN
                cases = [
                    ("FP: affirmed-mention but label=0", e) for e in neg_examples
                ] + [("FN: label=1 but no-affirmed-mention", e) for e in pos_examples]
                for kind, (sk, txt) in cases[:12]:
                    disagreements.append(
                        {
                            "label": label,
                            "kind": kind,
                            "scan_key": sk,
                            "excerpt": txt[:600],
                        }
                    )
    return rows, disagreements, dict(assertion_totals)


# ============================================================
# SECTION 6: label burden vs length
# ============================================================
def burden_vs_length(dfs: dict[str, pd.DataFrame]) -> dict:
    out = {}
    bands = [("all_zero", 0, 0), ("1-3", 1, 3), ("4-7", 4, 7), ("8+", 8, 18)]
    for split, df in dfs.items():
        fw = df["Findings_EN"].apply(word_count).values
        iw = df["Impressions_EN"].apply(word_count).values
        nl = df["n_labels"].values
        rho_f, p_f = spearmanr(nl, fw)
        rho_i, p_i = spearmanr(nl, iw)
        band_stats = {}
        for bname, lo, hi in bands:
            m = (nl >= lo) & (nl <= hi)
            band_stats[bname] = {
                "n": int(m.sum()),
                "findings_word_median": float(np.median(fw[m])) if m.sum() else 0,
                "impression_word_median": float(np.median(iw[m])) if m.sum() else 0,
            }
        out[split] = {
            "spearman_nlabels_vs_findings_words": {
                "rho": round(float(rho_f), 4),
                "p": float(p_f),
            },
            "spearman_nlabels_vs_impression_words": {
                "rho": round(float(rho_i), 4),
                "p": float(p_i),
            },
            "bands": band_stats,
        }
    return out


# ============================================================
# SAMPLES
# ============================================================
def write_samples(dfs: dict[str, pd.DataFrame]):
    rng = random.Random(42)
    df = dfs["train"]

    def fmt(r):
        return (
            f"### {r['scan_key']}  (n_labels={r['n_labels']}; "
            f"+labels: {', '.join([l for l in LABELS_18 if r[l] == 1]) or 'NONE'})\n\n"
            f"**Clinical:** {r['ClinicalInformation_EN']}\n\n"
            f"**Technique:** {r['Technique_EN']}\n\n"
            f"**Findings:** {r['Findings_EN']}\n\n"
            f"**Impression:** {r['Impressions_EN']}\n\n---\n"
        )

    normal = df[df["n_labels"] == 0]
    abnormal = df[df["n_labels"] >= 4]
    n_sel = normal.sample(min(15, len(normal)), random_state=1)
    a_sel = abnormal.sample(min(15, len(abnormal)), random_state=2)
    r_sel = df.sample(min(15, len(df)), random_state=3)

    (OUT / "samples" / "reports_normal.md").write_text(
        "# Normal (all-zero label) reports — TRAIN, clean, scan-unit\n\n"
        + "".join(fmt(r) for _, r in n_sel.iterrows())
    )
    (OUT / "samples" / "reports_abnormal.md").write_text(
        "# Abnormal (>=4 labels) reports — TRAIN, clean, scan-unit\n\n"
        + "".join(fmt(r) for _, r in a_sel.iterrows())
    )
    (OUT / "samples" / "reports_random.md").write_text(
        "# Random reports — TRAIN, clean, scan-unit\n\n"
        + "".join(fmt(r) for _, r in r_sel.iterrows())
    )


def write_disagreements(disagreements: list[dict]):
    by_label = {}
    for d in disagreements:
        by_label.setdefault(d["label"], []).append(d)
    lines = [
        "# Term<->Label disagreement cases (TRAIN)\n",
        "NegEx-style sentence assertion vs silver label. FP = affirmed surface "
        "mention but label=0; FN = label=1 but no affirmed mention.\n",
    ]
    for label in LABELS_18:
        cases = by_label.get(label, [])
        lines.append(f"\n## {label}  ({len(cases)} cases)\n")
        for c in cases:
            lines.append(f"- **{c['kind']}** [{c['scan_key']}]: {c['excerpt']}\n")
    (OUT / "samples" / "label_disagreements.md").write_text("\n".join(lines))


# ============================================================
# FIGURES
# ============================================================
def fig_report_length(dfs: dict[str, pd.DataFrame], burden: dict):
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    tr, va = dfs["train"], dfs["valid"]

    fw_tr = tr["Findings_EN"].apply(word_count).values
    fw_va = va["Findings_EN"].apply(word_count).values
    ax[0, 0].hist(
        fw_tr,
        bins=60,
        range=(0, 400),
        density=True,
        alpha=0.6,
        color=TRAIN_BLUE,
        label=f"train (n={len(fw_tr)})",
    )
    ax[0, 0].hist(
        fw_va,
        bins=60,
        range=(0, 400),
        density=True,
        alpha=0.6,
        color=VALID_ORANGE,
        label=f"valid (n={len(fw_va)})",
    )
    ax[0, 0].axvline(np.median(fw_tr), color=TRAIN_BLUE, ls="--")
    ax[0, 0].set_title("Findings word count (scan-unit, clean)")
    ax[0, 0].set_xlabel("words")
    ax[0, 0].set_ylabel("density")
    ax[0, 0].legend()

    iw_tr = tr["Impressions_EN"].apply(word_count).values
    iw_va = va["Impressions_EN"].apply(word_count).values
    ax[0, 1].hist(
        iw_tr,
        bins=60,
        range=(0, 150),
        density=True,
        alpha=0.6,
        color=TRAIN_BLUE,
        label="train",
    )
    ax[0, 1].hist(
        iw_va,
        bins=60,
        range=(0, 150),
        density=True,
        alpha=0.6,
        color=VALID_ORANGE,
        label="valid",
    )
    ax[0, 1].axvline(np.median(iw_tr), color=TRAIN_BLUE, ls="--")
    ax[0, 1].set_title("Impression word count (scan-unit, clean)")
    ax[0, 1].set_xlabel("words")
    ax[0, 1].set_ylabel("density")
    ax[0, 1].legend()

    # per-field word count boxplot (train)
    data = [tr[f].apply(word_count).values for f in FIELDS]
    ax[1, 0].boxplot(
        data,
        labels=["Clinical", "Technique", "Findings", "Impression"],
        showfliers=False,
    )
    ax[1, 0].set_title("Word count per field (TRAIN, scan-unit)")
    ax[1, 0].set_ylabel("words")

    # findings word median by burden band
    bands = ["all_zero", "1-3", "4-7", "8+"]
    fmed = [burden["train"]["bands"][b]["findings_word_median"] for b in bands]
    imed = [burden["train"]["bands"][b]["impression_word_median"] for b in bands]
    x = np.arange(len(bands))
    ax[1, 1].bar(x - 0.2, fmed, 0.4, color=TRAIN_BLUE, label="Findings")
    ax[1, 1].bar(x + 0.2, imed, 0.4, color=VALID_ORANGE, label="Impression")
    ax[1, 1].set_xticks(x)
    ax[1, 1].set_xticklabels(bands)
    ax[1, 1].set_title("Median length by label-burden band (TRAIN, scan-unit)")
    ax[1, 1].set_xlabel("# positive labels")
    ax[1, 1].set_ylabel("median words")
    ax[1, 1].legend()

    fig.suptitle("CT-RATE Report Length — clean population, unit = SCAN", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "figures" / "report_length.png", dpi=220)
    plt.close(fig)


def fig_language_patterns(
    lang: dict, agree_rows: list[dict], boiler_rows: list[dict], assertion_totals: dict
):
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))

    # (a) language rates all-zero vs positive (train)
    cats = ["negation", "normal_wording", "uncertainty", "comparison", "recommendation"]
    keys = [c + "_rate_pct" for c in cats]
    az = [lang["train"]["all_zero"][k] for k in keys]
    pos = [lang["train"]["positive"][k] for k in keys]
    x = np.arange(len(cats))
    ax[0, 0].bar(x - 0.2, az, 0.4, color="#2ca02c", label="all-zero")
    ax[0, 0].bar(x + 0.2, pos, 0.4, color="#d62728", label="positive")
    ax[0, 0].set_xticks(x)
    ax[0, 0].set_xticklabels(cats, rotation=20, ha="right")
    ax[0, 0].set_title(
        "Findings language rates: all-zero vs positive (TRAIN, scan-unit)"
    )
    ax[0, 0].set_ylabel("% of reports")
    ax[0, 0].legend()

    # (b) assertion status distribution
    labs = list(assertion_totals.keys())
    vals = [assertion_totals[k] for k in labs]
    ax[0, 1].bar(labs, vals, color=["#1f77b4", "#d62728", "#ff7f0e"])
    ax[0, 1].set_title(
        "NegEx assertion status over all disease-term mentions (train+valid)"
    )
    ax[0, 1].set_ylabel("mention count")
    for i, v in enumerate(vals):
        ax[0, 1].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

    # (c) top boilerplate Findings sentences (train)
    tr_find = [
        r for r in boiler_rows if r["split"] == "train" and r["field"] == "Findings_EN"
    ][:12]
    names = [r["sentence"][:45] + "..." for r in tr_find][::-1]
    counts = [r["count"] for r in tr_find][::-1]
    ax[1, 0].barh(range(len(names)), counts, color=TRAIN_BLUE)
    ax[1, 0].set_yticks(range(len(names)))
    ax[1, 0].set_yticklabels(names, fontsize=7)
    ax[1, 0].set_title("Top exact Findings sentences (TRAIN, scan-unit)")
    ax[1, 0].set_xlabel("count")

    # (d) term->label precision/recall (train)
    tr_agree = [r for r in agree_rows if r["split"] == "train"]
    labels = [r["label"][:22] for r in tr_agree]
    prec = [r["affirmed_pred_precision"] for r in tr_agree]
    rec = [r["affirmed_pred_recall"] for r in tr_agree]
    y = np.arange(len(labels))
    ax[1, 1].barh(y - 0.2, prec, 0.4, color="#1f77b4", label="precision")
    ax[1, 1].barh(y + 0.2, rec, 0.4, color="#ff7f0e", label="recall")
    ax[1, 1].set_yticks(y)
    ax[1, 1].set_yticklabels(labels, fontsize=7)
    ax[1, 1].set_title("Affirmed-mention vs silver label (TRAIN)")
    ax[1, 1].set_xlabel("score")
    ax[1, 1].legend()
    ax[1, 1].invert_yaxis()

    fig.suptitle(
        "CT-RATE Report Language & Term-Label Agreement — clean, scan-unit", fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "figures" / "report_language_patterns.png", dpi=220)
    plt.close(fig)


# ============================================================
def main():
    print("Loading clean scan-deduped reports...")
    dfs = {
        "train": load_clean_scan_reports("train"),
        "valid": load_clean_scan_reports("valid"),
    }
    for s, d in dfs.items():
        print(f"  {s}: {len(d)} clean scans")

    print("Section 1: per-field stats")
    stat_rows, stat_raw = field_statistics(dfs)
    pd.DataFrame(stat_rows).to_csv(
        OUT / "tables" / "report_statistics.csv", index=False
    )

    print("Section 2: tokenizer truncation")
    tok = tokenizer_truncation(dfs)

    print("Section 3: boilerplate")
    boiler_rows, boiler_raw = boilerplate_analysis(dfs)
    pd.DataFrame(boiler_rows).to_csv(
        OUT / "tables" / "report_boilerplate.csv", index=False
    )

    print("Section 4: language patterns")
    lang = language_patterns(dfs)

    print("Section 5: term<->label agreement")
    agree_rows, disagreements, assertion_totals = term_label_agreement(dfs)
    pd.DataFrame(agree_rows).to_csv(
        OUT / "tables" / "term_label_agreement.csv", index=False
    )

    print("Section 6: burden vs length")
    burden = burden_vs_length(dfs)

    print("Writing samples + disagreements")
    write_samples(dfs)
    write_disagreements(disagreements)

    print("Figures")
    fig_report_length(dfs, burden)
    fig_language_patterns(lang, agree_rows, boiler_rows, assertion_totals)

    raw = {
        "population": "clean (no_chest + unencodable excluded); unit=SCAN (recon-deduped)",
        "scan_counts": {s: len(d) for s, d in dfs.items()},
        "field_statistics": stat_raw,
        "tokenizer_truncation": tok,
        "boilerplate": boiler_raw,
        "language_patterns": lang,
        "assertion_totals": assertion_totals,
        "term_label_agreement": agree_rows,
        "burden_vs_length": burden,
    }
    with open(OUT / "tables" / "report_raw.json", "w") as f:
        json.dump(raw, f, indent=2, default=float)
    print("DONE. raw json ->", OUT / "tables" / "report_raw.json")

    # quick console digest
    for split in ["train", "valid"]:
        f = stat_raw[f"{split}__Findings_EN"]
        i = stat_raw[f"{split}__Impressions_EN"]
        c = stat_raw[f"{split}__ClinicalInformation_EN"]
        t = stat_raw[f"{split}__Technique_EN"]
        print(
            f"[{split}] Clinical missing {c['missing_rate_pct']:.1f}% | "
            f"Technique word median {t['word']['median']:.0f} | "
            f"Findings word median {f['word']['median']:.0f} "
            f"(IQR {f['word']['p25']:.0f}-{f['word']['p75']:.0f}) sent median {f['sent']['median']:.0f} "
            f"exact-uniq {f['exact_unique_ratio_pct']:.2f}% | "
            f"Impression word median {i['word']['median']:.0f} missing {i['missing_rate_pct']:.2f}% "
            f"exact-uniq {i['exact_unique_ratio_pct']:.2f}% | "
            f"imp/find ratio {raw['field_statistics'][f'{split}__imp_find_word_ratio_median']:.3f}"
        )


if __name__ == "__main__":
    main()
