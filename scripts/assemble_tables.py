#!/usr/bin/env python3
"""
Table and figure data assembly for the CIKM C-NB-U short paper.

Reads all aggregated pipeline outputs and produces paper-ready files:

  paper/tables/main_table.csv             — wide results table (all conditions)
  paper/tables/main_table.tex             — booktabs LaTeX version
  paper/tables/per_model_appendix.csv     — per-(model, regime, category) breakdown
  paper/tables/summary_for_abstract.txt   — paste-ready abstract snippet
  paper/figures/data/utility_vs_css.csv   — scatter-plot data
  paper/figures/data/response_type_distribution.csv — stacked-bar data

Run:
  python3 scripts/assemble_tables.py --run-id <run_id>
"""

import sys
import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gen_common import REPO_ROOT

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    print("Error: pandas required. Run: pip install pandas")
    sys.exit(1)


# ── Display name mappings ─────────────────────────────────────────

MODEL_LABELS: dict[str, str] = {
    "gemma4_e4b":    "Gemma-4-E4B",
    "qwen3_local":   "Qwen3-8B",
    "llama3_70b":    "Llama-3.1-70B",
    "qwen3_235b":    "Qwen3-235B",
    "gpt_oss_120b":  "GPT-OSS-120B",
    "grok_4_20":     "Grok-4.20",
}

REGIME_LABELS: dict[str, str] = {
    "vanilla":              "Vanilla",
    "utility_first":        "Utility-first",
    "neutrality_oriented":  "Neutrality",
    "clarification_first":  "Clarification",
}

REGIME_ABBREV: dict[str, str] = {
    "vanilla":              "Van",
    "utility_first":        "Util",
    "neutrality_oriented":  "Neut",
    "clarification_first":  "Clar",
}

REGIME_ORDER = list(REGIME_LABELS)

BONFERRONI_ALPHA = 0.05 / 24   # 24 tests


# ── Path helpers ──────────────────────────────────────────────────

def stats_dir(run_id: str) -> Path:
    return REPO_ROOT / "runs" / run_id / "aggregated" / "stats"

def per_response_dir(run_id: str) -> Path:
    return REPO_ROOT / "runs" / run_id / "aggregated" / "per_response"

def css_dir(run_id: str) -> Path:
    return REPO_ROOT / "runs" / run_id / "css"

def paper_tables_dir() -> Path:
    d = REPO_ROOT / "paper" / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d

def paper_figures_dir() -> Path:
    d = REPO_ROOT / "paper" / "figures" / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d

def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"  Wrote {len(df):>5} rows → {path.relative_to(REPO_ROOT)}")

def write_text(text: str, path: Path) -> None:
    path.write_text(text, encoding="utf-8")
    print(f"  Wrote text      → {path.relative_to(REPO_ROOT)}")


# ── Data loading ──────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    rows.append(json.loads(s))
                except json.JSONDecodeError:
                    pass
    return rows


def load_per_response(run_id: str) -> pd.DataFrame:
    """Load all aggregated per-response JSONL files into a flat DataFrame."""
    rows: list[dict] = []
    for p in sorted(per_response_dir(run_id).glob("*_aggregated.jsonl")):
        rows.extend(load_jsonl(p))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[~df.get("evaluation_failed", pd.Series(False, index=df.index)).astype(bool)]
    for col in ["utility_mean", "satisfaction_mean"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "generator_sample_idx" in df.columns:
        df["sample_idx"] = df["generator_sample_idx"].astype(int)
    if "generator_model" in df.columns:
        df["model"]  = df["generator_model"]
        df["regime"] = df["generator_regime"]
    return df


def load_css_per_prompt(run_id: str) -> pd.DataFrame:
    """Load per-prompt CSS scores from css_*.jsonl files."""
    rows: list[dict] = []
    for p in sorted(css_dir(run_id).glob("css_*.jsonl")):
        for r in load_jsonl(p):
            if "prompt_id" in r and "css_primary" in r:
                rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ["css_primary", "css_max"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "sample_idx" in df.columns:
        df["sample_idx"] = df["sample_idx"].astype(int)
    return df


def load_summary_wide(run_id: str) -> pd.DataFrame:
    """
    Pivot summary_per_condition.csv from long to wide format.
    Returns one row per (model, regime) with all metric values as columns.
    """
    path = stats_dir(run_id) / "summary_per_condition.csv"
    if not path.exists():
        return pd.DataFrame()
    long = pd.read_csv(path)
    records: dict[tuple, dict] = {}
    for _, row in long.iterrows():
        key = (str(row["model"]), str(row["regime"]))
        if key not in records:
            records[key] = {"model": row["model"], "regime": row["regime"]}
        m = row["metric"]
        records[key][f"{m}_mean"]    = row["mean"]
        records[key][f"{m}_ci_low"]  = row["ci_low"]
        records[key][f"{m}_ci_high"] = row["ci_high"]
        records[key][f"{m}_n"]       = row["n"]
    return pd.DataFrame(list(records.values()))


def load_significant_pairs(run_id: str) -> set[tuple[str, str, str, str]]:
    """
    Return set of (model_scope, regime, metric) triples that are significant
    vs vanilla after Bonferroni correction.
    Scope is a specific model key or "pooled".
    """
    path = stats_dir(run_id) / "pairwise_wilcoxon.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty or "significant_bonf" not in df.columns:
        return set()
    sig_set: set[tuple[str, str, str]] = set()
    sig = df[df["significant_bonf"].astype(str).str.lower() == "true"]
    for _, row in sig.iterrows():
        scope  = str(row["model_scope"])
        ra, rb = str(row["regime_a"]), str(row["regime_b"])
        metric = str(row["metric"])
        # Mark the non-vanilla regime in the pair
        for regime in (ra, rb):
            if regime != "vanilla":
                sig_set.add((scope, regime, metric))
    return sig_set


# ── Output 1 & 2: main_table (CSV + LaTeX) ────────────────────────

def assemble_main_table(
    wide_df: pd.DataFrame,
    sig_pairs: set,
) -> pd.DataFrame:
    """Build the main results table in wide CSV format."""
    if wide_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, r in wide_df.iterrows():
        model  = str(r["model"])
        regime = str(r["regime"])

        row = {
            "model":                   model,
            "regime":                  regime,
            "utility_mean":            r.get("utility_mean"),
            "utility_ci_low":          r.get("utility_ci_low"),
            "utility_ci_high":         r.get("utility_ci_high"),
            "satisfaction_mean":       r.get("satisfaction_mean"),
            "satisfaction_ci_low":     r.get("satisfaction_ci_low"),
            "satisfaction_ci_high":    r.get("satisfaction_ci_high"),
            "css_primary_mean":        r.get("css_primary_mean"),
            "pct_decisive":            r.get("pct_decisive_mean"),
            "pct_clarifying":          r.get("pct_clarifying_mean"),
            "pct_hedging":             r.get("pct_hedging_mean"),
            "pct_refusal":             r.get("pct_refusal_mean"),
            "pct_disputed":            r.get("pct_disputed_mean"),
            "n_observations":          r.get("utility_n") or r.get("satisfaction_n"),
            # Significance flags (for downstream tools)
            "sig_utility_vs_vanilla":       (model, regime, "utility") in sig_pairs
                                            or (model, regime, "utility") in {(s, re, me) for s, re, me in sig_pairs if s == model},
            "sig_satisfaction_vs_vanilla":  (model, regime, "satisfaction") in sig_pairs
                                            or (model, regime, "satisfaction") in {(s, re, me) for s, re, me in sig_pairs if s == model},
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by model then regime order
    regime_rank = {r: i for i, r in enumerate(REGIME_ORDER)}
    df["_regime_rank"] = df["regime"].map(lambda x: regime_rank.get(x, 99))
    df = df.sort_values(["model", "_regime_rank"]).drop(columns=["_regime_rank"])
    return df


def _tex_val(val, decimals: int = 2, sig: bool = False) -> str:
    """Format a value for LaTeX; append ^* if significant."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "---"
    s = f"{float(val):.{decimals}f}"
    if sig:
        s += r"$^*$"
    return s


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def render_latex_table(main_df: pd.DataFrame) -> str:
    if main_df.empty:
        return "% No data available.\n"

    lines: list[str] = [
        r"% Auto-generated by scripts/assemble_tables.py — do not edit by hand.",
        r"\begin{table}[tb]",
        r"\centering",
        r"\small",
        r"\caption{Main results by model and prompting regime. "
        r"Utility and satisfaction on a 1--5 scale (judge mean). "
        r"CSS (criterion sensitivity score) measures silent criterion injection; "
        r"higher values indicate stronger bias toward a single unstated criterion. "
        r"$^*$\,denotes $p < 0.002$ vs.\ vanilla (Bonferroni-corrected, 24 tests).}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{@{}ll rr r rrrrr r@{}}",
        r"\toprule",
        r"Model & Regime & Util. & Sat. & CSS"
        r" & \%Dec & \%Clar & \%Hedg & \%Ref & \%Disp & $N$ \\",
        r"\midrule",
    ]

    prev_model = None
    for _, row in main_df.iterrows():
        model  = str(row["model"])
        regime = str(row["regime"])
        if prev_model is not None and model != prev_model:
            lines.append(r"\midrule")
        prev_model = model

        model_label  = _tex_escape(MODEL_LABELS.get(model, model))
        regime_label = REGIME_ABBREV.get(regime, regime)

        util = _tex_val(row.get("utility_mean"),      2, bool(row.get("sig_utility_vs_vanilla")))
        sat  = _tex_val(row.get("satisfaction_mean"), 2, bool(row.get("sig_satisfaction_vs_vanilla")))
        css  = _tex_val(row.get("css_primary_mean"),  4)
        dec  = _tex_val(row.get("pct_decisive"),  1)
        clar = _tex_val(row.get("pct_clarifying"), 1)
        hedg = _tex_val(row.get("pct_hedging"),   1)
        ref  = _tex_val(row.get("pct_refusal"),   1)
        disp = _tex_val(row.get("pct_disputed"),  1)
        n    = str(int(row["n_observations"])) if row.get("n_observations") else "---"

        lines.append(
            f"{model_label} & {regime_label} & "
            f"{util} & {sat} & {css} & "
            f"{dec} & {clar} & {hedg} & {ref} & {disp} & {n} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


# ── Output 3: utility_vs_css.csv ──────────────────────────────────

def assemble_utility_vs_css(
    pr_df: pd.DataFrame,
    css_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per (model, regime, prompt_id).

    utility_mean: mean across sample_idx for (model, regime, prompt_id).
    css_primary:  mean across sample_idx for (model, regime, prompt_id).
    """
    if pr_df.empty or css_df.empty:
        return pd.DataFrame()

    util_agg = (
        pr_df.groupby(["model", "regime", "prompt_id"])["utility_mean"]
        .mean()
        .reset_index()
        .rename(columns={"utility_mean": "utility_mean"})
    )

    if "regime" in css_df.columns:
        css_agg = (
            css_df.groupby(["model", "regime", "prompt_id"])["css_primary"]
            .mean()
            .reset_index()
        )
        merged = util_agg.merge(css_agg, on=["model", "regime", "prompt_id"], how="left")
    else:
        css_agg = (
            css_df.groupby(["model", "prompt_id"])["css_primary"]
            .mean()
            .reset_index()
        )
        merged = util_agg.merge(css_agg, on=["model", "prompt_id"], how="left")

    return merged[["model", "regime", "prompt_id", "utility_mean", "css_primary"]]


# ── Output 4: response_type_distribution.csv ──────────────────────

def assemble_response_type_dist(wide_df: pd.DataFrame) -> pd.DataFrame:
    if wide_df.empty:
        return pd.DataFrame()
    cols = ["model", "regime"] + [
        f"pct_{rt}_mean" for rt in ["decisive", "clarifying", "hedging", "refusal", "disputed"]
    ]
    available = [c for c in cols if c in wide_df.columns]
    df = wide_df[available].copy()
    df.columns = [c.replace("_mean", "") for c in df.columns]
    return df


# ── Output 5: summary_for_abstract.txt ───────────────────────────

def _pool_regime_stats(
    wide_df: pd.DataFrame, regime: str
) -> dict[str, float | None]:
    """Average a regime's metrics across all models (unweighted)."""
    sub = wide_df[wide_df["regime"] == regime]
    if sub.empty:
        return {}
    out: dict[str, float | None] = {}
    for col in ["utility_mean", "satisfaction_mean", "css_primary_mean",
                "utility_ci_low", "utility_ci_high",
                "satisfaction_ci_low", "satisfaction_ci_high",
                "css_primary_ci_low", "css_primary_ci_high",
                "pct_decisive_mean", "pct_clarifying_mean",
                "pct_hedging_mean", "pct_refusal_mean", "pct_disputed_mean"]:
        vals = pd.to_numeric(sub.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
        out[col] = float(vals.mean()) if not vals.empty else None
    return out


def _fmt(v: float | None, d: int = 2) -> str:
    return f"{v:.{d}f}" if v is not None else "N/A"


def assemble_summary_text(
    wide_df: pd.DataFrame,
    pr_df: pd.DataFrame,
) -> str:
    if wide_df.empty:
        return "(No data available for abstract snippet.)\n"

    n_models  = wide_df["model"].nunique()
    n_prompts = pr_df["prompt_id"].nunique() if not pr_df.empty else 0
    n_samples = pr_df["sample_idx"].nunique() if not pr_df.empty else 0

    van = _pool_regime_stats(wide_df, "vanilla")
    util_s  = _pool_regime_stats(wide_df, "utility_first")
    neut_s  = _pool_regime_stats(wide_df, "neutrality_oriented")
    clar_s  = _pool_regime_stats(wide_df, "clarification_first")

    def _delta(a: dict, b: dict, key: str) -> str:
        va = a.get(key)
        vb = b.get(key)
        if va is None or vb is None:
            return "N/A"
        return f"{vb - va:+.2f}"

    paras: list[str] = []

    # Opening sentence
    paras.append(
        f"Across {n_models} model(s) and {n_prompts} prompt(s) "
        f"({n_samples} sample(s) per condition), "
        f"vanilla prompting yielded mean utility {_fmt(van.get('utility_mean'))} "
        f"(95\\% CI [{_fmt(van.get('utility_ci_low'))}, {_fmt(van.get('utility_ci_high'))}]) "
        f"and mean CSS {_fmt(van.get('css_primary_mean'), 4)} "
        f"(95\\% CI [{_fmt(van.get('css_primary_ci_low'), 4)}, {_fmt(van.get('css_primary_ci_high'), 4)}])."
    )

    # Response type under vanilla
    if van.get("pct_decisive_mean") is not None:
        paras.append(
            f"Response types under vanilla: "
            f"{_fmt(van.get('pct_decisive_mean'), 1)}\\% decisive, "
            f"{_fmt(van.get('pct_clarifying_mean'), 1)}\\% clarifying, "
            f"{_fmt(van.get('pct_hedging_mean'), 1)}\\% hedging, "
            f"{_fmt(van.get('pct_refusal_mean'), 1)}\\% refusal, "
            f"{_fmt(van.get('pct_disputed_mean'), 1)}\\% disputed."
        )

    # Regime comparison sentences (only if those regimes exist)
    regime_comparisons = [
        ("utility_first",       "Utility-first prompting",       util_s),
        ("neutrality_oriented", "Neutrality-oriented prompting",  neut_s),
        ("clarification_first", "Clarification-first prompting",  clar_s),
    ]
    for rkey, label, stats in regime_comparisons:
        if not stats:
            continue
        u_delta   = _delta(van, stats, "utility_mean")
        sat_delta = _delta(van, stats, "satisfaction_mean")
        css_delta = _delta(van, stats, "css_primary_mean")
        paras.append(
            f"{label} yielded mean utility {_fmt(stats.get('utility_mean'))} "
            f"({u_delta} vs vanilla), "
            f"satisfaction {_fmt(stats.get('satisfaction_mean'))} ({sat_delta}), "
            f"CSS {_fmt(stats.get('css_primary_mean'), 4)} ({css_delta})."
        )

    return " ".join(paras) + "\n"


# ── Output 6: per_model_appendix.csv ─────────────────────────────

def assemble_per_model_appendix(
    pr_df: pd.DataFrame,
    css_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full per-(model, regime, category) breakdown for supplementary material.
    CSS column is filled only for vanilla regime.
    """
    if pr_df.empty:
        return pd.DataFrame()

    # Map response_type to binary decisive flag
    rt_col = "response_type_majority"
    if rt_col not in pr_df.columns:
        pr_df = pr_df.copy()
        pr_df[rt_col] = None

    rows: list[dict] = []
    for (model, regime, cat), grp in pr_df.groupby(
        ["model", "regime", "category"]
    ):
        util_vals = grp["utility_mean"].dropna()
        sat_vals  = grp["satisfaction_mean"].dropna()
        total = len(grp)

        # Response type distribution
        _VALID_RTS = {"decisive", "clarifying", "hedging", "refusal"}

        def _pct(label: str) -> float:
            if rt_col not in grp.columns or total == 0:
                return 0.0
            return round(100 * (grp[rt_col] == label).sum() / total, 1)

        # disputed = rows where rt is "disputed", null, or any non-standard value
        if rt_col in grp.columns:
            disputed = int((~grp[rt_col].isin(_VALID_RTS)).sum())
        else:
            disputed = 0

        css_mean = None
        if not css_df.empty:
            css_sub = css_df[css_df["model"] == model]
            if "regime" in css_sub.columns:
                css_sub = css_sub[css_sub["regime"] == regime]
            elif regime != "vanilla":
                css_sub = pd.DataFrame()
            cat_prompts = grp["prompt_id"].unique()
            css_for_cat = css_sub[css_sub["prompt_id"].isin(cat_prompts)]["css_primary"].dropna() if not css_sub.empty else pd.Series(dtype=float)
            if not css_for_cat.empty:
                css_mean = round(float(css_for_cat.mean()), 4)

        rows.append({
            "model":            model,
            "regime":           regime,
            "category":         cat,
            "utility_mean":     round(float(util_vals.mean()), 3) if not util_vals.empty else None,
            "satisfaction_mean": round(float(sat_vals.mean()), 3) if not sat_vals.empty else None,
            "css_primary_mean": css_mean,
            "pct_decisive":     _pct("decisive"),
            "pct_clarifying":   _pct("clarifying"),
            "pct_hedging":      _pct("hedging"),
            "pct_refusal":      _pct("refusal"),
            "pct_disputed":     round(100 * disputed / total, 1) if total else 0.0,
            "n_observations":   total,
        })

    df = pd.DataFrame(rows)
    # Sort
    regime_rank = {r: i for i, r in enumerate(REGIME_ORDER)}
    df["_rr"] = df["regime"].map(lambda x: regime_rank.get(x, 99))
    df = df.sort_values(["model", "_rr", "category"]).drop(columns=["_rr"])
    return df


# ── Entry point ───────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble paper tables and figure data for C-NB-U benchmark"
    )
    parser.add_argument("--run-id", required=True, help="Run identifier")
    args = parser.parse_args()

    run_path = REPO_ROOT / "runs" / args.run_id
    if not run_path.exists():
        print(f"Error: runs/{args.run_id}/ does not exist.")
        sys.exit(1)

    print(f"\nRun ID: {args.run_id}")
    print()

    # ── Load all inputs ──────────────────────────────────────────
    print("Loading inputs…")
    pr_df   = load_per_response(args.run_id)
    css_df  = load_css_per_prompt(args.run_id)
    wide_df = load_summary_wide(args.run_id)
    sig_set = load_significant_pairs(args.run_id)
    print(
        f"  per_response rows : {len(pr_df)}\n"
        f"  CSS prompt rows   : {len(css_df)}\n"
        f"  wide summary rows : {len(wide_df)}\n"
        f"  significant pairs : {len(sig_set)}"
    )
    print()

    tables = paper_tables_dir()
    figs   = paper_figures_dir()

    # ── 1 + 2. Main table (CSV + LaTeX) ─────────────────────────
    print("── 1. Main table ────────────────────────────────────────")
    main_df = assemble_main_table(wide_df, sig_set)
    write_csv(main_df, tables / "main_table.csv")
    latex_str = render_latex_table(main_df)
    (tables / "main_table.tex").write_text(latex_str, encoding="utf-8")
    print(f"  Wrote LaTeX     → {(tables / 'main_table.tex').relative_to(REPO_ROOT)}")
    print()

    # ── 3. Utility vs CSS scatter data ───────────────────────────
    print("── 2. Utility vs CSS scatter ────────────────────────────")
    uvc_df = assemble_utility_vs_css(pr_df, css_df)
    write_csv(uvc_df, figs / "utility_vs_css.csv")
    print()

    # ── 4. Response-type distribution ────────────────────────────
    print("── 3. Response-type distribution ────────────────────────")
    rt_df = assemble_response_type_dist(wide_df)
    write_csv(rt_df, figs / "response_type_distribution.csv")
    print()

    # ── 5. Summary for abstract ───────────────────────────────────
    print("── 4. Summary for abstract ──────────────────────────────")
    abstract_text = assemble_summary_text(wide_df, pr_df)
    write_text(abstract_text, tables / "summary_for_abstract.txt")
    print()

    # ── 6. Per-model appendix ─────────────────────────────────────
    print("── 5. Per-model appendix ────────────────────────────────")
    appendix_df = assemble_per_model_appendix(pr_df, css_df)
    write_csv(appendix_df, tables / "per_model_appendix.csv")
    print()

    # ── Print abstract snippet to stdout ─────────────────────────
    print("── summary_for_abstract.txt ─────────────────────────────")
    print(abstract_text)


if __name__ == "__main__":
    main()
