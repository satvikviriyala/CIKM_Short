#!/usr/bin/env python3
"""
Statistical analysis for the CIKM C-NB-U benchmark.

Inputs:
  runs/{run_id}/aggregated/per_response/*.jsonl  — aggregated evaluations
  runs/{run_id}/css/css_*.jsonl                  — CSS scores (vanilla regime)

Outputs (under runs/{run_id}/aggregated/stats/):
  summary_per_condition.csv    — per-(model, regime) means + bootstrap 95% CI
  pairwise_wilcoxon.csv        — Wilcoxon signed-rank + Bonferroni + Cliff's delta
  friedman.csv                 — Friedman test across regimes
  judge_agreement.csv          — Krippendorff alpha (ordinal) + Fleiss kappa (nominal)
  human_validation.csv         — Spearman + annotator agreement (if data present)
  summary.json                 — top-line numbers for the paper
"""

import sys
import json
import csv
import argparse
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gen_common import load_model_config, REPO_ROOT


# ── Dependency checks ─────────────────────────────────────────────

def _require(pkg: str):
    try:
        return __import__(pkg)
    except ImportError:
        print(f"Error: '{pkg}' is not installed. Run: pip install {pkg}")
        sys.exit(1)


_pd_mod = _require("pandas")
_scipy  = _require("scipy")
_kripp  = _require("krippendorff")

import pandas as pd
from scipy import stats as sp_stats

try:
    from statsmodels.stats.inter_rater import fleiss_kappa as _sm_fleiss, aggregate_raters as _sm_agg
    _STATSMODELS = True
except ImportError:
    _STATSMODELS = False
    print("[Warning] statsmodels not found — Fleiss' kappa skipped. pip install statsmodels")


# ── Constants ─────────────────────────────────────────────────────

REGIMES = ["vanilla", "utility_first", "neutrality_oriented", "clarification_first"]
REGIME_PAIRS = list(combinations(REGIMES, 2))          # 6 pairs
NUMERIC_METRICS = ["utility", "satisfaction", "css_primary", "decisive_rate"]
N_BONFERRONI = len(REGIME_PAIRS) * len(NUMERIC_METRICS)  # 24
BONFERRONI_ALPHA = 0.05 / N_BONFERRONI                   # ≈ 0.002083
RT_CATEGORIES = ["decisive", "clarifying", "hedging", "refusal"]


# ── File I/O ──────────────────────────────────────────────────────

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


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Wrote {len(df):>4} rows → {path.relative_to(REPO_ROOT)}")


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  Wrote summary  → {path.relative_to(REPO_ROOT)}")


# ── Data loading ──────────────────────────────────────────────────

def load_eval_records(run_id: str) -> list[dict]:
    d = REPO_ROOT / "runs" / run_id / "aggregated" / "per_response"
    rows: list[dict] = []
    for p in sorted(d.glob("*_aggregated.jsonl")):
        rows.extend(load_jsonl(p))
    print(f"  Eval rows loaded : {len(rows)}")
    return rows


def load_css_records(run_id: str) -> list[dict]:
    d = REPO_ROOT / "runs" / run_id / "css"
    rows: list[dict] = []
    for p in sorted(d.glob("css_*.jsonl")):
        for r in load_jsonl(p):
            if "prompt_id" in r and "css_primary" in r:
                rows.append(r)
    print(f"  CSS rows loaded  : {len(rows)}")
    return rows


def build_df(eval_rows: list[dict], css_rows: list[dict]) -> pd.DataFrame:
    css_idx: dict[tuple, dict] = {
        (str(r["model"]), str(r["prompt_id"]), int(r["sample_idx"]),
         str(r.get("regime", "vanilla"))): r
        for r in css_rows
    }
    flat = []
    for row in eval_rows:
        if row.get("evaluation_failed"):
            continue
        model  = str(row["generator_model"])
        regime = str(row["generator_regime"])
        sidx   = int(row["generator_sample_idx"])
        pid    = str(row["prompt_id"])
        rt_raw = str(row.get("response_type_majority", "disputed"))
        rt     = rt_raw if rt_raw in RT_CATEGORIES else None

        css = css_idx.get((model, pid, sidx, regime))
        flat.append({
            "prompt_id":     pid,
            "category":      row.get("category"),
            "model":         model,
            "regime":        regime,
            "sample_idx":    sidx,
            "utility":       row.get("utility_mean"),
            "satisfaction":  row.get("satisfaction_mean"),
            "response_type": rt,
            "decisive_rate": (1.0 if rt == "decisive" else 0.0) if rt else None,
            "css_primary":   float(css["css_primary"]) if css else None,
            "css_max":       float(css["css_max"])     if css else None,
            "n_judges":      int(row.get("n_judges_used", 0)),
            # per-judge raw scores (private, for agreement computation)
            "_ujudge":  row.get("utility_per_judge", {}),
            "_sjudge":  row.get("satisfaction_per_judge", {}),
            "_rtjudge": row.get("response_type_per_judge", {}),
        })

    df = pd.DataFrame(flat)
    if not df.empty:
        print(
            f"  DataFrame built  : {len(df)} rows | "
            f"{df['model'].nunique()} model(s) | "
            f"{df['regime'].nunique()} regime(s) | "
            f"{df['prompt_id'].nunique()} unique prompt(s)"
        )
    return df


# ── Statistical helpers ───────────────────────────────────────────

def bootstrap_ci(
    arr, n_iter: int = 1000, seed: int = 42
) -> tuple[float, float]:
    a = np.asarray(arr, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.fromiter(
        (rng.choice(a, len(a), replace=True).mean() for _ in range(n_iter)),
        dtype=float, count=n_iter,
    )
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def cliffs_delta_paired(x: np.ndarray, y: np.ndarray) -> float:
    """δ = (#{d>0} − #{d<0}) / n for paired differences d = x − y."""
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    if n == 0:
        return np.nan
    return float((np.sum(d > 0) - np.sum(d < 0)) / n)


def safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank (two-sided). Returns (statistic, p_value)."""
    if len(x) < 3:
        return np.nan, np.nan
    diffs = np.asarray(x, float) - np.asarray(y, float)
    if np.all(diffs == 0):
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp_stats.wilcoxon(x, y, zero_method="zsplit", alternative="two-sided")
    return float(r.statistic), float(r.pvalue)


def _fmt(v, decimals: int = 6) -> str:
    """Format a float for CSV, returning '' for NaN/None."""
    if v is None:
        return ""
    try:
        f = float(v)
        return "" if np.isnan(f) else f"{f:.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _aligned_pairs(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric: str,
    key_col: str = "key",
) -> tuple[np.ndarray, np.ndarray]:
    """Align two DataFrames on key_col; return paired non-null arrays."""
    ma = df_a[[key_col, metric]].dropna(subset=[metric]).rename(columns={metric: "a"})
    mb = df_b[[key_col, metric]].dropna(subset=[metric]).rename(columns={metric: "b"})
    merged = ma.merge(mb, on=key_col)
    return merged["a"].values, merged["b"].values


# ── 1. Summary per (model, regime) ────────────────────────────────

def compute_summary(df: pd.DataFrame, n_iter: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for (model, regime), grp in df.groupby(["model", "regime"]):
        # Numeric metrics with bootstrap CI
        for metric in ["utility", "satisfaction", "decisive_rate", "css_primary"]:
            vals = grp[metric].dropna()
            if vals.empty:
                continue
            lo, hi = bootstrap_ci(vals.values, n_iter, seed)
            rows.append({
                "model":   model,
                "regime":  regime,
                "metric":  metric,
                "mean":    round(float(vals.mean()), 4),
                "ci_low":  "" if np.isnan(lo)  else round(lo, 4),
                "ci_high": "" if np.isnan(hi) else round(hi, 4),
                "n":       int(len(vals)),
            })

        # Response-type distribution (percentage of rows, no CI)
        total = len(grp)
        for rt in RT_CATEGORIES:
            cnt = int((grp["response_type"] == rt).sum())
            rows.append({
                "model":  model, "regime": regime,
                "metric": f"pct_{rt}",
                "mean":   round(cnt / total * 100, 2) if total else 0.0,
                "ci_low": "", "ci_high": "", "n": total,
            })
        disputed = total - int(grp["response_type"].notna().sum())
        rows.append({
            "model":  model, "regime": regime,
            "metric": "pct_disputed",
            "mean":   round(disputed / total * 100, 2) if total else 0.0,
            "ci_low": "", "ci_high": "", "n": total,
        })

    return pd.DataFrame(rows)


# ── 2. Pairwise Wilcoxon + Bonferroni + Cliff's delta ─────────────

def compute_pairwise_wilcoxon(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["key"] = df["prompt_id"] + "|" + df["sample_idx"].astype(str)

    rows: list[dict] = []
    model_scopes = list(df["model"].unique()) + ["pooled"]

    for scope in model_scopes:
        if scope == "pooled":
            sub = df.copy()
            # Include model in key so (model A, prompt P) ≠ (model B, prompt P)
            sub["key"] = sub["model"] + "|" + sub["key"]
        else:
            sub = df[df["model"] == scope].copy()

        for regime_a, regime_b in REGIME_PAIRS:
            df_a = sub[sub["regime"] == regime_a]
            df_b = sub[sub["regime"] == regime_b]
            if df_a.empty or df_b.empty:
                continue

            for metric in NUMERIC_METRICS:
                x, y = _aligned_pairs(df_a, df_b, metric)
                if len(x) < 3:
                    continue
                stat, pval = safe_wilcoxon(x, y)
                delta       = cliffs_delta_paired(x, y)
                significant = (
                    str(pval < BONFERRONI_ALPHA)
                    if not np.isnan(pval) else ""
                )
                rows.append({
                    "model_scope":        scope,
                    "regime_a":           regime_a,
                    "regime_b":           regime_b,
                    "metric":             metric,
                    "n_pairs":            int(len(x)),
                    "statistic":          _fmt(stat),
                    "p_value":            _fmt(pval),
                    "p_bonferroni_alpha": _fmt(BONFERRONI_ALPHA),
                    "significant_bonf":   significant,
                    "cliffs_delta":       _fmt(delta),
                })

    return pd.DataFrame(rows)


# ── 3. Friedman test across regimes ──────────────────────────────

def compute_friedman(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["block"] = (
        df["model"] + "|" + df["prompt_id"] + "|" + df["sample_idx"].astype(str)
    )

    rows: list[dict] = []
    model_scopes = list(df["model"].unique()) + ["pooled"]

    for scope in model_scopes:
        sub = df if scope == "pooled" else df[df["model"] == scope]

        for metric in ["utility", "satisfaction", "decisive_rate"]:
            pivot = sub.pivot_table(
                index="block", columns="regime", values=metric, aggfunc="first"
            )
            avail = [r for r in REGIMES if r in pivot.columns]
            if len(avail) < 3:
                continue
            complete = pivot[avail].dropna()
            if len(complete) < 3:
                continue
            groups = [complete[r].values for r in avail]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = sp_stats.friedmanchisquare(*groups)
            rows.append({
                "model_scope": scope,
                "metric":      metric,
                "n_regimes":   len(avail),
                "n_blocks":    len(complete),
                "statistic":   _fmt(result.statistic),
                "p_value":     _fmt(result.pvalue),
            })

    return pd.DataFrame(rows)


# ── 4. Inter-judge agreement ──────────────────────────────────────

def compute_judge_agreement(df: pd.DataFrame, judge_keys: list[str]) -> pd.DataFrame:
    if df.empty or not judge_keys:
        return pd.DataFrame()

    rows: list[dict] = []

    # ── Krippendorff's alpha for utility and satisfaction (ordinal) ──
    df_records = df.to_dict("records")
    for metric, col in [("utility", "_ujudge"), ("satisfaction", "_sjudge")]:
        # Build (n_judges, n_items) matrix; -1 → NaN
        matrix = []
        for jk in judge_keys:
            scores = []
            for rec in df_records:
                jdict = rec.get(col) or {}
                v = jdict.get(jk, -1)
                scores.append(float(v) if (v != -1 and v is not None) else np.nan)
            matrix.append(scores)

        rel = np.array(matrix, dtype=float)
        n_valid = int(np.sum(~np.all(np.isnan(rel), axis=0)))
        if n_valid < 2:
            print(f"  [Skip] Krippendorff {metric}: fewer than 2 items with any judge score")
            continue
        try:
            alpha = _kripp.alpha(rel, level_of_measurement="ordinal")
            rows.append({
                "metric":              f"{metric}_ordinal",
                "method":              "krippendorff_alpha",
                "value":               round(float(alpha), 4),
                "n_judges":            len(judge_keys),
                "n_items":             n_valid,
                "note":                "",
            })
        except Exception as e:
            print(f"  [Warning] Krippendorff alpha ({metric}): {e}")

    # ── Fleiss' kappa for response_type (nominal) ─────────────────
    if _STATSMODELS:
        rt_to_int = {rt: i for i, rt in enumerate(RT_CATEGORIES)}
        items = []
        for rec in df_records:
            jdict = rec.get("_rtjudge") or {}
            item_row = []
            for jk in judge_keys:
                rt = str(jdict.get(jk, "unknown"))
                item_row.append(rt_to_int.get(rt, np.nan))
            items.append(item_row)

        data_arr = np.array(items, dtype=float)
        # Only items where all judges have valid labels (no NaN)
        valid_mask = ~np.any(np.isnan(data_arr), axis=1)
        clean = data_arr[valid_mask].astype(int)

        if len(clean) < 2:
            note = f"only {len(clean)} item(s) with all-judge valid labels — skipped"
            print(f"  [Skip] Fleiss kappa: {note}")
            rows.append({
                "metric": "response_type_nominal",
                "method": "fleiss_kappa",
                "value":  "",
                "n_judges": len(judge_keys),
                "n_items": len(clean),
                "note":   note,
            })
        else:
            try:
                table, _ = _sm_agg(clean)
                kappa = _sm_fleiss(table)
                rows.append({
                    "metric":   "response_type_nominal",
                    "method":   "fleiss_kappa",
                    "value":    round(float(kappa), 4),
                    "n_judges": len(judge_keys),
                    "n_items":  len(clean),
                    "note":     "",
                })
            except Exception as e:
                print(f"  [Warning] Fleiss kappa: {e}")

    return pd.DataFrame(rows)


# ── 5. Human validation (optional) ───────────────────────────────

def compute_human_validation(run_id: str, df_llm: pd.DataFrame) -> pd.DataFrame | None:
    """
    Expects CSVs under runs/{run_id}/human_validation/ with columns:
      prompt_id, model, regime, sample_idx, annotator_id,
      utility_human, satisfaction_human, response_type_human
    """
    hv_dir = REPO_ROOT / "runs" / run_id / "human_validation"
    if not hv_dir.exists():
        return None

    import glob as _glob
    csvs = sorted(hv_dir.glob("*.csv"))
    if not csvs:
        return None

    human_rows: list[dict] = []
    for p in csvs:
        with open(p, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            human_rows.extend(reader)

    if not human_rows:
        return None

    hdf = pd.DataFrame(human_rows)
    for col in ["utility_human", "satisfaction_human"]:
        if col in hdf.columns:
            hdf[col] = pd.to_numeric(hdf[col], errors="coerce")

    results: list[dict] = []

    # Spearman: human consensus (mean across annotators per item) vs LLM judge mean
    if "utility_human" in hdf.columns and "satisfaction_human" in hdf.columns:
        hdf["item_key"] = (
            hdf["prompt_id"] + "|" + hdf.get("model", pd.Series([""] * len(hdf))) + "|"
            + hdf.get("regime", pd.Series([""] * len(hdf))) + "|"
            + hdf.get("sample_idx", pd.Series(["0"] * len(hdf))).astype(str)
        )
        consensus = hdf.groupby("item_key")[["utility_human", "satisfaction_human"]].mean()

        llm_df = df_llm.copy()
        llm_df["item_key"] = (
            llm_df["prompt_id"] + "|" + llm_df["model"] + "|"
            + llm_df["regime"] + "|" + llm_df["sample_idx"].astype(str)
        )
        llm_agg = llm_df.groupby("item_key")[["utility", "satisfaction"]].mean()
        merged = consensus.join(llm_agg, how="inner")

        for metric, human_col, llm_col in [
            ("utility", "utility_human", "utility"),
            ("satisfaction", "satisfaction_human", "satisfaction"),
        ]:
            sub = merged[[human_col, llm_col]].dropna()
            if len(sub) < 3:
                continue
            corr, pval = sp_stats.spearmanr(sub[human_col], sub[llm_col])
            results.append({
                "analysis": f"spearman_{metric}",
                "value":    round(float(corr), 4),
                "p_value":  _fmt(pval),
                "n":        len(sub),
                "note":     "human consensus vs LLM-judge mean",
            })

    # Krippendorff's alpha among human annotators
    if "annotator_id" in hdf.columns:
        annotators = sorted(hdf["annotator_id"].unique())
        hdf["item_key"] = hdf["prompt_id"].astype(str)
        for metric, human_col in [("utility_human", "utility_human"),
                                   ("satisfaction_human", "satisfaction_human")]:
            if human_col not in hdf.columns:
                continue
            pivot = hdf.pivot_table(
                index="item_key", columns="annotator_id", values=human_col, aggfunc="first"
            )
            rel = pivot[annotators].values.T  # (n_annotators, n_items)
            if rel.shape[1] < 2:
                continue
            try:
                alpha = _kripp.alpha(rel.astype(float), level_of_measurement="ordinal")
                results.append({
                    "analysis": f"krippendorff_{metric}_human",
                    "value":    round(float(alpha), 4),
                    "p_value":  "",
                    "n":        rel.shape[1],
                    "note":     f"among {len(annotators)} human annotators",
                })
            except Exception as e:
                print(f"  [Warning] Human Krippendorff ({metric}): {e}")

    return pd.DataFrame(results) if results else None


# ── 6. Summary JSON ───────────────────────────────────────────────

def assemble_summary_json(
    summary_df: pd.DataFrame,
    wilcoxon_df: pd.DataFrame,
    friedman_df: pd.DataFrame,
    judge_df: pd.DataFrame,
    df: pd.DataFrame,
) -> dict:
    out: dict = {
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bonferroni_n_tests": N_BONFERRONI,
        "bonferroni_alpha":   round(BONFERRONI_ALPHA, 6),
        "per_regime": {},
        "judge_agreement": {},
        "friedman": {},
        "significant_pairwise": [],
    }

    # Per-regime top-line means
    if not summary_df.empty:
        for (model, regime), grp in summary_df.groupby(["model", "regime"]):
            key = f"{model}/{regime}"
            rec: dict = {}
            for _, row in grp.iterrows():
                m = row["metric"]
                v = row["mean"]
                if m in ("utility", "satisfaction", "css_primary", "decisive_rate"):
                    rec[m] = v
                    lo = row["ci_low"]
                    hi = row["ci_high"]
                    if lo != "" and hi != "":
                        rec[f"{m}_ci"] = [lo, hi]
            if rec:
                out["per_regime"][key] = rec

    # Judge agreement
    if not judge_df.empty:
        for _, row in judge_df.iterrows():
            k = f"{row['method']}_{row['metric']}"
            v = row["value"]
            out["judge_agreement"][k] = float(v) if v not in ("", None) else None

    # Friedman
    if not friedman_df.empty:
        for _, row in friedman_df.iterrows():
            k = f"{row['model_scope']}_{row['metric']}"
            out["friedman"][k] = {
                "statistic": row["statistic"],
                "p_value":   row["p_value"],
                "n_blocks":  int(row["n_blocks"]),
            }

    # Significant pairwise Wilcoxon (after Bonferroni)
    if not wilcoxon_df.empty and "significant_bonf" in wilcoxon_df.columns:
        sig = wilcoxon_df[wilcoxon_df["significant_bonf"] == "True"]
        for _, row in sig.iterrows():
            out["significant_pairwise"].append({
                "scope":    row["model_scope"],
                "pair":     f"{row['regime_a']} vs {row['regime_b']}",
                "metric":   row["metric"],
                "p":        row["p_value"],
                "delta":    row["cliffs_delta"],
                "n_pairs":  int(row["n_pairs"]),
            })

    # Overall counts for sanity check
    if not df.empty:
        out["data_summary"] = {
            "n_eval_rows":       int(len(df)),
            "n_prompts":         int(df["prompt_id"].nunique()),
            "n_models":          int(df["model"].nunique()),
            "n_regimes":         int(df["regime"].nunique()),
            "pct_disputed":      round(float(df["response_type"].isna().mean() * 100), 2),
            "n_judges_mean":     round(float(df["n_judges"].mean()), 2),
        }

    return out


# ── Entry point ───────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistical analysis for C-NB-U benchmark"
    )
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=1000,
        help="Bootstrap resamples for CI (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for bootstrap (default: 42)",
    )
    args = parser.parse_args()

    run_path = REPO_ROOT / "runs" / args.run_id
    if not run_path.exists():
        print(f"Error: runs/{args.run_id}/ does not exist.")
        sys.exit(1)

    cfg = load_model_config()
    judge_keys = sorted(cfg.get("judges", {}).keys())

    stats_dir = run_path / "aggregated" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRun ID     : {args.run_id}")
    print(f"Bootstrap  : {args.bootstrap_iterations} iterations, seed={args.seed}")
    print(f"Bonferroni : α = 0.05 / {N_BONFERRONI} = {BONFERRONI_ALPHA:.6f}")
    print(f"Judge keys : {judge_keys}")
    print()

    # ── Load data ────────────────────────────────────────────────
    eval_rows = load_eval_records(args.run_id)
    css_rows  = load_css_records(args.run_id)
    df        = build_df(eval_rows, css_rows)
    print()

    if df.empty:
        print("No usable evaluation rows found. Exiting.")
        sys.exit(0)

    # ── 1. Summary per (model, regime) ──────────────────────────
    print("── 1. Summary per condition ──────────────────────────────")
    summary_df = compute_summary(df, args.bootstrap_iterations, args.seed)
    write_csv(summary_df, stats_dir / "summary_per_condition.csv")
    print()

    # ── 2. Pairwise Wilcoxon ────────────────────────────────────
    print("── 2. Pairwise Wilcoxon signed-rank ─────────────────────")
    n_regimes = df["regime"].nunique()
    if n_regimes < 2:
        print(f"  [Skip] Only {n_regimes} regime(s) present — need ≥2 for pairwise tests.")
        wilcoxon_df = pd.DataFrame()
    else:
        wilcoxon_df = compute_pairwise_wilcoxon(df)
        if not wilcoxon_df.empty:
            n_sig = (wilcoxon_df["significant_bonf"] == "True").sum()
            print(f"  {len(wilcoxon_df)} tests computed, {n_sig} significant after Bonferroni")
        write_csv(wilcoxon_df, stats_dir / "pairwise_wilcoxon.csv")
    print()

    # ── 3. Friedman test ────────────────────────────────────────
    print("── 3. Friedman test ─────────────────────────────────────")
    if n_regimes < 3:
        print(f"  [Skip] Only {n_regimes} regime(s) — Friedman requires ≥3.")
        friedman_df = pd.DataFrame()
    else:
        friedman_df = compute_friedman(df)
        write_csv(friedman_df, stats_dir / "friedman.csv")
    print()

    # ── 4. Inter-judge agreement ────────────────────────────────
    print("── 4. Inter-judge agreement ─────────────────────────────")
    judge_df = compute_judge_agreement(df, judge_keys)
    write_csv(judge_df, stats_dir / "judge_agreement.csv")
    print()

    # ── 5. Human validation ─────────────────────────────────────
    print("── 5. Human validation ──────────────────────────────────")
    human_df = compute_human_validation(args.run_id, df)
    if human_df is None:
        print(f"  [Skip] No human_validation/ directory or CSV files found.")
    else:
        write_csv(human_df, stats_dir / "human_validation.csv")
    print()

    # ── 6. Summary JSON ─────────────────────────────────────────
    print("── 6. Summary JSON ──────────────────────────────────────")
    summary_json = assemble_summary_json(
        summary_df, wilcoxon_df, friedman_df, judge_df, df
    )
    write_json(summary_json, stats_dir / "summary.json")
    print()

    # Print to stdout for quick inspection
    print("── summary.json ─────────────────────────────────────────")
    print(json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()
