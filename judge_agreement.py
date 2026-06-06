"""
Compute comprehensive inter-judge agreement metrics for the C-NB-U paper.
Filters out invalid scores (-1). Computes: Krippendorff's alpha, ICC(2,k),
within-±1 agreement, Quadratic Weighted Kappa, Spearman rank correlation.
"""
import pandas as pd
import numpy as np
import glob
import os
from itertools import combinations
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import pingouin as pg

# ── Load all evaluation data ──
eval_dir = "runs/run_20260521/evaluations"

rows = []
for f in glob.glob(os.path.join(eval_dir, "*.jsonl")):
    df = pd.read_json(f, lines=True)
    for _, r in df.iterrows():
        # Skip invalid scores
        if r["utility_score"] < 1 or r["satisfaction_score"] < 1:
            continue
        rows.append({
            "prompt_id": r["prompt_id"],
            "model": r["generator_model"],
            "regime": r["generator_regime"],
            "sample": r["generator_sample_idx"],
            "judge": r["judge_key"],
            "utility": int(r["utility_score"]),
            "satisfaction": int(r["satisfaction_score"]),
        })

data = pd.DataFrame(rows)
print(f"Loaded {len(data)} valid evaluation records (dropped scores < 1)")
print(f"Unique judges: {sorted(data['judge'].unique())}")
print()

def get_judge_matrix(data, metric):
    """Pivot to items × judges matrix, drop items without all 3 judges."""
    pivot = data.pivot_table(
        index=["prompt_id", "model", "regime", "sample"],
        columns="judge",
        values=metric,
        aggfunc="first"
    ).dropna()
    return pivot

for metric_name in ["utility", "satisfaction"]:
    print(f"{'='*60}")
    print(f"  {metric_name.upper()} AGREEMENT")
    print(f"{'='*60}")
    
    mat = get_judge_matrix(data, metric_name)
    judge_cols = sorted(mat.columns)
    n_items = len(mat)
    print(f"Items with all 3 judges: {n_items}")
    print(f"Score range: {int(mat.values.min())} - {int(mat.values.max())}")
    print(f"Mean ± std: {mat.values.mean():.2f} ± {mat.values.std():.2f}")
    print()
    
    # ── 1. Krippendorff's alpha (interval) ──
    def krippendorff_alpha_interval(mat):
        n, k = mat.shape
        vals = mat.values
        all_vals = vals.flatten()
        N = len(all_vals)
        
        # Observed disagreement
        Do = 0
        n_pairs = 0
        for i in range(n):
            for a in range(k):
                for b in range(a+1, k):
                    Do += (vals[i, a] - vals[i, b])**2
                    n_pairs += 1
        Do /= n_pairs
        
        # Expected disagreement
        De = np.var(all_vals, ddof=0) * N / (N - 1)
        
        return 1 - Do / De if De > 0 else 0.0
    
    alpha = krippendorff_alpha_interval(mat)
    print(f"Krippendorff's alpha (interval): {alpha:.3f}")
    
    # ── 2. ICC(2,k) — Two-way random, average measures ──
    long_data = []
    for idx, (item, row) in enumerate(mat.iterrows()):
        for j in judge_cols:
            long_data.append({"item": idx, "judge": j, "score": row[j]})
    long_df = pd.DataFrame(long_data)
    
    icc_result = pg.intraclass_corr(
        data=long_df, targets="item", raters="judge", ratings="score"
    )
    # Print all ICC types to find the right one
    for _, icc_row in icc_result.iterrows():
        if "2k" in icc_row["Type"] or "ICC2k" in icc_row["Type"]:
            icc_val = icc_row["ICC"]
            ci = icc_row["CI95%"]
            print(f"ICC({icc_row['Type']}): {icc_val:.3f} [95% CI: {ci[0]:.3f}, {ci[1]:.3f}]")
    
    # Also print all types for reference
    print("\n  All ICC types:")
    for _, icc_row in icc_result.iterrows():
        print(f"    {icc_row['Type']}: {icc_row['ICC']:.3f}")
    print()
    
    # ── 3. Percent agreement within ±1 ──
    agree_count = 0
    total_pairs = 0
    for i in range(n_items):
        vals = mat.iloc[i].values
        for a, b in combinations(range(len(judge_cols)), 2):
            if abs(vals[a] - vals[b]) <= 1:
                agree_count += 1
            total_pairs += 1
    pct_agree = 100.0 * agree_count / total_pairs
    print(f"Within-±1 agreement: {pct_agree:.1f}%")
    
    # ── 4. Quadratic Weighted Kappa ──
    kappa_vals = []
    for j1, j2 in combinations(judge_cols, 2):
        kappa = cohen_kappa_score(
            mat[j1].astype(int), mat[j2].astype(int), weights="quadratic"
        )
        kappa_vals.append(kappa)
        print(f"  QWK({j1} vs {j2}): {kappa:.3f}")
    mean_qwk = np.mean(kappa_vals)
    print(f"Mean Quadratic Weighted Kappa: {mean_qwk:.3f}")
    
    # ── 5. Spearman rank correlation ──
    spearman_vals = []
    for j1, j2 in combinations(judge_cols, 2):
        rho, pval = stats.spearmanr(mat[j1], mat[j2])
        spearman_vals.append(rho)
        print(f"  Spearman({j1} vs {j2}): ρ={rho:.3f}, p={pval:.2e}")
    mean_rho = np.mean(spearman_vals)
    print(f"Mean Spearman ρ: {mean_rho:.3f}")
    print()

print("="*60)
print("  SUMMARY FOR PAPER")
print("="*60)
print("Use these values to update the Results and Limitations sections.")
