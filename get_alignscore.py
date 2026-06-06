import pandas as pd
import glob
import numpy as np

# Load alignscore data
dfs = []
for p in glob.glob("runs/run_20260521/alignscore/alignscore_*.jsonl"):
    dfs.append(pd.read_json(p, lines=True))
df = pd.concat(dfs, ignore_index=True)

print("POOLED ALIGNSCORE (STRONG CORRECTNESS) ACROSS MODELS:")
for regime in ["vanilla", "utility_first", "neutrality_oriented", "clarification_first"]:
    sub = df[df["regime"] == regime]
    c_strong = sub["c_strong"].mean() * 100
    c_strong_std = sub["c_strong"].std() * 100
    
    print(f"Regime: {regime}")
    print(f"  % Strong Correctness: {c_strong:.1f}% ± {c_strong_std:.1f}%")

# We can also compute per-model strong correctness if we want to add to Table 3
print("\nPER MODEL, PER REGIME STRONG CORRECTNESS:")
for model in df["model"].unique():
    for regime in ["vanilla", "utility_first", "neutrality_oriented", "clarification_first"]:
        sub = df[(df["model"] == model) & (df["regime"] == regime)]
        if not sub.empty:
            c_strong = sub["c_strong"].mean() * 100
            print(f"{model} - {regime}: {c_strong:.1f}%")
