import pandas as pd
import numpy as np
import glob

# Load aggregated data
dfs = []
for p in glob.glob("runs/run_20260521/aggregated/per_response/*_aggregated.jsonl"):
    dfs.append(pd.read_json(p, lines=True))
df = pd.concat(dfs, ignore_index=True)
df = df[df["evaluation_failed"] != True]

print("POOLED ACROSS MODELS:")
for regime in ["vanilla", "utility_first", "neutrality_oriented", "clarification_first"]:
    sub = df[df["generator_regime"] == regime]
    util_mean = sub["utility_mean"].mean()
    util_std = sub["utility_mean"].std()
    sat_mean = sub["satisfaction_mean"].mean()
    sat_std = sub["satisfaction_mean"].std()
    
    # CSS std
    css_dfs = []
    for p in glob.glob("runs/run_20260521/css/css_*.jsonl"):
        css_dfs.append(pd.read_json(p, lines=True))
    css_df = pd.concat(css_dfs, ignore_index=True)
    
    if "regime" in css_df.columns:
        css_sub = css_df[css_df["regime"] == regime]
    else:
        # if regime isn't saved in css_df, it is only for vanilla?
        # Actually CSS computes per-regime in newer versions
        css_sub = css_df
        
    css_mean = css_sub["css_primary"].mean() if "css_primary" in css_sub.columns else 0.0
    css_std = css_sub["css_primary"].std() if "css_primary" in css_sub.columns else 0.0

    print(f"Regime: {regime}")
    print(f"  Utility: {util_mean:.2f} ± {util_std:.2f}")
    print(f"  Satisf.: {sat_mean:.2f} ± {sat_std:.2f}")
    
    dec = 100 * (sub['response_type_majority'] == 'decisive').mean()
    cla = 100 * (sub['response_type_majority'] == 'clarifying').mean()
    hed = 100 * (sub['response_type_majority'] == 'hedging').mean()
    ref = 100 * (sub['response_type_majority'] == 'refusal').mean()
    
    print(f"  % Decisive: {dec:.1f}%")
    print(f"  % Clarify:  {cla:.1f}%")
    print(f"  % Hedging:  {hed:.1f}%")
    print(f"  % Refusal:  {ref:.1f}%")
