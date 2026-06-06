import pandas as pd
import glob
dfs = []
for p in glob.glob("runs/run_20260521/css/css_*.jsonl"):
    dfs.append(pd.read_json(p, lines=True))
css_df = pd.concat(dfs, ignore_index=True)
for regime in ["vanilla", "utility_first", "neutrality_oriented", "clarification_first"]:
    sub = css_df[css_df["regime"] == regime] if "regime" in css_df.columns else css_df
    print(f"{regime}: CSS = {sub['css_primary'].mean():.4f} ± {sub['css_primary'].std():.4f}")
