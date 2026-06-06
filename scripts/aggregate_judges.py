#!/usr/bin/env python3
"""
Multi-judge aggregation for the CIKM C-NB-U benchmark.

For each (generator_model, regime, sample_idx) group, combines the per-judge
evaluation JSONL files into a single aggregated file applying CLAUDE.md rules:
  - utility/satisfaction: mean of non-failed (≠ -1) judge scores (1–5 scale)
  - response_type: majority vote; "disputed" when all 3 judges disagree

Input:
  runs/{run_id}/evaluations/{judge_key}_evaluations_{model}_{regime}_sample{n}.jsonl

Output:
  runs/{run_id}/aggregated/per_response/{model}_{regime}_sample{n}_aggregated.jsonl
"""

import sys
import json
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gen_common import load_model_config, REPO_ROOT


# ── File I/O ──────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


# ── File discovery ────────────────────────────────────────────────

def discover_eval_data(
    eval_dir: Path,
    judge_keys: list[str],
) -> dict[tuple[str, str, int, str], dict[str, dict]]:
    """
    Scan eval_dir for all *_evaluations_*.jsonl files.
    Returns a mapping:
      (generator_model, generator_regime, generator_sample_idx, prompt_id)
      → {judge_key: row_dict}
    """
    data: dict[tuple[str, str, int, str], dict[str, dict]] = defaultdict(dict)
    files_loaded = 0

    for path in sorted(eval_dir.glob("*_evaluations_*.jsonl")):
        judge_key = None
        for jk in judge_keys:
            if path.name.startswith(f"{jk}_evaluations_"):
                judge_key = jk
                break
        if judge_key is None:
            print(f"  [Skip] Unrecognised judge prefix in: {path.name}")
            continue

        rows = load_jsonl(path)
        for row in rows:
            model = str(row.get("generator_model", ""))
            regime = str(row.get("generator_regime", ""))
            sidx = int(row.get("generator_sample_idx", 0))
            pid = str(row.get("prompt_id", ""))
            if not (model and regime and pid):
                continue
            data[(model, regime, sidx, pid)][judge_key] = row

        files_loaded += 1

    print(f"  Loaded {files_loaded} evaluation file(s) from {eval_dir.relative_to(REPO_ROOT)}")
    return data


# ── Per-row aggregation ───────────────────────────────────────────

def aggregate_prompt_row(
    prompt_id: str,
    category,
    model: str,
    regime: str,
    sidx: int,
    judge_data: dict[str, dict],
    all_judge_keys: list[str],
) -> dict:
    """
    Aggregate 3-judge scores for a single (prompt, model, regime, sample) tuple.
    """
    utility_per_judge: dict[str, int] = {}
    satisfaction_per_judge: dict[str, int] = {}
    response_type_per_judge: dict[str, str] = {}

    for jk in all_judge_keys:
        if jk in judge_data:
            row = judge_data[jk]
            utility_per_judge[jk] = int(row.get("utility_score", -1))
            satisfaction_per_judge[jk] = int(row.get("satisfaction_score", -1))
            rt = str(row.get("response_type", "unknown")).strip().lower()
            response_type_per_judge[jk] = rt if rt in {
                "decisive", "clarifying", "hedging", "refusal"
            } else "unknown"
        else:
            utility_per_judge[jk] = -1
            satisfaction_per_judge[jk] = -1
            response_type_per_judge[jk] = "unknown"

    valid_utilities = [v for v in utility_per_judge.values() if v != -1]
    valid_satisfactions = [v for v in satisfaction_per_judge.values() if v != -1]
    valid_types = [v for v in response_type_per_judge.values() if v != "unknown"]

    # Evaluation failure: all utility scores failed OR all response_types unknown
    evaluation_failed = (not valid_utilities) or (not valid_types)

    if evaluation_failed:
        return {
            "prompt_id": prompt_id,
            "category": category,
            "generator_model": model,
            "generator_regime": regime,
            "generator_sample_idx": sidx,
            "utility_per_judge": utility_per_judge,
            "satisfaction_per_judge": satisfaction_per_judge,
            "response_type_per_judge": response_type_per_judge,
            "utility_mean": None,
            "satisfaction_mean": None,
            "response_type_majority": "disputed",
            "evaluation_failed": True,
            "n_judges_used": 0,
            "aggregated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    utility_mean = round(sum(valid_utilities) / len(valid_utilities), 4)
    satisfaction_mean = (
        round(sum(valid_satisfactions) / len(valid_satisfactions), 4)
        if valid_satisfactions
        else None
    )

    # Majority vote: need 2+ of the valid labels to agree
    type_counts = Counter(valid_types)
    most_common_label, most_common_count = type_counts.most_common(1)[0]
    rt_majority = most_common_label if most_common_count >= 2 else "disputed"

    return {
        "prompt_id": prompt_id,
        "category": category,
        "generator_model": model,
        "generator_regime": regime,
        "generator_sample_idx": sidx,
        "utility_per_judge": utility_per_judge,
        "satisfaction_per_judge": satisfaction_per_judge,
        "response_type_per_judge": response_type_per_judge,
        "utility_mean": utility_mean,
        "satisfaction_mean": satisfaction_mean,
        "response_type_majority": rt_majority,
        "evaluation_failed": False,
        "n_judges_used": len(valid_utilities),
        "aggregated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ── Output path ───────────────────────────────────────────────────

def make_aggregated_output_path(
    run_id: str, model: str, regime: str, sidx: int
) -> Path:
    out_dir = REPO_ROOT / "runs" / run_id / "aggregated" / "per_response"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{model}_{regime}_sample{sidx}_aggregated.jsonl"


# ── Group processing ──────────────────────────────────────────────

def process_group(
    model: str,
    regime: str,
    sidx: int,
    group_data: dict[str, dict[str, dict]],
    all_judge_keys: list[str],
    run_id: str,
) -> tuple[int, int, int]:
    """
    Aggregate all prompt rows for one (model, regime, sidx) group.
    Returns (n_clean, n_partial, n_failed, n_disputed).
    Actually returns (n_clean, n_partial, n_failed) — disputed is a subtype of clean.
    Returns (n_complete, n_partial, n_failed) where:
      n_complete = all judges present, no parse failures
      n_partial  = ≥1 valid judge but <3 judges
      n_failed   = evaluation_failed=True
    """
    out_path = make_aggregated_output_path(run_id, model, regime, sidx)
    n_complete = n_partial = n_failed = n_disputed = 0

    agg_rows = []
    for pid in sorted(group_data.keys()):
        judge_data = group_data[pid]
        # Get category from whichever judge row has it
        category = ""
        for row in judge_data.values():
            category = row.get("category", "")
            if category != "":
                break

        agg = aggregate_prompt_row(
            prompt_id=pid,
            category=category,
            model=model,
            regime=regime,
            sidx=sidx,
            judge_data=judge_data,
            all_judge_keys=all_judge_keys,
        )
        agg_rows.append(agg)

        if agg["evaluation_failed"]:
            n_failed += 1
        elif agg["n_judges_used"] < len(all_judge_keys):
            n_partial += 1
        else:
            n_complete += 1

        if agg["response_type_majority"] == "disputed":
            n_disputed += 1

    with open(out_path, "w", encoding="utf-8") as f:
        for row in agg_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(agg_rows)
    print(
        f"  [{model} / {regime} / sample{sidx}] "
        f"{total} rows → "
        f"complete={n_complete}, partial={n_partial}, "
        f"failed={n_failed}, disputed={n_disputed}"
    )
    print(f"    → {out_path.relative_to(REPO_ROOT)}")

    return n_complete, n_partial, n_failed, n_disputed


# ── Entry point ───────────────────────────────────────────────────

def main() -> None:
    cfg = load_model_config()
    all_judge_keys = sorted(cfg.get("judges", {}).keys())
    if not all_judge_keys:
        print("Error: no 'judges' entries in config/models.json.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Aggregate per-judge evaluation files for C-NB-U benchmark"
    )
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument(
        "--model", default=None,
        help="Generator model key (omit to process all)",
    )
    parser.add_argument(
        "--regime", default=None,
        help="Prompting regime (omit to process all 4)",
    )
    parser.add_argument(
        "--sample-idx", type=int, default=None,
        help="Sample index (omit to process all)",
    )
    args = parser.parse_args()

    runs_dir = REPO_ROOT / "runs"
    run_path = runs_dir / args.run_id
    if not run_path.exists():
        print(f"Error: runs/{args.run_id}/ does not exist.")
        sys.exit(1)

    eval_dir = run_path / "evaluations"
    if not eval_dir.exists():
        print(f"Error: runs/{args.run_id}/evaluations/ does not exist.")
        sys.exit(1)

    print(f"Run ID     : {args.run_id}")
    print(f"Judge keys : {all_judge_keys}")
    print()

    # Load all eval data indexed by (model, regime, sidx, prompt_id) -> {jk: row}
    raw_data = discover_eval_data(eval_dir, all_judge_keys)

    if not raw_data:
        print("No evaluation rows found. Check that eval files exist and judge prefixes match.")
        sys.exit(0)

    # Re-index into groups: (model, regime, sidx) -> {prompt_id -> {judge_key: row}}
    groups: dict[tuple[str, str, int], dict[str, dict[str, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (model, regime, sidx, pid), judge_data in raw_data.items():
        groups[(model, regime, sidx)][pid] = judge_data

    # Apply CLI filters
    filtered_groups = {}
    for (model, regime, sidx), prompts in groups.items():
        if args.model and model != args.model:
            continue
        if args.regime and regime != args.regime:
            continue
        if args.sample_idx is not None and sidx != args.sample_idx:
            continue
        filtered_groups[(model, regime, sidx)] = prompts

    if not filtered_groups:
        print("No groups match the specified filters.")
        sys.exit(0)

    print(f"\nGroups to process: {len(filtered_groups)}")
    print()

    total_complete = total_partial = total_failed = total_disputed = 0

    for (model, regime, sidx) in sorted(filtered_groups.keys()):
        n_c, n_p, n_f, n_d = process_group(
            model=model,
            regime=regime,
            sidx=sidx,
            group_data=filtered_groups[(model, regime, sidx)],
            all_judge_keys=all_judge_keys,
            run_id=args.run_id,
        )
        total_complete += n_c
        total_partial += n_p
        total_failed += n_f
        total_disputed += n_d

    total_rows = total_complete + total_partial + total_failed
    print(f"\n{'─'*50}")
    print(f"Aggregation complete. Total rows: {total_rows}")
    print(f"  All {len(all_judge_keys)} judges present : {total_complete}")
    print(f"  Partial (≥1 judge missing)  : {total_partial}")
    print(f"  Evaluation failed (all bad) : {total_failed}")
    print(f"  Response type disputed      : {total_disputed}")


if __name__ == "__main__":
    main()
