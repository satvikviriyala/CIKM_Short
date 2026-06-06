#!/usr/bin/env python3
"""
Criterion Sensitivity Score (CSS) computation for CIKM C-NB-U benchmark.

For each (prompt_id, model, sample_idx) tuple reads:
  - the vanilla response from  runs/{run_id}/responses/{model}_vanilla_sample{s}_responses.jsonl
  - the criterion-variant responses from
      runs/{run_id}/responses/{model}_variant_{i}_sample{s}_responses.jsonl

Embeds every response with a sentence-transformer, computes cosine similarities
between the vanilla embedding and each criterion-variant embedding, and derives:

  css_primary = max_i(sim_i) - mean_i(sim_i)   [spread-based; used in main paper]
  css_max     = max_i(sim_i)                     [raw max; reported in supplementary]

Output:
  runs/{run_id}/css/css_{model}_sample{n}.jsonl  (one per sample)
  runs/{run_id}/css/css_aggregated.csv           (mean by model × category)
"""

import sys
import json
import argparse
import csv
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gen_common import load_model_config, REPO_ROOT

# ── Device detection ─────────────────────────────────────────────

try:
    import torch
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _DEVICE = "cpu"


# ── Sentence-transformer loading ──────────────────────────────────

PRIMARY_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
FALLBACK_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_sentence_transformer(model_name: str) -> tuple:
    """
    Return (SentenceTransformer, actual_model_name).
    Falls back to all-MiniLM-L6-v2 if the primary fails to load.
    """
    from sentence_transformers import SentenceTransformer

    for name in (model_name, FALLBACK_EMBED_MODEL):
        try:
            st = SentenceTransformer(name, device=_DEVICE)
            if name != model_name:
                print(
                    f"[Fallback] Could not load '{model_name}'; "
                    f"using '{name}' instead."
                )
            return st, name
        except Exception as exc:
            print(f"[Warning] Could not load '{name}': {exc}")

    raise RuntimeError(
        "Could not load any sentence-transformer model. "
        "Run: pip install sentence-transformers"
    )


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


# ── Cosine similarity ─────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ── Per-model CSS computation ─────────────────────────────────────

ALL_REGIMES = ["vanilla", "utility_first", "neutrality_oriented", "clarification_first"]


def process_model(
    run_id: str,
    model_key: str,
    st_model,
    embedding_model_name: str,
    runs_dir: Path,
    regimes: list[str] | None = None,
) -> list[dict]:
    """
    For each regime, load that regime's responses and compare against criterion-
    variant responses.  Embed everything in one batch for efficiency.  Returns
    CSS output rows with a ``regime`` field.
    """
    regimes = regimes or ALL_REGIMES
    resp_dir = runs_dir / run_id / "responses"

    # ── 1. Load variant responses (shared across all regimes) ────
    variants: dict[tuple[str, int, int], dict] = {}
    n_variant_files = 0
    vi = 0
    while True:
        found = False
        for s in range(3):
            path = resp_dir / f"{model_key}_variant_{vi}_sample{s}_responses.jsonl"
            if not path.exists():
                continue
            found = True
            for row in load_jsonl(path):
                key = (
                    str(row.get("prompt_id", "")),
                    int(row.get("criterion_variant_idx", vi)),
                    int(row.get("sample_idx", s)),
                )
                variants[key] = row
        if not found:
            break
        n_variant_files += 1
        vi += 1

    if n_variant_files == 0:
        print(f"  [{model_key}] No variant responses found — skipping.")
        return []

    # ── 2. Load regime responses and build work items ────────────
    work_items: list[dict] = []
    n_skipped = 0

    for regime in regimes:
        regime_responses: dict[tuple[str, int], dict] = {}
        for s in range(3):
            path = resp_dir / f"{model_key}_{regime}_sample{s}_responses.jsonl"
            for row in load_jsonl(path):
                key = (str(row.get("prompt_id", "")), int(row.get("sample_idx", s)))
                regime_responses[key] = row

        if not regime_responses:
            print(f"  [{model_key}/{regime}] No responses found — skipping regime.")
            continue

        regime_skipped = 0
        for (pid, sidx), resp_row in sorted(regime_responses.items()):
            resp_text = resp_row.get("response", "")
            if not resp_text.strip() or resp_text.startswith("ERROR:"):
                regime_skipped += 1
                continue

            matched_variants = []
            for v_i in range(n_variant_files):
                vkey = (pid, v_i, sidx)
                if vkey not in variants:
                    continue
                var_text = variants[vkey].get("response", "")
                if not var_text.strip() or var_text.startswith("ERROR:"):
                    continue
                matched_variants.append((v_i, variants[vkey]))

            if len(matched_variants) < 2:
                regime_skipped += 1
                continue

            work_items.append({
                "regime": regime,
                "pid": pid,
                "sidx": sidx,
                "resp_row": resp_row,
                "variant_list": matched_variants,
            })
        n_skipped += regime_skipped
        print(
            f"  [{model_key}/{regime}] {len(regime_responses)} responses, "
            f"{regime_skipped} skipped"
        )

    print(
        f"  [{model_key}] variants={len(variants)} rows across "
        f"{n_variant_files} variant index(es)"
    )

    if not work_items:
        print(f"  [{model_key}] No complete (response + ≥2 variants) tuples — skipping.")
        return []

    print(
        f"  [{model_key}] Computing CSS for {len(work_items)} "
        f"(regime, prompt, sample) tuples  ({n_skipped} skipped total)"
    )

    # ── 3. Batch-encode all unique response texts ────────────────
    text_to_idx: dict[str, int] = {}
    all_texts: list[str] = []

    def register(text: str) -> int:
        if text not in text_to_idx:
            text_to_idx[text] = len(all_texts)
            all_texts.append(text)
        return text_to_idx[text]

    for item in work_items:
        register(item["resp_row"]["response"])
        for _, vrow in item["variant_list"]:
            register(vrow["response"])

    print(f"  [{model_key}] Encoding {len(all_texts)} unique texts on {_DEVICE}…")
    embeddings = st_model.encode(
        all_texts,
        batch_size=64,
        show_progress_bar=len(all_texts) > 100,
        convert_to_numpy=True,
    )

    # ── 4. Compute CSS ────────────────────────────────────────────
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    css_rows: list[dict] = []

    for item in work_items:
        pid = item["pid"]
        sidx = item["sidx"]
        regime = item["regime"]
        resp_row = item["resp_row"]
        variant_list = item["variant_list"]

        resp_emb = embeddings[text_to_idx[resp_row["response"]]]

        sims: list[float] = []
        for _, vrow in variant_list:
            var_emb = embeddings[text_to_idx[vrow["response"]]]
            sims.append(cosine_sim(resp_emb, var_emb))

        sims_arr = np.array(sims, dtype=np.float64)
        css_primary = float(sims_arr.max() - sims_arr.mean())
        css_max = float(sims_arr.max())
        best_local_idx = int(np.argmax(sims_arr))
        best_vi, best_vrow = variant_list[best_local_idx]

        css_rows.append({
            "run_id": run_id,
            "prompt_id": pid,
            "category": resp_row.get("category", ""),
            "model": model_key,
            "regime": regime,
            "sample_idx": sidx,
            "n_variants": len(variant_list),
            "similarities": [round(s, 6) for s in sims],
            "css_primary": round(css_primary, 6),
            "css_max": round(css_max, 6),
            "argmax_criterion_idx": best_vi,
            "argmax_criterion_text": best_vrow.get("criterion_variant_text", ""),
            "embedding_model": embedding_model_name,
            "computed_at": now,
        })

    return css_rows


# ── Output writers ────────────────────────────────────────────────

def write_per_sample_jsonl(
    run_id: str, model_key: str, css_rows: list[dict], runs_dir: Path
) -> None:
    css_dir = runs_dir / run_id / "css"
    css_dir.mkdir(parents=True, exist_ok=True)

    by_regime_sample: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in css_rows:
        by_regime_sample[(row["regime"], row["sample_idx"])].append(row)

    for (regime, sidx), rows in sorted(by_regime_sample.items()):
        path = css_dir / f"css_{model_key}_{regime}_sample{sidx}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  Wrote {len(rows)} rows → {path.relative_to(REPO_ROOT)}")


def write_aggregated_csv(
    run_id: str, all_css_rows: list[dict], runs_dir: Path
) -> None:
    css_dir = runs_dir / run_id / "css"
    css_dir.mkdir(parents=True, exist_ok=True)

    agg: dict[tuple[str, str, str], dict] = {}
    for row in all_css_rows:
        key = (row["model"], row["regime"], str(row["category"]))
        if key not in agg:
            agg[key] = {"primary_sum": 0.0, "max_sum": 0.0, "n": 0}
        agg[key]["primary_sum"] += row["css_primary"]
        agg[key]["max_sum"] += row["css_max"]
        agg[key]["n"] += 1

    path = css_dir / "css_aggregated.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "regime", "category",
                "mean_css_primary", "mean_css_max", "n_observations",
            ],
        )
        writer.writeheader()
        for (model, regime, category), vals in sorted(agg.items()):
            n = vals["n"]
            writer.writerow({
                "model": model,
                "regime": regime,
                "category": category,
                "mean_css_primary": round(vals["primary_sum"] / n, 6),
                "mean_css_max": round(vals["max_sum"] / n, 6),
                "n_observations": n,
            })

    print(f"\n  Aggregated CSV ({len(agg)} rows) → {path.relative_to(REPO_ROOT)}")


# ── Entry point ───────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute CSS (Criterion Sensitivity Score) for C-NB-U benchmark"
    )
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument(
        "--model", default=None,
        help="Model key from config/models.json (omit to process all models)",
    )
    parser.add_argument(
        "--embedding-model", default=PRIMARY_EMBED_MODEL,
        help=f"Sentence-transformers model (default: {PRIMARY_EMBED_MODEL})",
    )
    args = parser.parse_args()

    cfg = load_model_config()
    all_gen_models = cfg.get("generation", {})

    if args.model:
        if args.model not in all_gen_models:
            print(f"Error: '{args.model}' not in config/models.json generation keys.")
            sys.exit(1)
        model_keys = [args.model]
    else:
        model_keys = sorted(all_gen_models.keys())

    runs_dir = REPO_ROOT / "runs"
    run_path = runs_dir / args.run_id
    if not run_path.exists():
        print(f"Error: runs/{args.run_id}/ does not exist.")
        sys.exit(1)

    print(f"Run ID         : {args.run_id}")
    print(f"Models         : {model_keys}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Device         : {_DEVICE}")
    print()

    st_model, actual_embed_name = load_sentence_transformer(args.embedding_model)

    all_css_rows: list[dict] = []

    for model_key in model_keys:
        print(f"\n{'─'*50}")
        print(f"  Model: {model_key}")
        css_rows = process_model(
            run_id=args.run_id,
            model_key=model_key,
            st_model=st_model,
            embedding_model_name=actual_embed_name,
            runs_dir=runs_dir,
        )

        if not css_rows:
            continue

        write_per_sample_jsonl(args.run_id, model_key, css_rows, runs_dir)
        all_css_rows.extend(css_rows)

        primaries = np.array([r["css_primary"] for r in css_rows])
        maxes = np.array([r["css_max"] for r in css_rows])
        print(
            f"  Summary: n={len(css_rows)}, "
            f"mean css_primary={primaries.mean():.4f} "
            f"(std={primaries.std():.4f}), "
            f"mean css_max={maxes.mean():.4f}"
        )

    if all_css_rows:
        write_aggregated_csv(args.run_id, all_css_rows, runs_dir)
        print(f"\n✔ CSS computation complete. Total rows: {len(all_css_rows)}")
    else:
        print("\nNo CSS rows produced — check that vanilla and variant files exist in the run.")


if __name__ == "__main__":
    main()
