#!/usr/bin/env python3
"""
Supplementary alignment/entailment check for CIKM C-NB-U benchmark.

For each response row, scores how well the model's output aligns with
(is entailed by) the original prompt premises. This is a supplementary
correctness proxy — it does not replace the LLM-as-judge evaluation.

Checker priority:
  1. AlignScore-large  (https://github.com/yuh-zha/AlignScore)
     Install: pip install alignscore && python -m alignscore.download_ckpt large
  2. Fallback: cross-encoder/nli-deberta-v3-large via sentence_transformers

alignscore_raw: float in [0, 1]  (entailment probability or AlignScore output)
c_strong:       1 if alignscore_raw >= 0.80, else 0

Input:
  runs/{run_id}/responses/{model}_{regime}_sample{n}_responses.jsonl
Output:
  runs/{run_id}/alignscore/alignscore_{model}_{regime}_sample{n}.jsonl
"""

import sys
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gen_common import REPO_ROOT

# ── Device detection ──────────────────────────────────────────────

try:
    import torch
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _DEVICE = "cpu"

# ── Checker backend detection ─────────────────────────────────────

_BACKEND = None          # set in load_checker()
_CHECKER = None          # loaded model object
_CHECKER_NAME = None     # string name logged to each output row
_ENTAIL_IDX = 2          # label index for entailment (NLI fallback)
_ALIGN_THRESHOLD = 0.80  # c_strong threshold


def _try_alignscore():
    """Try to import and initialise AlignScore-large. Returns True on success."""
    global _CHECKER, _BACKEND, _CHECKER_NAME
    try:
        from alignscore import AlignScore  # type: ignore
        ckpt_path = None  # AlignScore resolves the large checkpoint automatically
        # Attempt to find a pre-downloaded checkpoint next to the package
        try:
            import alignscore as _as_pkg
            candidate = (
                Path(_as_pkg.__file__).parent / "ckpts" / "AlignScore-large.ckpt"
            )
            if candidate.exists():
                ckpt_path = str(candidate)
        except Exception:
            pass

        _CHECKER = AlignScore(
            model="roberta-large",
            batch_size=16,
            device=_DEVICE,
            ckpt_path=ckpt_path or "AlignScore-large",
        )
        _BACKEND = "alignscore"
        _CHECKER_NAME = "alignscore-large"
        print(f"[Checker] AlignScore-large loaded on {_DEVICE}.")
        return True
    except Exception as e:
        print(f"[Checker] AlignScore unavailable ({type(e).__name__}: {e})")
        return False


def _load_nli_fallback(batch_size: int = 32):
    """Load cross-encoder/nli-deberta-v3-large. Always succeeds if sentence_transformers installed."""
    global _CHECKER, _BACKEND, _CHECKER_NAME, _ENTAIL_IDX
    from sentence_transformers import CrossEncoder  # type: ignore

    model_name = "cross-encoder/nli-deberta-v3-large"
    print(f"[Checker] Loading {model_name} on {_DEVICE}…")
    _CHECKER = CrossEncoder(model_name, device=_DEVICE)
    _CHECKER._batch_size = batch_size

    # Determine entailment label index from model config
    try:
        l2i = _CHECKER.model.config.label2id
        _ENTAIL_IDX = l2i.get("entailment", l2i.get("ENTAILMENT", 2))
    except Exception:
        _ENTAIL_IDX = 2  # MNLI convention: 0=contradiction, 1=neutral, 2=entailment

    _BACKEND = "nli_deberta"
    _CHECKER_NAME = "nli-deberta-v3-large"
    print(f"[Checker] {model_name} ready. Entailment label index: {_ENTAIL_IDX}")


def load_checker(batch_size: int = 32, device: str | None = None) -> None:
    """Attempt AlignScore, fall back to NLI DeBERTa."""
    global _DEVICE
    if device:
        _DEVICE = device
    # Free any cached GPU memory from other scripts before loading
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    if not _try_alignscore():
        _load_nli_fallback(batch_size)


# ── Score computation ─────────────────────────────────────────────

def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def score_pairs(premises: list[str], hypotheses: list[str]) -> list[float]:
    """
    Return a list of alignment scores (floats in [0, 1]) parallel to the inputs.
    Falls back to CPU inference on CUDA OOM.
    """
    if not premises:
        return []

    if _BACKEND == "alignscore":
        return list(_CHECKER.score(contexts=premises, claims=hypotheses))

    # NLI DeBERTa fallback
    pairs = list(zip(premises, hypotheses))
    bs = getattr(_CHECKER, "_batch_size", 32)

    try:
        raw = _CHECKER.predict(pairs, batch_size=bs, convert_to_numpy=True)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        # CUDA OOM — clear cache and re-run on CPU
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        print("  [OOM] CUDA out of memory — switching NLI scorer to CPU for this batch.")
        _CHECKER.model = _CHECKER.model.to("cpu")
        _CHECKER.device = "cpu"
        raw = _CHECKER.predict(pairs, batch_size=bs, convert_to_numpy=True)

    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    probs = _softmax(raw)
    return [float(probs[i, _ENTAIL_IDX]) for i in range(len(probs))]


# ── File helpers ──────────────────────────────────────────────────

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


def load_done_keys(path: Path) -> set[tuple[str, int]]:
    """Return (prompt_id, sample_idx) pairs already written to output."""
    done: set[tuple[str, int]] = set()
    for row in load_jsonl(path):
        pid  = row.get("prompt_id")
        sidx = row.get("sample_idx")
        if pid is not None and sidx is not None:
            done.add((str(pid), int(sidx)))
    return done


def make_output_path(run_id: str, model: str, regime: str, sidx: int) -> Path:
    d = REPO_ROOT / "runs" / run_id / "alignscore"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"alignscore_{model}_{regime}_sample{sidx}.jsonl"


# ── Per-file processing ───────────────────────────────────────────

def process_response_file(
    resp_path: Path,
    run_id: str,
    model: str,
    regime: str,
    sidx: int,
    batch_size: int,
) -> None:
    out_path = make_output_path(run_id, model, regime, sidx)
    done     = load_done_keys(out_path)

    rows = load_jsonl(resp_path)
    # Exclude variant files (criterion_variant_idx != -1) and failed/empty responses
    rows = [
        r for r in rows
        if int(r.get("criterion_variant_idx", -1)) == -1
        and r.get("response", "").strip()
        and not str(r.get("response", "")).startswith("ERROR:")
        and (str(r.get("prompt_id", "")), int(r.get("sample_idx", sidx))) not in done
    ]

    if not rows:
        print(f"  [{model}/{regime}/s{sidx}] Nothing new to score.")
        return

    print(f"  [{model}/{regime}/s{sidx}] Scoring {len(rows)} row(s)…")

    premises    = [str(r.get("user_prompt", ""))   for r in rows]
    hypotheses  = [str(r.get("response", ""))      for r in rows]

    # Batch inference
    scores: list[float] = []
    for start in range(0, len(premises), batch_size):
        batch_p = premises[start : start + batch_size]
        batch_h = hypotheses[start : start + batch_size]
        batch_s = score_pairs(batch_p, batch_h)
        scores.extend(batch_s)
        print(f"    scored {min(start + batch_size, len(premises))}/{len(premises)}", end="\r")
    print()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(out_path, "a", encoding="utf-8") as f:
        for row, score in zip(rows, scores):
            out_row = {
                "prompt_id":      str(row.get("prompt_id", "")),
                "category":       row.get("category", ""),
                "model":          str(row.get("model", model)),
                "regime":         str(row.get("regime", regime)),
                "sample_idx":     int(row.get("sample_idx", sidx)),
                "alignscore_raw": round(float(score), 6),
                "c_strong":       int(float(score) >= _ALIGN_THRESHOLD),
                "checker_model":  _CHECKER_NAME,
                "computed_at":    now,
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    n_strong = sum(1 for s in scores if s >= _ALIGN_THRESHOLD)
    print(
        f"  [{model}/{regime}/s{sidx}] Done. "
        f"mean={np.mean(scores):.3f}, "
        f"c_strong={n_strong}/{len(scores)} "
        f"({100*n_strong/len(scores):.0f}%)"
    )
    print(f"    → {out_path.relative_to(REPO_ROOT)}")


# ── File discovery ────────────────────────────────────────────────

def discover_response_files(
    run_id: str,
    model_filter: str | None,
    regime_filter: str | None,
) -> list[tuple[Path, str, str, int]]:
    """
    Scan runs/{run_id}/responses/ for regime response files.
    Returns list of (path, model, regime, sample_idx) tuples.
    """
    resp_dir = REPO_ROOT / "runs" / run_id / "responses"
    if not resp_dir.exists():
        return []

    results = []
    for p in sorted(resp_dir.glob("*_responses.jsonl")):
        if "_variant_" in p.name:
            continue  # skip criterion-variant files

        stem = p.stem  # e.g. gemma4_e4b_vanilla_sample0_responses → strip
        if stem.endswith("_responses"):
            stem = stem[:-10]  # → gemma4_e4b_vanilla_sample0

        # Extract sample index from suffix _sample{n}
        import re
        m = re.search(r"_sample(\d+)$", stem)
        if not m:
            continue
        sidx = int(m.group(1))
        stem_no_sample = stem[: m.start()]  # → gemma4_e4b_vanilla

        # Split model and regime: regime is one of the known tokens at the end
        # Use row data instead — more robust than filename parsing
        first_rows = load_jsonl(p)
        if not first_rows:
            continue
        model  = str(first_rows[0].get("model",  ""))
        regime = str(first_rows[0].get("regime", ""))
        if not model or not regime:
            continue

        if model_filter  and model  != model_filter:
            continue
        if regime_filter and regime != regime_filter:
            continue

        results.append((p, model, regime, sidx))

    return results


# ── Entry point ───────────────────────────────────────────────────

def main() -> None:
    global _ALIGN_THRESHOLD  # must precede any use of the name in this scope

    parser = argparse.ArgumentParser(
        description="Supplementary alignment/NLI check for C-NB-U benchmark"
    )
    parser.add_argument("--run-id",  required=True,  help="Run identifier")
    parser.add_argument("--model",   default=None,   help="Model key filter")
    parser.add_argument("--regime",  default=None,   help="Regime filter")
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Inference batch size (default: 32)",
    )
    parser.add_argument(
        "--threshold", type=float, default=_ALIGN_THRESHOLD,
        help=f"c_strong threshold (default: {_ALIGN_THRESHOLD})",
    )
    parser.add_argument(
        "--device", default=None,
        help="Force inference device: 'cpu' or 'cuda' (default: auto-detect)",
    )
    args = parser.parse_args()
    _ALIGN_THRESHOLD = args.threshold

    run_path = REPO_ROOT / "runs" / args.run_id
    if not run_path.exists():
        print(f"Error: runs/{args.run_id}/ does not exist.")
        sys.exit(1)

    print(f"Run ID    : {args.run_id}")
    print(f"Device    : {_DEVICE}")
    print(f"Threshold : c_strong = 1 if score >= {_ALIGN_THRESHOLD}")
    print()

    load_checker(args.batch_size, device=args.device)
    print()

    files = discover_response_files(args.run_id, args.model, args.regime)
    if not files:
        print("No regime response files found matching the given filters.")
        sys.exit(0)

    print(f"Files to process: {len(files)}")
    print()

    for resp_path, model, regime, sidx in files:
        process_response_file(
            resp_path=resp_path,
            run_id=args.run_id,
            model=model,
            regime=regime,
            sidx=sidx,
            batch_size=args.batch_size,
        )
    print("\n✔ Alignment scoring complete.")


if __name__ == "__main__":
    main()
