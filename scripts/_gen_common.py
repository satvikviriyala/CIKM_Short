"""
Shared utilities for CIKM C-NB-U generation scripts.

Consumed by generate_responses.py (cloud) and generate_responses_local.py (local).
"""

import json
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = REPO_ROOT / "config" / "models.json"
DEFAULT_INPUT = REPO_ROOT / "dataset" / "benchmark_prompts.json"
DEFAULT_PROMPT_FILE = REPO_ROOT / "prompts" / "system_prompts.json"


# ── Config loading ───────────────────────────────────────────────

def load_model_config() -> dict:
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return json.load(f)


def get_generation_models(cfg: dict, *, local_only: bool) -> dict:
    """Filter the generation block to local or cloud models."""
    gen = cfg.get("generation", {})
    if local_only:
        return {k: v for k, v in gen.items() if v.get("location") == "local"}
    return {k: v for k, v in gen.items() if v.get("location") != "local"}


def load_regimes(prompt_file: Path) -> tuple[str, dict[str, str]]:
    """Return (version_string, {regime_key: system_prompt_text})."""
    with open(prompt_file, encoding="utf-8") as f:
        data = json.load(f)
    regimes_raw = data.get("regimes", data)
    version = data.get("version", prompt_file.stem)
    regimes = {
        k: v["system_prompt"] if isinstance(v, dict) else v
        for k, v in regimes_raw.items()
    }
    return version, regimes


def load_prompts(input_file: Path, max_rows: int = 0) -> list[dict]:
    with open(input_file, encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts[:max_rows] if max_rows else prompts


# ── Output path & resumption ─────────────────────────────────────

def make_output_path(run_id: str, model_key: str, regime_key: str, sample_idx: int) -> Path:
    """Build and mkdir the canonical output path for one (model, regime, sample) run."""
    p = (
        REPO_ROOT
        / "runs" / run_id / "responses"
        / f"{model_key}_{regime_key}_sample{sample_idx}_responses.jsonl"
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def make_variant_output_path(
    run_id: str, model_key: str, variant_idx: int, sample_idx: int
) -> Path:
    """Build and mkdir the output path for one criterion-variant (model, variant, sample) run."""
    p = (
        REPO_ROOT
        / "runs" / run_id / "responses"
        / f"{model_key}_variant_{variant_idx}_sample{sample_idx}_responses.jsonl"
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_done_keys(output_file: Path) -> set[tuple[str, int]]:
    """Return the set of (prompt_id, sample_idx) pairs already written to output_file."""
    done: set[tuple[str, int]] = set()
    if not output_file.exists():
        return done
    with open(output_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                pid = row.get("prompt_id")
                sidx = row.get("sample_idx")
                if pid is not None and sidx is not None:
                    done.add((str(pid), int(sidx)))
            except json.JSONDecodeError:
                pass
    return done


def load_done_variant_keys(output_file: Path) -> set[tuple[str, int, int]]:
    """Return the set of (prompt_id, variant_idx, sample_idx) pairs already written."""
    done: set[tuple[str, int, int]] = set()
    if not output_file.exists():
        return done
    with open(output_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                pid = row.get("prompt_id")
                vidx = row.get("criterion_variant_idx")
                sidx = row.get("sample_idx")
                if pid is not None and vidx is not None and sidx is not None:
                    done.add((str(pid), int(vidx), int(sidx)))
            except json.JSONDecodeError:
                pass
    return done


# ── Canonical output row ─────────────────────────────────────────

def build_output_row(
    *,
    prompt_id: str,
    category,
    model_key: str,
    model_cfg: dict,
    regime_key: str,
    system_prompt_version: str,
    system_prompt: str,
    user_prompt: str,
    sample_idx: int,
    run_id: str,
    response: str,
    reasoning_content: str,
    raw_provider_response_excerpt: str,
    criterion_variant_idx: int = -1,
    criterion_variant_text: str = "",
) -> dict:
    return {
        "prompt_id": prompt_id,
        "category": category,
        "model": model_key,
        "model_id": model_cfg["provider_id"],
        "region": model_cfg["location"],
        "regime": regime_key,
        "criterion_variant_idx": criterion_variant_idx,
        "criterion_variant_text": criterion_variant_text,
        "system_prompt_version": system_prompt_version,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "inference_settings": {
            "temperature": model_cfg["temperature"],
            "top_p": model_cfg["top_p"],
            "max_tokens": model_cfg["max_tokens"],
        },
        "sample_idx": sample_idx,
        "run_id": run_id,
        "response": response,
        "reasoning_content": reasoning_content,
        "raw_provider_response_excerpt": raw_provider_response_excerpt,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ── Retry helpers ────────────────────────────────────────────────

def parse_retry_after(exc: urllib.error.HTTPError) -> float | None:
    """Extract Retry-After seconds from a 429 HTTPError header, or None."""
    try:
        val = exc.headers.get("Retry-After")
        if val is not None:
            return float(val)
    except Exception:
        pass
    return None


# ── Progress reporting ───────────────────────────────────────────

def print_progress(completed: int, total: int, start_time: float) -> None:
    elapsed = time.time() - start_time
    rate = completed / elapsed if elapsed > 0 else 0.0
    eta = (total - completed) / rate if rate > 0 else 0.0
    print(
        f"  Progress: {completed}/{total} | "
        f"Rate: {rate:.2f}/s | ETA: {eta / 60:.1f}m"
    )
