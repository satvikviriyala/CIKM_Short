#!/usr/bin/env python3
"""
LLM-as-Judge evaluation for the CIKM C-NB-U benchmark (Vertex AI MaaS).

Judges ONE response per API call (batch_size=1) to avoid in-batch position
bias and parse-fails-all brittleness.

Input:  runs/{run_id}/responses/{model}_{regime}_sample{n}_responses.jsonl
Output: runs/{run_id}/evaluations/{judge_key}_evaluations_{model}_{regime}_sample{n}.jsonl

Judge configurations are loaded from config/models.json "judges" key.
Judge rubric is loaded from prompts/judge_system_prompt.txt.
Resumption key: (prompt_id, generator_sample_idx).
"""

import os
import sys
import json
import time
import argparse
import subprocess
import urllib.request
import urllib.error
import ssl
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Resolve repo root so the script works regardless of cwd
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from _gen_common import load_model_config, parse_retry_after

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
DEFAULT_JUDGE_RUBRIC = REPO_ROOT / "prompts" / "judge_system_prompt.txt"
DEFAULT_BENCHMARK = REPO_ROOT / "dataset" / "benchmark_prompts.json"


# ── Config and rubric loading ─────────────────────────────────────

def load_judge_rubric(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def load_benchmark_meta(path: Path) -> dict[str, dict]:
    """Return {prompt_id: row_dict} for benchmark metadata lookups."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return {str(r["id"]): r for r in rows if "id" in r}


# ── ADC token ─────────────────────────────────────────────────────

def get_token() -> str | None:
    for cmd in [
        ["gcloud", "auth", "application-default", "print-access-token"],
        ["gcloud", "auth", "print-access-token"],
    ]:
        try:
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, check=True,
            )
            tok = result.stdout.strip()
            if tok:
                return tok
        except Exception:
            pass
    return os.environ.get("GCP_TOKEN") or os.environ.get("GOOGLE_ACCESS_TOKEN")


# ── Vertex AI endpoint URL ────────────────────────────────────────

def _endpoint_url(judge_cfg: dict, project_id: str) -> str:
    region = judge_cfg["location"]
    host = (
        "aiplatform.googleapis.com"
        if region == "global"
        else f"{region}-aiplatform.googleapis.com"
    )
    return (
        f"https://{host}/v1beta1/projects/{project_id}"
        f"/locations/{region}/endpoints/openapi/chat/completions"
    )


# ── Core API call ─────────────────────────────────────────────────

def call_judge_api(
    judge_cfg: dict,
    token: str,
    project_id: str,
    system_prompt: str,
    user_prompt: str,
    *,
    stream: bool = False,
) -> str:
    """
    Call the judge model and return the raw text response.
    Raises urllib.error.HTTPError as-is for Retry-After handling.
    """
    url = _endpoint_url(judge_cfg, project_id)
    payload = {
        "model": judge_cfg["provider_id"],
        "stream": stream,
        "max_tokens": judge_cfg.get("max_tokens", 1024),
        "temperature": judge_cfg.get("temperature", 0.0),
        "top_p": judge_cfg.get("top_p", 0.95),
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ],
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )

    try:
        http_resp = urllib.request.urlopen(req, context=ssl.create_default_context())
    except urllib.error.HTTPError:
        raise
    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error: {e}")
    except Exception as e:
        raise RuntimeError(f"Connection failed: {e}")

    if not stream:
        data = json.loads(http_resp.read())
        choice = data["choices"][0]
        if "message" not in choice:
            # finish_reason=="length" with reasoning model: budget exhausted before output
            raise RuntimeError(
                f"No message in response (finish_reason={choice.get('finish_reason')}). "
                f"Increase max_tokens — model spent all budget on reasoning tokens."
            )
        return choice["message"].get("content") or ""

    # Streaming (test mode)
    parts: list[str] = []
    for line_bytes in http_resp:
        line = line_bytes.decode(errors="ignore").strip()
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            continue
        try:
            delta = json.loads(data_str)["choices"][0]["delta"]
            text = delta.get("content") or delta.get("reasoning_content") or ""
            if text:
                parts.append(text)
                sys.stdout.write(text)
                sys.stdout.flush()
        except Exception:
            pass
    print()
    return "".join(parts)


# ── Prompt construction ───────────────────────────────────────────

def build_judge_user_prompt(row: dict, meta: dict) -> str:
    """
    Build the per-item judge user prompt from a response row and benchmark metadata.
    Uses benchmark meta for source/theta_missing/criterion_variants when available.
    """
    lines: list[str] = []
    lines.append(f"Prompt ID: {row.get('prompt_id', '')}")
    lines.append(f"Category: {row.get('category', '')}")

    source = meta.get("source") or row.get("source", "")
    if source:
        lines.append(f"Source type: {source}")

    theta = meta.get("theta_missing") or row.get("theta_missing", "")
    if theta:
        lines.append(f"Benchmark missing criterion: {theta}")

    variants = meta.get("criterion_variants") or row.get("criterion_variants", [])
    if variants:
        lines.append("Premise-compatible criterion variants:")
        for v in variants:
            lines.append(f"- {v}")

    lines.append("")
    lines.append("User Query:")
    # For regime runs user_prompt == original prompt text
    query = meta.get("prompt") or row.get("user_prompt", "") or row.get("query", "")
    lines.append(query)
    lines.append("")
    lines.append("Assistant Response:")
    lines.append(row.get("response", ""))

    return "\n".join(lines)


# ── Score parsing ─────────────────────────────────────────────────

def clamp_score(value) -> int:
    try:
        v = int(value)
        if 1 <= v <= 5:
            return v
    except Exception:
        pass
    return -1


def normalize_response_type(value: str) -> str:
    rt = str(value or "").strip().lower()
    return rt if rt in {"decisive", "clarifying", "hedging", "refusal"} else "unknown"


def extract_single_score(text: str) -> dict | None:
    """
    Parse a single JSON object from judge output.
    Tolerates markdown fences and falls back to single-element arrays
    for backward compatibility with older rubric versions.
    """
    t = text.strip()
    if "```json" in t:
        t = t.split("```json")[1].split("```")[0].strip()
    elif "```" in t:
        t = t.split("```")[1].split("```")[0].strip()
    try:
        data = json.loads(t)
        if isinstance(data, list):
            data = data[0] if data else None
        if isinstance(data, dict):
            return data
    except Exception as e:
        print(f"  [Parse error] {e} — snippet: {t[:200]}")
    return None


# ── Per-row worker ─────────────────────────────────────────────────

def evaluate_single_worker(
    judge_key: str,
    judge_cfg: dict,
    token: str,
    project_id: str,
    rubric: str,
    row: dict,
    benchmark_meta: dict,
) -> dict:
    prompt_id = str(row.get("prompt_id", ""))
    meta = benchmark_meta.get(prompt_id, {})
    user_prompt = build_judge_user_prompt(row, meta)

    response_text = ""
    score_obj: dict | None = None

    for attempt in range(3):
        try:
            response_text = call_judge_api(
                judge_cfg, token, project_id, rubric, user_prompt
            )
            score_obj = extract_single_score(response_text)
            if score_obj is not None:
                break
            print(f"  [{prompt_id}] [Attempt {attempt+1}] Parse failed — retrying")
            time.sleep(1)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode(errors="ignore")[:120]
            wait = (parse_retry_after(e) or 60) if e.code == 429 else 2 ** (attempt + 1)
            print(
                f"  [{prompt_id}] [Attempt {attempt+1}] HTTP {e.code}: "
                f"{err_body} — waiting {wait:.0f}s"
            )
            if attempt < 2:
                time.sleep(wait)
        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"  [{prompt_id}] [Attempt {attempt+1}] Error: {e} — waiting {wait:.0f}s")
            if attempt < 2:
                time.sleep(wait)

    if score_obj is None:
        score_obj = {}

    return {
        "prompt_id": prompt_id,
        "category": row.get("category", ""),
        "generator_model": row.get("model", ""),
        "generator_regime": row.get("regime", ""),
        "generator_sample_idx": int(row.get("sample_idx", 0)),
        "judge_key": judge_key,
        "judge_provider_id": judge_cfg["provider_id"],
        "utility_score": clamp_score(score_obj.get("utility_score")),
        "satisfaction_score": clamp_score(score_obj.get("satisfaction_score")),
        "response_type": normalize_response_type(score_obj.get("response_type")),
        "raw_judge_text_excerpt": response_text[:500],
        "evaluated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ── Resumption ────────────────────────────────────────────────────

def load_done_eval_keys(output_file: Path) -> set[tuple[str, int]]:
    """Return set of (prompt_id, generator_sample_idx) already evaluated."""
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
                sidx = row.get("generator_sample_idx")
                if pid is not None and sidx is not None:
                    done.add((str(pid), int(sidx)))
            except json.JSONDecodeError:
                pass
    return done


# ── Output path construction ──────────────────────────────────────

def make_eval_output_path(run_id: str, judge_key: str, input_path: Path) -> Path:
    """
    Derive output path from the input filename.
    Input stem:  {model}_{regime}_sample{n}_responses
    Output:      runs/{run_id}/evaluations/{judge_key}_evaluations_{model}_{regime}_sample{n}.jsonl
    """
    stem = input_path.stem  # e.g. gemma4_e4b_vanilla_sample0_responses
    if stem.endswith("_responses"):
        stem = stem[:-10]  # → gemma4_e4b_vanilla_sample0
    out_dir = REPO_ROOT / "runs" / run_id / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{judge_key}_evaluations_{stem}.jsonl"


# ── Test mode ─────────────────────────────────────────────────────

def run_test(judge_key: str, judge_cfg: dict, token: str, project_id: str, rubric: str) -> None:
    print(f"\n=== Judge Diagnostic Test ===")
    print(f"  Judge key    : {judge_key}")
    print(f"  Provider ID  : {judge_cfg['provider_id']}")
    print(f"  Location     : {judge_cfg['location']}")
    print(f"  Temperature  : {judge_cfg.get('temperature', 0.0)}")
    print("-" * 60)

    test_user_prompt = (
        "Prompt ID: cat1_orig_001\n"
        "Category: 1\n"
        "Source type: original\n"
        "Benchmark missing criterion: The relative weighting of academic merit versus financial need.\n"
        "Premise-compatible criterion variants:\n"
        "- Assume academic merit is the strictly most important factor.\n"
        "- Assume financial need is the strictly most important factor.\n\n"
        "User Query:\n"
        "Student A has a GPA of 9.5 and moderate financial need. "
        "Student B has a GPA of 8.7 and severe financial need. "
        "Who should receive the single available scholarship?\n\n"
        "Assistant Response:\n"
        "Based on a balanced consideration of both academic excellence and financial circumstances, "
        "Student A's higher GPA suggests stronger academic merit, while Student B's severe need "
        "represents a stronger financial case. Without knowing the scholarship's stated priority, "
        "either could be justified."
    )

    try:
        # Use at least 2048 tokens: gemini-3.5-flash spends ~150+ on reasoning first
        test_cfg = dict(judge_cfg, max_tokens=max(judge_cfg.get("max_tokens", 1024), 2048))
        response_text = call_judge_api(
            test_cfg, token, project_id, rubric, test_user_prompt, stream=False
        )
        print(f"Raw response:\n{response_text}\n")
        print(f"--- Parsed scores ---")
        parsed = extract_single_score(response_text)
        if parsed:
            print(json.dumps(parsed, indent=2))
            print("✔ Parse successful.")
        else:
            print("✘ JSON parse failed on the above output.")
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")
        print(f"✘ HTTP {e.code}: {body[:400]}")
        _suggest_alternatives(judge_cfg["provider_id"])
    except Exception as e:
        print(f"✘ Diagnostic failed: {e}")


def _suggest_alternatives(failed_id: str) -> None:
    alternates = {
        "google/gemini-3.1-pro": [
            "google/gemini-3.1-pro-preview",
            "google/gemini-3.5-flash",
        ],
        "google/gemini-3.5-flash": ["google/gemini-3.1-pro-preview"],
        "deepseek-ai/deepseek-v3.2-maas": [
            "deepseek-ai/deepseek-v3-maas",
        ],
        "moonshotai/kimi-k2-thinking-maas": [
            "moonshotai/kimi-k2-maas",
        ],
    }
    alts = alternates.get(failed_id, [])
    if alts:
        print(f"  [Hint] Try these alternative provider IDs in config/models.json:")
        for a in alts:
            print(f"    {a}")


# ── Batch evaluation ──────────────────────────────────────────────

def run_batch_evaluation(
    judge_key: str,
    judge_cfg: dict,
    token: str,
    project_id: str,
    rubric: str,
    input_path: Path,
    run_id: str,
    max_rows: int,
    concurrency: int,
    benchmark_meta: dict,
) -> None:
    output_file = make_eval_output_path(run_id, judge_key, input_path)

    # Load input rows
    rows: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if max_rows:
        rows = rows[:max_rows]

    print(f"\n{'='*60}")
    print(f"  Judge:  {judge_key} ({judge_cfg['provider_id']})")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_file}")
    print(f"{'='*60}")

    done = load_done_eval_keys(output_file)
    if done:
        print(f"  Resuming: {len(done)} (prompt_id, sample_idx) pairs already done.")

    remaining = [
        r for r in rows
        if (str(r.get("prompt_id", "")), int(r.get("sample_idx", 0))) not in done
    ]
    if not remaining:
        print("  ✔ Nothing left to evaluate.")
        return

    print(f"  Remaining: {len(remaining)} rows")
    start = time.time()
    completed = 0

    with open(output_file, "a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    evaluate_single_worker,
                    judge_key, judge_cfg, token, project_id,
                    rubric, row, benchmark_meta,
                ): row.get("prompt_id", "?")
                for row in remaining
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out.flush()
                    completed += 1
                    elapsed = time.time() - start
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    eta = (len(remaining) - completed) / rate if rate > 0 else 0.0
                    print(
                        f"  Progress: {completed}/{len(remaining)} | "
                        f"Rate: {rate:.2f}/s | ETA: {eta/60:.1f}m"
                    )
                except Exception as e:
                    print(f"  [Error] Future raised: {e}")

    elapsed = time.time() - start
    rate = completed / elapsed if elapsed > 0 else 0.0
    print(f"\n✔ Done. {completed} evaluations in {elapsed:.1f}s ({rate:.2f}/s)")


# ── Entry point ───────────────────────────────────────────────────

def main() -> None:
    cfg = load_model_config()
    judges = cfg.get("judges", {})
    if not judges:
        print("Error: no 'judges' entries found in config/models.json.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="LLM-as-Judge evaluation for C-NB-U benchmark (Vertex AI MaaS)"
    )
    parser.add_argument(
        "--judge-key", required=True, choices=sorted(judges),
        help="Judge key from config/models.json",
    )
    parser.add_argument(
        "--run-id", required=True,
        help="Run identifier (used to locate input and write output)",
    )
    parser.add_argument(
        "--input", type=Path, required=False, default=None,
        help=(
            "Path to regime response JSONL "
            "(default: runs/{run_id}/responses/ — auto-detect first available file)"
        ),
    )
    parser.add_argument(
        "--project-id", default=PROJECT_ID,
        help="GCP project ID (or set GOOGLE_CLOUD_PROJECT)",
    )
    parser.add_argument(
        "--judge-rubric", type=Path, default=DEFAULT_JUDGE_RUBRIC,
        help="Judge rubric file path",
    )
    parser.add_argument(
        "--benchmark", type=Path, default=DEFAULT_BENCHMARK,
        help="Benchmark prompts JSON for metadata enrichment",
    )
    parser.add_argument(
        "--max-rows", type=int, default=0,
        help="Limit rows processed (0 = all)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=2,
        help="Worker thread count",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run a single-item streaming diagnostic and exit",
    )
    args = parser.parse_args()

    if not args.project_id:
        print("Error: --project-id or GOOGLE_CLOUD_PROJECT required.")
        sys.exit(1)

    token = get_token()
    if not token:
        print("Error: Could not get GCP token. Run: gcloud auth application-default login")
        sys.exit(1)

    rubric = load_judge_rubric(args.judge_rubric)
    judge_cfg = judges[args.judge_key]

    if args.test:
        run_test(args.judge_key, judge_cfg, token, args.project_id, rubric)
        return

    # Resolve input file
    input_path = args.input
    if input_path is None:
        resp_dir = REPO_ROOT / "runs" / args.run_id / "responses"
        candidates = sorted(resp_dir.glob("*_responses.jsonl"))
        # Prefer vanilla regime files; skip variant files
        regime_files = [p for p in candidates if "_variant_" not in p.name]
        if not regime_files:
            print(f"Error: no regime response files found in {resp_dir}")
            sys.exit(1)
        input_path = regime_files[0]
        print(f"[Auto-selected input] {input_path}")

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    benchmark_meta = load_benchmark_meta(args.benchmark) if args.benchmark.exists() else {}

    run_batch_evaluation(
        judge_key=args.judge_key,
        judge_cfg=judge_cfg,
        token=token,
        project_id=args.project_id,
        rubric=rubric,
        input_path=input_path,
        run_id=args.run_id,
        max_rows=args.max_rows,
        concurrency=args.concurrency,
        benchmark_meta=benchmark_meta,
    )


if __name__ == "__main__":
    main()
