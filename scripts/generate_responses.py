#!/usr/bin/env python3
"""
Cloud response generation for CIKM C-NB-U benchmark (Vertex AI MaaS).

Runs benchmark prompts through one cloud generation model, under one prompting
regime, for one sample index.  Writes to:
  runs/{run_id}/responses/{model_key}_{regime_key}_sample{n}_responses.jsonl

Resumption is keyed on (prompt_id, sample_idx), not positional index.
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
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gen_common import (
    load_model_config, get_generation_models, load_regimes,
    load_prompts, make_output_path, load_done_keys,
    build_output_row, parse_retry_after, print_progress,
    DEFAULT_INPUT, DEFAULT_PROMPT_FILE,
)

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")


# ── GCP auth ─────────────────────────────────────────────────────

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


# ── HTTP helpers ─────────────────────────────────────────────────

def _endpoint_url(model_cfg: dict, project_id: str) -> str:
    region = model_cfg["location"]
    # global uses the root hostname; every named region prefixes its own subdomain
    host = (
        "aiplatform.googleapis.com"
        if region == "global"
        else f"{region}-aiplatform.googleapis.com"
    )
    return (
        f"https://{host}/v1beta1/projects/{project_id}"
        f"/locations/{region}/endpoints/openapi/chat/completions"
    )


def call_cloud_chat(
    model_cfg: dict,
    token: str,
    project_id: str,
    system_prompt: str,
    user_prompt: str,
    *,
    stream: bool = False,
) -> tuple[str, str, str]:
    """
    Send one chat completion request.

    Returns (response_text, reasoning_content, raw_excerpt_500).
    - reasoning_content is non-empty only for reasoning models.
    - raw_excerpt_500 is the first 500 chars of the provider JSON; empty for streaming.
    - Raises urllib.error.HTTPError as-is so callers can inspect Retry-After.
    - Wraps non-HTTP errors in RuntimeError.
    """
    url = _endpoint_url(model_cfg, project_id)
    payload = {
        "model": model_cfg["provider_id"],
        "stream": stream,
        "max_tokens": model_cfg["max_tokens"],
        "temperature": model_cfg["temperature"],
        "top_p": model_cfg["top_p"],
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
        raise  # caller handles Retry-After
    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error: {e}")
    except Exception as e:
        raise RuntimeError(f"Connection failed: {e}")

    if not stream:
        raw_bytes = http_resp.read()
        raw_excerpt = raw_bytes.decode(errors="replace")[:500]
        data = json.loads(raw_bytes)
        msg = data["choices"][0]["message"]
        response_text = msg.get("content") or ""
        reasoning_content = (
            msg.get("reasoning_content")
            or msg.get("thinking")
            or ""
        )
        return response_text, reasoning_content, raw_excerpt

    # Streaming path (test mode) — print as it arrives, accumulate separately
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    for line_bytes in http_resp:
        line = line_bytes.decode(errors="ignore").strip()
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            continue
        try:
            delta = json.loads(data_str)["choices"][0]["delta"]
            rc = delta.get("reasoning_content") or delta.get("thinking") or ""
            ct = delta.get("content") or ""
            if rc:
                reasoning_parts.append(rc)
                sys.stdout.write(rc)
                sys.stdout.flush()
            if ct:
                content_parts.append(ct)
                sys.stdout.write(ct)
                sys.stdout.flush()
        except Exception:
            pass
    print()
    return "".join(content_parts), "".join(reasoning_parts), ""


# ── Batch worker ─────────────────────────────────────────────────

def generate_row_worker(
    model_key: str,
    model_cfg: dict,
    token: str,
    project_id: str,
    regime_key: str,
    prompt_version: str,
    system_prompt: str,
    row: dict,
    sample_idx: int,
    run_id: str,
) -> dict:
    prompt_id = str(row.get("id") or row.get("prompt_id") or "unknown")
    user_prompt = row.get("prompt") or row.get("query") or ""

    response_text = ""
    reasoning_content = ""
    raw_excerpt = ""

    for attempt in range(3):
        try:
            response_text, reasoning_content, raw_excerpt = call_cloud_chat(
                model_cfg, token, project_id, system_prompt, user_prompt
            )
            if response_text.strip():
                break
            print(f"  [{prompt_id}] [Attempt {attempt+1}] Empty response — retrying")
        except urllib.error.HTTPError as e:
            err_snippet = e.read().decode(errors="ignore")[:120]
            wait = (parse_retry_after(e) or 60) if e.code == 429 else 2 ** (attempt + 1)
            print(
                f"  [{prompt_id}] [Attempt {attempt+1}] HTTP {e.code}: "
                f"{err_snippet} — waiting {wait:.0f}s"
            )
            if attempt < 2:
                time.sleep(wait)
        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"  [{prompt_id}] [Attempt {attempt+1}] Error: {e} — waiting {wait:.0f}s")
            if attempt < 2:
                time.sleep(wait)

    if not response_text.strip():
        response_text = "ERROR: generation failed after 3 attempts."

    return build_output_row(
        prompt_id=prompt_id,
        category=row.get("category", ""),
        model_key=model_key,
        model_cfg=model_cfg,
        regime_key=regime_key,
        system_prompt_version=prompt_version,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        sample_idx=sample_idx,
        run_id=run_id,
        response=response_text,
        reasoning_content=reasoning_content,
        raw_provider_response_excerpt=raw_excerpt,
    )


# ── Test mode ─────────────────────────────────────────────────────

def run_test(
    model_key: str,
    model_cfg: dict,
    token: str,
    project_id: str,
    regime_key: str,
    regimes: dict,
) -> None:
    system_prompt = regimes[regime_key]
    user_prompt = (
        "Should phone Y (battery lasts 3 days, mediocre camera) or "
        "phone X (great camera, charge twice a day) be bought?"
    )
    # Use a reduced max_tokens for the diagnostic so it finishes quickly
    test_cfg = dict(model_cfg, max_tokens=min(256, model_cfg["max_tokens"]))

    print(f"\n=== Cloud Diagnostic Test ===")
    print(f"  Model key : {model_key}")
    print(f"  Provider  : {model_cfg['provider_id']}")
    print(f"  Region    : {model_cfg['location']}")
    print(f"  Regime    : {regime_key}")
    print(f"  Temp={model_cfg['temperature']}  top_p={model_cfg['top_p']}  "
          f"max_tokens={test_cfg['max_tokens']} (capped for test)")
    print(f"  is_reasoning: {model_cfg.get('is_reasoning', False)}")
    print("-" * 60)
    try:
        content, reasoning, _ = call_cloud_chat(
            test_cfg, token, project_id, system_prompt, user_prompt, stream=True
        )
        if reasoning:
            print(f"\n[reasoning_content captured — {len(reasoning)} chars]")
        else:
            print("\n[reasoning_content: empty]")
        print("✔ Diagnostic completed.")
    except urllib.error.HTTPError as e:
        print(f"✘ HTTP {e.code}: {e.read().decode(errors='ignore')[:300]}")
    except Exception as e:
        print(f"✘ Diagnostic failed: {e}")


# ── Batch mode ────────────────────────────────────────────────────

def run_batch(
    model_key: str,
    model_cfg: dict,
    token: str,
    project_id: str,
    regime_key: str,
    prompt_version: str,
    regimes: dict,
    input_file: Path,
    max_rows: int,
    concurrency: int,
    sample_idx: int,
    run_id: str,
) -> None:
    system_prompt = regimes[regime_key]
    output_file = make_output_path(run_id, model_key, regime_key, sample_idx)
    prompts = load_prompts(input_file, max_rows)

    print(f"\n{'='*60}")
    print(f"  Model:   {model_key} ({model_cfg['provider_id']})")
    print(f"  Regime:  {regime_key}  |  Sample: {sample_idx}  |  Run: {run_id}")
    print(f"  Output:  {output_file}")
    print(f"{'='*60}")

    done = load_done_keys(output_file)
    if done:
        print(f"  Resuming: {len(done)} (prompt_id, sample_idx) pairs already written.")

    remaining = [
        row for row in prompts
        if (str(row.get("id") or "unknown"), sample_idx) not in done
    ]
    if not remaining:
        print("  ✔ Nothing left to generate for this (model, regime, sample).")
        return

    print(f"  Remaining: {len(remaining)} prompts")
    start = time.time()
    completed = 0

    with open(output_file, "a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    generate_row_worker,
                    model_key, model_cfg, token, project_id,
                    regime_key, prompt_version, system_prompt,
                    row, sample_idx, run_id,
                ): row.get("id", "?")
                for row in remaining
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out.flush()
                    completed += 1
                    print_progress(completed, len(remaining), start)
                except Exception as e:
                    print(f"  [Error] Future raised: {e}")

    elapsed = time.time() - start
    rate = completed / elapsed if elapsed > 0 else 0.0
    print(f"\n✔ Done. {completed} responses in {elapsed:.1f}s ({rate:.2f}/s)")


# ── Entry point ───────────────────────────────────────────────────

def main() -> None:
    cfg = load_model_config()
    cloud_models = get_generation_models(cfg, local_only=False)
    prompt_version, regimes = load_regimes(DEFAULT_PROMPT_FILE)

    parser = argparse.ArgumentParser(
        description="Cloud benchmark response generation (Vertex AI MaaS)"
    )
    parser.add_argument(
        "--model", required=True, choices=sorted(cloud_models),
        help="Model key from config/models.json (cloud entries only)",
    )
    parser.add_argument(
        "--regime", required=True, choices=sorted(regimes),
        help="Prompting regime",
    )
    parser.add_argument(
        "--sample-idx", type=int, default=None,
        help="Sample index 0–2 (required for batch mode)",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Run identifier, e.g. smoke or 2026-05-21 (required for batch mode)",
    )
    parser.add_argument(
        "--project-id", default=PROJECT_ID,
        help="GCP project ID (or set GOOGLE_CLOUD_PROJECT env var)",
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Benchmark prompts JSON",
    )
    parser.add_argument(
        "--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE,
        help="System prompts JSON",
    )
    parser.add_argument(
        "--max-rows", type=int, default=0,
        help="Limit number of prompts processed (0 = all)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=2,
        help="Worker thread count",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run a short streaming diagnostic and exit",
    )
    args = parser.parse_args()

    # Reload regimes if a custom prompt file was given
    if args.prompt_file != DEFAULT_PROMPT_FILE:
        prompt_version, regimes = load_regimes(args.prompt_file)

    if not args.project_id:
        print("Error: --project-id or GOOGLE_CLOUD_PROJECT required.")
        sys.exit(1)

    token = get_token()
    if not token:
        print("Error: Could not get GCP token. Run: gcloud auth application-default login")
        sys.exit(1)

    model_cfg = cloud_models[args.model]

    if args.test:
        run_test(args.model, model_cfg, token, args.project_id, args.regime, regimes)
        return

    # Batch mode: sample-idx and run-id are required
    if args.sample_idx is None:
        parser.error("--sample-idx is required for batch mode")
    if args.run_id is None:
        parser.error("--run-id is required for batch mode")

    run_batch(
        model_key=args.model,
        model_cfg=model_cfg,
        token=token,
        project_id=args.project_id,
        regime_key=args.regime,
        prompt_version=prompt_version,
        regimes=regimes,
        input_file=args.input,
        max_rows=args.max_rows,
        concurrency=args.concurrency,
        sample_idx=args.sample_idx,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
