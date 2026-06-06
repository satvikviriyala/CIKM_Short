#!/usr/bin/env python3
"""
GCP Model Response Generation Script (Benchmark Evaluator)
-----------------------------------------------------------
Queries CIKM generation baseline models to produce responses for 140 benchmark
prompts under 4 distinct prompting regimes (vanilla, utility_first,
neutrality_oriented, clarification_first).

Supports:
1. **Interactive/Diagnostic Test Bed Mode** (`--test`) - Live SSE stream tests.
2. **Batch Response Generation Mode** (Default) - Highly concurrent multithreaded
   generation, automatic resumption, and progressive JSONL writing.
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

# ── Configuration & Generation Models Registry ──────────────────
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
DEFAULT_INPUT = "dataset/benchmark_prompts.json"
DEFAULT_PROMPT_FILE = "prompts/system_prompts.json"

MODELS = {
    "grok": {"id": "xai/grok-4.20-reasoning", "region": "global", "top_p": 0.95},
    "qwen3": {"id": "qwen/qwen3-235b-a22b-instruct-2507-maas", "region": "us-south1", "top_p": 0.8},
    "gpt-oss": {"id": "openai/gpt-oss-120b-maas", "region": "global", "top_p": 0.95},
    "llama3": {"id": "meta/llama-3.3-70b-instruct-maas", "region": "global", "top_p": 0.95}
}

# ── Prompting Regimes Specifications ────────────────────────────
def load_regimes(prompt_file: str) -> tuple[str, dict[str, str]]:
    """Load versioned generation system prompts from disk."""
    with open(prompt_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    regimes = data.get("regimes", data)
    return data.get("version", Path(prompt_file).stem), {
        key: value["system_prompt"] if isinstance(value, dict) else value
        for key, value in regimes.items()
    }

PROMPT_VERSION, REGIMES = load_regimes(DEFAULT_PROMPT_FILE)

# ── Helper Functions ────────────────────────────────────────────
def get_token():
    """Retrieve active OAuth access token from ADC, then user credentials as fallback."""
    for cmd in [["gcloud", "auth", "application-default", "print-access-token"], ["gcloud", "auth", "print-access-token"]]:
        try:
            return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True).stdout.strip()
        except:
            pass
    return os.environ.get("GCP_TOKEN") or os.environ.get("GOOGLE_ACCESS_TOKEN")

# ── Core API Execution Engine ───────────────────────────────────
def call_openapi_chat(model_key: str, token: str, project_id: str, system_prompt: str, user_prompt: str, stream=False, max_tokens=1024):
    """Core HTTP Executor making calls directly to the unified GCP OpenAPI endpoint."""
    cfg = MODELS[model_key]
    endpoint = "us-south1-aiplatform.googleapis.com" if cfg["region"] == "us-south1" else "aiplatform.googleapis.com"
    url = f"https://{endpoint}/v1beta1/projects/{project_id}/locations/{cfg['region']}/endpoints/openapi/chat/completions"
    
    payload = {
        "model": cfg["id"],
        "stream": stream,
        "max_tokens": max_tokens,
        "temperature": 1.0,  # Standard decoding temperature
        "top_p": cfg.get("top_p", 0.95),
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        ]
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST"
    )
    
    try:
        response = urllib.request.urlopen(req, context=ssl.create_default_context())
        if not stream:
            res_data = json.loads(response.read().decode())
            return res_data["choices"][0]["message"]["content"]
            
        # Stream parser for Test Mode
        full_text = []
        for line_bytes in response:
            line = line_bytes.decode(errors="ignore").strip()
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    continue
                try:
                    delta = json.loads(data_str)["choices"][0]["delta"]
                    text = delta.get("content") or delta.get("reasoning_content") or delta.get("thinking") or ""
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    full_text.append(text)
                except:
                    pass
        print()
        return "".join(full_text)
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode(errors="ignore")
        raise RuntimeError(f"HTTP ERROR {e.code}: {err_msg}")
    except Exception as e:
        raise RuntimeError(f"API Connection Failed: {e}")

# ── Operational Mode 1: Test Bed Mode ───────────────────────────
def run_diagnostic_test(model_key: str, regime_key: str, token: str, project_id: str, regimes: dict[str, str]):
    """Executes a diagnostic stream request to verify connectivity and parse capability."""
    system_prompt = regimes[regime_key]
    print(f"\n=== Running Diagnostic Generation Test ===")
    print(f"  Model:  {model_key}")
    print(f"  Regime: {regime_key}")
    print(f"  System Prompt: \"{system_prompt}\"")
    prompt = "Should phone Y (battery lasts 3 days, mediocre camera) or phone X (great camera, charge twice a day) be bought?"
    print("-" * 58)
    try:
        call_openapi_chat(model_key, token, project_id, system_prompt, prompt, stream=True, max_tokens=256)
        print("✔ Diagnostic successfully completed.")
    except Exception as e:
        print(f"✘ Diagnostic Failed: {e}")

# ── Operational Mode 2: Batch Generation Engine ─────────────────
def generate_row_worker(
    model_key: str,
    token: str,
    project_id: str,
    regime_key: str,
    prompt_version: str,
    system_prompt: str,
    row: dict,
    index: int
) -> dict:
    """Single worker thread generating a response for a benchmark prompt."""
    prompt = row.get("prompt") or row.get("query")
    cfg = MODELS[model_key]
    
    for attempt in range(3):
        try:
            response_text = call_openapi_chat(model_key, token, project_id, system_prompt, prompt, max_tokens=1024)
            if response_text and response_text.strip():
                return {
                    "index": index,
                    "prompt_id": row.get("id") or str(index),
                    "query": prompt,
                    "category": row.get("category", ""),
                    "context": str(row.get("category", "")),
                    "source": row.get("source", ""),
                    "source_notes": row.get("source_notes", ""),
                    "theta_missing": row.get("theta_missing", ""),
                    "criterion_variants": row.get("criterion_variants", []),
                    "model": model_key,
                    "model_id": cfg["id"],
                    "region": cfg["region"],
                    "regime": regime_key,
                    "system_prompt_version": prompt_version,
                    "system_prompt": system_prompt,
                    "response": response_text,
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
        except Exception as e:
            print(f"  [Row {index}] [Attempt {attempt+1}] Request failed: {e}")
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                
    # Fallback response on failure
    print(f"  [Row {index}] ✘ FAILED entirely. Returning fallback response.")
    return {
        "index": index,
        "prompt_id": row.get("id") or str(index),
        "query": prompt,
        "category": row.get("category", ""),
        "context": str(row.get("category", "")),
        "source": row.get("source", ""),
        "source_notes": row.get("source_notes", ""),
        "theta_missing": row.get("theta_missing", ""),
        "criterion_variants": row.get("criterion_variants", []),
        "model": model_key,
        "model_id": cfg["id"],
        "region": cfg["region"],
        "regime": regime_key,
        "system_prompt_version": prompt_version,
        "system_prompt": system_prompt,
        "response": "ERROR: Model response generation failed.",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

def run_batch_generation(
    model_key: str,
    regime_key: str,
    token: str,
    project_id: str,
    input_file: str,
    max_rows: int,
    concurrency: int,
    prompt_version: str,
    regimes: dict[str, str]
):
    """Coordinates concurrent response generation and records progress incrementally."""
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_file} not found.")
        sys.exit(1)
        
    output_file = Path(f"{model_key}_{regime_key}_responses.jsonl")
    system_prompt = regimes[regime_key]
    
    print(f"\n============================================================")
    print(f"  Generating Responses: {model_key} | Regime: {regime_key}")
    print(f"  Input File:           {input_file}")
    print(f"  Output File:          {output_file}")
    print(f"============================================================")
    
    # 1. Read input prompts
    with open(input_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if max_rows:
        prompts = prompts[:max_rows]
        
    # 2. Check for resumption progress
    done_indices = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_indices.add(json.loads(line).get("index"))
        print(f"  Resuming: Found {len(done_indices)} already generated responses.")
        
    # Standardize row indexes sequentially if not present
    remaining_rows = []
    for idx, row in enumerate(prompts, 1):
        row_idx = row.get("index") or idx
        if row_idx not in done_indices:
            remaining_rows.append((row, row_idx))
            
    if not remaining_rows:
        print("  ✔ All responses have already been generated!")
        return

    print(f"  Total prompts remaining: {len(remaining_rows)}")
    
    # 3. Multithreaded Execution via ThreadPoolExecutor
    start_time = time.time()
    completed_count = 0
    
    with open(output_file, "a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    generate_row_worker,
                    model_key,
                    token,
                    project_id,
                    regime_key,
                    prompt_version,
                    system_prompt,
                    row,
                    row_idx
                ): row_idx
                for row, row_idx in remaining_rows
            }
            
            for future in as_completed(futures):
                row_idx = futures[future]
                try:
                    res = future.result()
                    out.write(json.dumps(res, ensure_ascii=False) + "\n")
                    
                    completed_count += 1
                    elapsed = time.time() - start_time
                    rate = completed_count / elapsed if elapsed > 0 else 0
                    eta = (len(remaining_rows) - completed_count) / rate if rate > 0 else 0
                    
                    print(f"  Progress: {completed_count}/{len(remaining_rows)} prompts | "
                          f"Rate: {rate:.2f} prompts/s | ETA: {eta/60:.1f}m")
                except Exception as e:
                    print(f"  [Error] Prompt {row_idx} future threw an exception: {e}")
                    
    total_elapsed = time.time() - start_time
    print(f"\n✔ Done! Generated {len(remaining_rows)} responses in {total_elapsed:.1f}s ({len(remaining_rows)/total_elapsed:.2f} prompts/s)")

# ── Main Entrypoint ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GCP Benchmark Response Generation Engine")
    parser.add_argument("--model", type=str, choices=list(MODELS.keys()), required=True, help="Model to generate from")
    parser.add_argument("--regime", type=str, choices=list(REGIMES.keys()), required=True, help="Prompting regime to use")
    parser.add_argument("--project-id", type=str, default=PROJECT_ID, help="GCP project ID for Vertex AI MaaS")
    parser.add_argument("--test", action="store_true", help="Run a quick diagnostic streaming connection test")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Path to input JSON prompts dataset")
    parser.add_argument("--prompt-file", type=str, default=DEFAULT_PROMPT_FILE, help="Versioned JSON file containing system prompts")
    parser.add_argument("--max-rows", type=int, default=0, help="Max prompts to process (0 for unlimited)")
    parser.add_argument("--concurrency", type=int, default=2, help="Number of concurrent worker threads")
    args = parser.parse_args()

    prompt_version, regimes = load_regimes(args.prompt_file)
    if args.regime not in regimes:
        print(f"Error: Regime '{args.regime}' not found in {args.prompt_file}. Available: {', '.join(sorted(regimes))}")
        sys.exit(1)
    if not args.project_id:
        print("Error: GCP project ID is required. Pass --project-id or set GOOGLE_CLOUD_PROJECT/GCP_PROJECT.")
        sys.exit(1)

    token = get_token()
    if not token:
        print("Error: Could not retrieve active GCP Auth Token. Please run: gcloud auth login")
        sys.exit(1)
        
    if args.test:
        run_diagnostic_test(args.model, args.regime, token, args.project_id, regimes)
    else:
        run_batch_generation(
            model_key=args.model,
            regime_key=args.regime,
            token=token,
            project_id=args.project_id,
            input_file=args.input,
            max_rows=args.max_rows,
            concurrency=args.concurrency,
            prompt_version=prompt_version,
            regimes=regimes
        )

if __name__ == "__main__":
    main()
