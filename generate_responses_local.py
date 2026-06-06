#!/usr/bin/env python3
"""
GCP/Local Model Response Generation Script (Local Benchmark Evaluator)
----------------------------------------------------------------------
Queries locally run open models (Gemma 4 E4B, Qwen 3.5 9B, etc.) to generate
responses for 140 prompts in dataset/benchmark_prompts.json across 4 distinct
prompting regimes.

Includes automatic VRAM OOM prevention management for 8GB GPUs (like the RTX 5060)
by dynamically unloading other active models and loading the target model in LM Studio.

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
import urllib.request
import urllib.parse
import urllib.error
import ssl
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Prompting Regimes Specifications ────────────────────────────
DEFAULT_PROMPT_FILE = "prompts/system_prompts.json"

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

# ── VRAM OOM Prevention & Dynamic LM Studio Model Manager ──────
def get_vram_info():
    """Retrieve and display current GPU and VRAM status using nvidia-smi."""
    try:
        res = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            lines = res.stdout.splitlines()
            for line in lines:
                if "GeForce" in line or "RTX" in line or "MiB /" in line:
                    print(f"  [GPU Status] {line.strip()}")
            return True
    except:
        pass
    print("  [GPU Status] nvidia-smi is unavailable from this shell; confirm GPU visibility before local runs.")
    return False

def manage_lmstudio_model(host: str, target_model: str):
    """Automatically unloads other active models and loads the target model to prevent OOM on 8GB GPU."""
    print("\n--- [VRAM & Model Management] ---")
    get_vram_info()
    
    parsed_url = urllib.parse.urlparse(host)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    
    # 1. Fetch currently loaded models
    loaded_models = []
    try:
        req = urllib.request.Request(f"{base_url}/v1/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            models_data = json.loads(resp.read().decode())
            loaded_models = [m["id"] for m in models_data.get("data", [])]
        print(f"  Active models in LM Studio: {loaded_models}")
    except Exception as e:
        print(f"  [Info] Could not query loaded models or LM Studio server is not running: {e}")
        return

    # 2. Check if target model is already loaded
    if target_model in loaded_models:
        print(f"  ✔ Target model '{target_model}' is already loaded.")
        return

    # 3. Unload other models to free up VRAM
    for model in loaded_models:
        if model != target_model:
            print(f"  [Memory Safeguard] Unloading model '{model}' to free up GPU memory...")
            try:
                unload_payload = {"model": model}
                req = urllib.request.Request(
                    f"{base_url}/api/v1/models/unload",
                    data=json.dumps(unload_payload).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                urllib.request.urlopen(req, timeout=10)
                print(f"    ✔ Successfully unloaded '{model}'")
            except Exception as e:
                print(f"    ✘ Failed to unload '{model}': {e}")

    # 4. Load the target model
    print(f"  [Memory Safeguard] Loading model '{target_model}' into GPU...")
    try:
        load_payload = {
            "model": target_model, 
            "context_length": 4096,
            "flash_attention": True
        }
        req = urllib.request.Request(
            f"{base_url}/api/v1/models/load",
            data=json.dumps(load_payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        # 90 second timeout for model loading over local network/disk
        urllib.request.urlopen(req, timeout=90)
        print(f"  ✔ Model '{target_model}' successfully loaded.")
        time.sleep(2) # Give GPU a brief moment to stabilize
    except Exception as e:
        print(f"  ✘ Failed to load model '{target_model}' dynamically: {e}")
        print("  [Note] Please load this model manually inside the LM Studio user interface if dynamic load fails.")

# ── Core Local API Execution Engine ─────────────────────────────
def call_local_chat(host: str, model_name: str, system_prompt: str, user_prompt: str, stream=False, max_tokens=1024):
    """Core HTTP Executor making calls to OpenAI-compatible local APIs (LM Studio, Ollama, vLLM)."""
    base_url = host.rstrip("/")
    url = f"{base_url}/chat/completions"
    
    payload = {
        "model": model_name,
        "stream": stream,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
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
                    text = delta.get("content") or ""
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    full_text.append(text)
                except:
                    pass
        print()
        return "".join(full_text)
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode(errors="ignore")
        raise RuntimeError(f"Local Server HTTP ERROR {e.code}: {err_msg}")
    except Exception as e:
        raise RuntimeError(f"Local API Connection Failed at {url}. Make sure your local server (LM Studio) is running: {e}")

# ── Operational Mode 1: Test Bed Mode ───────────────────────────
def run_diagnostic_test(host: str, model_name: str, regime_key: str, regimes: dict[str, str]):
    """Executes a diagnostic stream request to verify local connectivity and parse capability."""
    manage_lmstudio_model(host, model_name)
    
    system_prompt = regimes[regime_key]
    print(f"\n=== Running Local Diagnostic Generation Test ===")
    print(f"  Server URL: {host}")
    print(f"  Model:      {model_name}")
    print(f"  Regime:     {regime_key}")
    print(f"  System Prompt: \"{system_prompt}\"")
    prompt = "Should phone Y (battery lasts 3 days, mediocre camera) or phone X (great camera, charge twice a day) be bought?"
    print("-" * 58)
    try:
        call_local_chat(host, model_name, system_prompt, prompt, stream=True, max_tokens=256)
        print("✔ Local Diagnostic successfully completed.")
    except Exception as e:
        print(f"✘ Local Diagnostic Failed: {e}")

# ── Operational Mode 2: Batch Generation Engine ─────────────────
def generate_row_worker(
    host: str,
    model_name: str,
    regime_key: str,
    prompt_version: str,
    system_prompt: str,
    row: dict,
    index: int
) -> dict:
    """Single worker thread generating a response for a benchmark prompt."""
    prompt = row.get("prompt") or row.get("query")
    
    for attempt in range(3):
        try:
            response_text = call_local_chat(host, model_name, system_prompt, prompt, max_tokens=1024)
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
                    "model": model_name,
                    "model_id": model_name,
                    "region": "local",
                    "regime": regime_key,
                    "system_prompt_version": prompt_version,
                    "system_prompt": system_prompt,
                    "response": response_text,
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
        except Exception as e:
            print(f"  [Row {index}] [Attempt {attempt+1}] Local request failed: {e}")
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
        "model": model_name,
        "model_id": model_name,
        "region": "local",
        "regime": regime_key,
        "system_prompt_version": prompt_version,
        "system_prompt": system_prompt,
        "response": "ERROR: Local model response generation failed.",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

def run_batch_generation(
    host: str,
    model_name: str,
    regime_key: str,
    input_file: str,
    max_rows: int,
    concurrency: int,
    prompt_version: str,
    regimes: dict[str, str]
):
    """Coordinates concurrent local response generation and records progress incrementally."""
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_file} not found.")
        sys.exit(1)
        
    manage_lmstudio_model(host, model_name)
    
    # Standardize output filenames
    sanitized_model_name = "".join(c if c.isalnum() else "_" for c in model_name).split("/")[-1].lower()
    output_file = Path(f"{sanitized_model_name}_{regime_key}_responses.jsonl")
    system_prompt = regimes[regime_key]
    
    print(f"\n============================================================")
    print(f"  Generating Responses Locally: {model_name} | Regime: {regime_key}")
    print(f"  Local Server Host:            {host}")
    print(f"  Input File:                   {input_file}")
    print(f"  Output File:                  {output_file}")
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
                    host,
                    model_name,
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
    parser = argparse.ArgumentParser(description="Local Benchmark Response Generation Engine")
    parser.add_argument("--model", type=str, default="google/gemma-4-e4b", help="Model ID as registered on local server")
    parser.add_argument("--host", type=str, default="http://127.0.0.1:1234/v1", help="Local server OpenAI API URL")
    parser.add_argument("--regime", type=str, choices=list(REGIMES.keys()), required=True, help="Prompting regime to use")
    parser.add_argument("--test", action="store_true", help="Run a quick diagnostic streaming connection test")
    parser.add_argument("--input", type=str, default="dataset/benchmark_prompts.json", help="Path to input JSON prompts dataset")
    parser.add_argument("--prompt-file", type=str, default=DEFAULT_PROMPT_FILE, help="Versioned JSON file containing system prompts")
    parser.add_argument("--max-rows", type=int, default=0, help="Max prompts to process (0 for unlimited)")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent worker threads")
    args = parser.parse_args()

    prompt_version, regimes = load_regimes(args.prompt_file)
    if args.regime not in regimes:
        print(f"Error: Regime '{args.regime}' not found in {args.prompt_file}. Available: {', '.join(sorted(regimes))}")
        sys.exit(1)
    
    if args.test:
        run_diagnostic_test(args.host, args.model, args.regime, regimes)
    else:
        run_batch_generation(
            host=args.host,
            model_name=args.model,
            regime_key=args.regime,
            input_file=args.input,
            max_rows=args.max_rows,
            concurrency=args.concurrency,
            prompt_version=prompt_version,
            regimes=regimes
        )

if __name__ == "__main__":
    main()
