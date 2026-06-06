#!/usr/bin/env python3
"""
Criterion-variant response generation for CIKM C-NB-U benchmark.

For each prompt p and each criterion variant v_i, generates responses under
the VANILLA system prompt with the user prompt extended by:

    {p.prompt}\\n\\n{v_i}

These responses are used exclusively for Criterion Sensitivity Score (CSS)
computation.  They are NOT a fifth prompting regime.

Output path per variant:
    runs/{run_id}/responses/{model_key}_variant_{variant_idx}_sample{n}_responses.jsonl

Resumption key: (prompt_id, variant_idx, sample_idx).

Supports both cloud (Vertex AI MaaS) and local (LM Studio / vLLM) backends
via --backend {cloud,local}.  Cloud and local model sets are filtered from
config/models.json automatically.
"""

import sys
import json
import time
import argparse
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gen_common import (
    load_model_config,
    get_generation_models,
    load_regimes,
    load_prompts,
    make_variant_output_path,
    load_done_variant_keys,
    build_output_row,
    parse_retry_after,
    print_progress,
    DEFAULT_INPUT,
    DEFAULT_PROMPT_FILE,
)


# ── Backend-agnostic API callable factory ────────────────────────

def _make_api_fn(backend: str, model_cfg: dict, **kwargs):
    """
    Return a callable  fn(system_prompt, user_prompt) -> (response, reasoning, raw_excerpt)
    that dispatches to the right transport without the caller knowing which.

    kwargs for cloud: token, project_id
    kwargs for local: host
    """
    if backend == "cloud":
        from generate_responses import call_cloud_chat
        token = kwargs["token"]
        project_id = kwargs["project_id"]

        def fn(sp: str, up: str) -> tuple[str, str, str]:
            return call_cloud_chat(model_cfg, token, project_id, sp, up)

    else:
        from generate_responses_local import call_local_chat
        host = kwargs["host"]

        def fn(sp: str, up: str) -> tuple[str, str, str]:
            return call_local_chat(model_cfg, host, sp, up)

    return fn


# ── Per-row worker ────────────────────────────────────────────────

def generate_variant_worker(
    api_fn,
    model_key: str,
    model_cfg: dict,
    vanilla_system_prompt: str,
    system_prompt_version: str,
    base_prompt: str,
    prompt_id: str,
    category,
    variant_idx: int,
    variant_text: str,
    sample_idx: int,
    run_id: str,
) -> dict:
    user_prompt = base_prompt + "\n\n" + variant_text

    response_text = ""
    reasoning_content = ""
    raw_excerpt = ""

    for attempt in range(3):
        try:
            response_text, reasoning_content, raw_excerpt = api_fn(
                vanilla_system_prompt, user_prompt
            )
            if response_text.strip():
                break
            print(
                f"  [{prompt_id}/v{variant_idx}] [Attempt {attempt+1}] "
                f"Empty response — retrying"
            )
        except urllib.error.HTTPError as e:
            err_snippet = e.read().decode(errors="ignore")[:120]
            wait = (parse_retry_after(e) or 60) if e.code == 429 else 2 ** (attempt + 1)
            print(
                f"  [{prompt_id}/v{variant_idx}] [Attempt {attempt+1}] "
                f"HTTP {e.code}: {err_snippet} — waiting {wait:.0f}s"
            )
            if attempt < 2:
                time.sleep(wait)
        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(
                f"  [{prompt_id}/v{variant_idx}] [Attempt {attempt+1}] "
                f"Error: {e} — waiting {wait:.0f}s"
            )
            if attempt < 2:
                time.sleep(wait)

    if not response_text.strip():
        response_text = "ERROR: generation failed after 3 attempts."

    return build_output_row(
        prompt_id=prompt_id,
        category=category,
        model_key=model_key,
        model_cfg=model_cfg,
        regime_key="vanilla",
        system_prompt_version=system_prompt_version,
        system_prompt=vanilla_system_prompt,
        user_prompt=user_prompt,
        sample_idx=sample_idx,
        run_id=run_id,
        response=response_text,
        reasoning_content=reasoning_content,
        raw_provider_response_excerpt=raw_excerpt,
        criterion_variant_idx=variant_idx,
        criterion_variant_text=variant_text,
    )


# ── Test mode ─────────────────────────────────────────────────────

def run_test(
    model_key: str,
    model_cfg: dict,
    backend: str,
    api_fn,
    vanilla_system_prompt: str,
    system_prompt_version: str,
    prompts: list[dict],
    variant_idx: int,
) -> None:
    # Find first prompt that has the requested variant
    target = None
    for row in prompts:
        variants = row.get("criterion_variants", [])
        if variant_idx < len(variants):
            target = row
            break

    if target is None:
        print(f"✘ No prompt has a variant at index {variant_idx}.")
        return

    pid = str(target.get("id") or "unknown")
    base_prompt = target.get("prompt", "")
    variant_text = target["criterion_variants"][variant_idx]
    user_prompt = base_prompt + "\n\n" + variant_text

    print(f"\n=== Criterion-Variant Diagnostic Test ===")
    print(f"  Backend       : {backend}")
    print(f"  Model key     : {model_key} ({model_cfg['provider_id']})")
    print(f"  Prompt ID     : {pid}")
    print(f"  Variant idx   : {variant_idx}")
    print(f"  Variant text  : {variant_text[:80]}{'...' if len(variant_text) > 80 else ''}")
    print(f"  System prompt : vanilla (version={system_prompt_version})")
    print(f"  user_prompt   : {user_prompt[:120]}{'...' if len(user_prompt) > 120 else ''}")
    print("-" * 60)

    test_cfg = dict(model_cfg, max_tokens=min(256, model_cfg["max_tokens"]))

    # Rebuild api_fn with capped max_tokens for speed
    if backend == "cloud":
        from generate_responses import call_cloud_chat, get_token
        import os
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
        if not project_id:
            # Try to read from args — callers pass it via env or flag; fall back gracefully
            print("  [Note] GOOGLE_CLOUD_PROJECT not set; set it or pass --project-id.")
            return
        token = get_token()
        if not token:
            print("  [Note] Could not get GCP token.")
            return

        def test_api_fn(sp, up):
            return call_cloud_chat(test_cfg, token, project_id, sp, up, stream=True)
    else:
        from generate_responses_local import call_local_chat, manage_lmstudio_model
        host_url = model_cfg.get("_host", "http://127.0.0.1:1234/v1")
        manage_lmstudio_model(host_url, model_cfg["provider_id"])

        def test_api_fn(sp, up):
            return call_local_chat(test_cfg, host_url, sp, up, stream=True)

    try:
        response_text, reasoning_content, _ = test_api_fn(vanilla_system_prompt, user_prompt)
        print()
        print(f"[reasoning_content: {len(reasoning_content)} chars]")
        print("\n--- Reconstructed output row (key fields) ---")
        row = build_output_row(
            prompt_id=pid,
            category=target.get("category", ""),
            model_key=model_key,
            model_cfg=model_cfg,
            regime_key="vanilla",
            system_prompt_version=system_prompt_version,
            system_prompt=vanilla_system_prompt,
            user_prompt=user_prompt,
            sample_idx=0,
            run_id="test",
            response=response_text,
            reasoning_content=reasoning_content,
            raw_provider_response_excerpt="",
            criterion_variant_idx=variant_idx,
            criterion_variant_text=variant_text,
        )
        key_fields = {
            k: row[k]
            for k in (
                "prompt_id", "regime", "criterion_variant_idx",
                "criterion_variant_text", "system_prompt",
                "user_prompt", "sample_idx",
            )
        }
        print(json.dumps(key_fields, indent=2, ensure_ascii=False))
        print("\n✔ Diagnostic completed.")
    except urllib.error.HTTPError as e:
        print(f"\n✘ HTTP {e.code}: {e.read().decode(errors='ignore')[:300]}")
    except Exception as e:
        print(f"\n✘ Diagnostic failed: {e}")


# ── Batch mode ────────────────────────────────────────────────────

def run_batch(
    model_key: str,
    model_cfg: dict,
    api_fn,
    vanilla_system_prompt: str,
    system_prompt_version: str,
    prompts: list[dict],
    variant_indices: list[int],
    sample_idx: int,
    run_id: str,
    concurrency: int,
) -> None:
    print(f"\n{'='*60}")
    print(f"  Model:    {model_key} ({model_cfg['provider_id']})")
    print(f"  Sample:   {sample_idx}  |  Run: {run_id}")
    print(f"  Variants: {variant_indices}")
    print(f"{'='*60}")

    for variant_idx in variant_indices:
        output_file = make_variant_output_path(run_id, model_key, variant_idx, sample_idx)
        done = load_done_variant_keys(output_file)
        if done:
            print(f"\n  [v{variant_idx}] Resuming: {len(done)} pairs already written.")

        work_items = []
        for row in prompts:
            variants = row.get("criterion_variants", [])
            if variant_idx >= len(variants):
                continue  # prompt has fewer variants than requested — skip cleanly
            pid = str(row.get("id") or "unknown")
            if (pid, variant_idx, sample_idx) in done:
                continue
            work_items.append({
                "prompt": row.get("prompt", ""),
                "prompt_id": pid,
                "category": row.get("category", ""),
                "variant_text": variants[variant_idx],
            })

        if not work_items:
            print(f"  [v{variant_idx}] ✔ Nothing to generate.")
            continue

        print(f"\n  [v{variant_idx}] Output: {output_file}")
        print(f"  [v{variant_idx}] Remaining: {len(work_items)} prompts")
        start = time.time()
        completed = 0

        with open(output_file, "a", encoding="utf-8") as out:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(
                        generate_variant_worker,
                        api_fn,
                        model_key,
                        model_cfg,
                        vanilla_system_prompt,
                        system_prompt_version,
                        item["prompt"],
                        item["prompt_id"],
                        item["category"],
                        variant_idx,
                        item["variant_text"],
                        sample_idx,
                        run_id,
                    ): item["prompt_id"]
                    for item in work_items
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out.flush()
                        completed += 1
                        print_progress(completed, len(work_items), start)
                    except Exception as e:
                        print(f"  [Error] Future raised: {e}")

        elapsed = time.time() - start
        rate = completed / elapsed if elapsed > 0 else 0.0
        print(
            f"  [v{variant_idx}] ✔ Done. "
            f"{completed} responses in {elapsed:.1f}s ({rate:.2f}/s)"
        )


# ── Entry point ───────────────────────────────────────────────────

def main() -> None:
    cfg = load_model_config()
    prompt_version, regimes = load_regimes(DEFAULT_PROMPT_FILE)
    vanilla_system_prompt = regimes["vanilla"]

    parser = argparse.ArgumentParser(
        description="Criterion-variant generation for CSS computation (CIKM C-NB-U)"
    )
    parser.add_argument(
        "--backend", required=True, choices=["cloud", "local"],
        help="cloud = Vertex AI MaaS; local = LM Studio / vLLM",
    )
    parser.add_argument(
        "--model", required=True,
        help="Model key from config/models.json (must match the chosen backend)",
    )

    var_group = parser.add_mutually_exclusive_group()
    var_group.add_argument(
        "--variant-idx", type=int, default=None,
        help="Generate responses for this single variant index across all prompts",
    )
    var_group.add_argument(
        "--all-variants", action="store_true",
        help="Generate responses for ALL variant indices found in the dataset",
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
        "--project-id", default=None,
        help="GCP project ID (cloud backend; or set GOOGLE_CLOUD_PROJECT)",
    )
    parser.add_argument(
        "--host", default="http://127.0.0.1:1234/v1",
        help="Local server URL (local backend; default: http://127.0.0.1:1234/v1)",
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
        help="Limit number of prompts (0 = all); useful for smoke runs",
    )
    parser.add_argument(
        "--concurrency", type=int, default=0,
        help="Worker threads (0 = auto: 2 for cloud, 1 for local)",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run a single-prompt diagnostic and print the key output fields",
    )
    args = parser.parse_args()

    # ── Validate model key against backend ───────────────────────
    local_only = args.backend == "local"
    available = get_generation_models(cfg, local_only=local_only)
    if args.model not in available:
        all_models = get_generation_models(cfg, local_only=not local_only)
        if args.model in all_models:
            parser.error(
                f"Model '{args.model}' is a {'local' if not local_only else 'cloud'} model; "
                f"use --backend {'local' if not local_only else 'cloud'} for it."
            )
        else:
            parser.error(
                f"Unknown model '{args.model}'. "
                f"Available for --backend {args.backend}: {sorted(available)}"
            )

    model_cfg = available[args.model]

    # Stash host on model_cfg so run_test can reach it without extra params
    model_cfg = dict(model_cfg, _host=args.host)

    # ── Resolve concurrency ───────────────────────────────────────
    concurrency = args.concurrency or (1 if args.backend == "local" else 2)

    # ── Reload prompts / regimes if custom files given ────────────
    if args.prompt_file != DEFAULT_PROMPT_FILE:
        prompt_version, regimes = load_regimes(args.prompt_file)
        vanilla_system_prompt = regimes["vanilla"]

    prompts = load_prompts(args.input, args.max_rows)

    # ── Build backend-specific API callable ───────────────────────
    import os

    if args.backend == "cloud":
        from generate_responses import get_token
        project_id = (
            args.project_id
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCP_PROJECT")
        )
        if not project_id:
            parser.error("--project-id or GOOGLE_CLOUD_PROJECT required for cloud backend.")
        token = get_token()
        if not token:
            print("Error: Could not get GCP token. Run: gcloud auth application-default login")
            sys.exit(1)
        api_fn = _make_api_fn("cloud", model_cfg, token=token, project_id=project_id)

    else:
        from generate_responses_local import manage_lmstudio_model
        if not args.test:
            manage_lmstudio_model(args.host, model_cfg["provider_id"])
        api_fn = _make_api_fn("local", model_cfg, host=args.host)

    # ── Test mode ─────────────────────────────────────────────────
    if args.test:
        test_variant_idx = args.variant_idx if args.variant_idx is not None else 0
        run_test(
            model_key=args.model,
            model_cfg=model_cfg,
            backend=args.backend,
            api_fn=api_fn,
            vanilla_system_prompt=vanilla_system_prompt,
            system_prompt_version=prompt_version,
            prompts=prompts,
            variant_idx=test_variant_idx,
        )
        return

    # ── Batch mode: validate remaining required args ───────────────
    if args.sample_idx is None:
        parser.error("--sample-idx is required for batch mode")
    if args.run_id is None:
        parser.error("--run-id is required for batch mode")
    if args.variant_idx is None and not args.all_variants:
        parser.error("Either --variant-idx N or --all-variants is required for batch mode")

    # ── Determine variant index list ──────────────────────────────
    if args.all_variants:
        max_variants = max(
            (len(row.get("criterion_variants", [])) for row in prompts),
            default=0,
        )
        if max_variants == 0:
            print("No criterion_variants found in the dataset. Nothing to do.")
            sys.exit(0)
        variant_indices = list(range(max_variants))
    else:
        variant_indices = [args.variant_idx]

    run_batch(
        model_key=args.model,
        model_cfg=model_cfg,
        api_fn=api_fn,
        vanilla_system_prompt=vanilla_system_prompt,
        system_prompt_version=prompt_version,
        prompts=prompts,
        variant_indices=variant_indices,
        sample_idx=args.sample_idx,
        run_id=args.run_id,
        concurrency=concurrency,
    )


if __name__ == "__main__":
    main()
