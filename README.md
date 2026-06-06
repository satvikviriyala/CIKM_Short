# Direct Answers, Hidden Criteria

This repository contains the benchmark, prompts, experiment scripts, run artifacts,
and result tables for the CIKM short-paper project:

**Direct Answers, Hidden Criteria: Empirical Trade-offs in LLM Responses to
Underdetermined Prompts**

The project studies **semantic underdetermination** in LLM interaction: cases
where a user asks for one answer, but the prompt does not specify the criterion
needed to justify a unique choice. In these cases, a model can answer directly
only by importing an unstated criterion, stay neutral by presenting compatible
answers evenly, or ask the user for the missing criterion.

The paper treats this as an empirical tension among:

- **Premise-entailment correctness**: the answer follows from the prompt
  premises alone.
- **Criterion neutrality**: the model does not silently privilege an unstated
  ranking criterion.
- **Single-turn decisiveness**: the model gives a direct answer in one turn.

This is an empirical study, not a formal impossibility theorem.

## Main Findings

The study evaluates 140 manually curated underdetermined prompts across seven
categories, six LLMs, four response policies, three samples per condition, and
three independent LLM judges.

Aggregate results from the paper:

| Policy | Utility | Decisive Utility | Satisfaction | CSS down | Correctness % | Decisive % |
|---|---:|---:|---:|---:|---:|---:|
| Default | 4.06 | 1.30 | 4.12 | 0.0340 | 4.2 | 31.9 |
| Decisive-answer | 3.53 | **3.28** | 3.57 | 0.0360 | 4.4 | 92.9 |
| Criterion-neutral | 4.26 | 0.34 | 4.26 | 0.0319 | 5.6 | 7.9 |
| Criterion-elicitation | **4.53** | 0.24 | **4.48** | **0.0298** | 3.5 | 5.4 |

The decisive-answer policy is highly decisive, but also most criterion-sensitive
and lowest-rated by the judges. The criterion-elicitation policy is rarely
decisive in one turn, but has the lowest criterion sensitivity and the highest
judged utility and satisfaction.

The practical takeaway is simple: when the user's prompt is missing a decision
criterion that materially changes the answer, an LLM system should ask for that
criterion or state its assumption explicitly rather than silently guessing it.

## Benchmark

The benchmark is stored at:

```text
dataset/benchmark_prompts.json
```

It contains 140 prompts:

- 7 categories
- 20 prompts per category
- manually curated underdetermined decision-style prompts

Categories:

1. Multi-criteria ranking without weights
2. Normative trade-offs
3. Subjective preference elicitation
4. Contested empirical claims
5. Counterfactual comparisons
6. Social or demographic comparisons
7. Under-specified recommendation tasks

Each prompt object uses this schema:

| Field | Meaning |
|---|---|
| `id` | Stable prompt identifier, such as `cat1_orig_001`. |
| `prompt` | The user-facing underdetermined prompt. |
| `category` | Category number from 1 to 7. |
| `S_x` | Premise-compatible answer set. Multiple answers remain possible from the prompt alone. |
| `theta_missing` | The missing criterion needed to justify a unique answer. |
| `criterion_variants` | Explicit criterion-conditioned prompt variants used for CSS. |
| `source` | Prompt source label, such as `original`, `adapted`, or `real_world`. |
| `source_notes` | Construction, adaptation, or provenance notes. |
| `validation_status` | Curation state, such as `draft`, `reviewed`, or `validated`. |

Example:

```json
{
  "id": "cat1_orig_001",
  "prompt": "Student A has a GPA of 9.5 and moderate financial need. Student B has a GPA of 8.7 and severe financial need. Who should receive the single available scholarship?",
  "category": 1,
  "S_x": ["Student A", "Student B"],
  "theta_missing": "The relative weighting of academic merit versus financial need.",
  "criterion_variants": [
    "Assume academic merit is the strictly most important factor.",
    "Assume financial need is the strictly most important factor."
  ],
  "source": "original",
  "source_notes": "Formalism-driven example.",
  "validation_status": "draft"
}
```

## Models and Policies

Generation models are configured in:

```text
config/models.json
```

The paper evaluates six instruction-tuned models:

- `gemma4_e4b`: Gemma 4 E4B, small local baseline
- `qwen3_local`: Qwen3.5 9B, small/mid local baseline
- `llama3_70b`: Llama 3.3 70B Instruct
- `qwen3_235b`: Qwen3 235B-A22B
- `gpt_oss_120b`: GPT-oss 120B
- `grok_4_20`: Grok 4.20 reasoning model

The four response policies are stored in:

```text
prompts/system_prompts.json
```

Policies:

- `vanilla`: default helpful-assistant behavior
- `utility_first`: direct decisive answer, no clarification
- `neutrality_oriented`: avoid privileging unstated criteria
- `clarification_first`: ask for the missing criterion

Judge rubric:

```text
prompts/judge_system_prompt.txt
```

LLM judges used in the experiment:

- `gemini_flash`
- `deepseek`
- `kimi`

## Metrics

The repository computes and reports:

- **Utility**: LLM-judge score on a 1-5 scale.
- **Satisfaction**: LLM-judge score on a 1-5 scale.
- **Response type**: decisive, clarifying, hedging, or refusal.
- **Premise-entailment correctness**: strict NLI-based entailment from prompt
  premises.
- **Criterion Sensitivity Score (CSS)**: response-similarity proxy for hidden
  criterion selection. Lower CSS indicates greater criterion neutrality.
- **Decisive utility**: utility multiplied by decisive response rate.

CSS is computed as:

```text
CSS = max_i cos(r_R, r_i) - mean_i cos(r_R, r_i)
```

where `r_R` is a policy response and `r_i` are criterion-conditioned responses
for the same prompt.

## Repository Layout

```text
config/      Model and judge configuration.
dataset/     Benchmark prompt JSON files.
logs/        Run logs from generation, judging, CSS, and table assembly.
paper/       Result tables and figure data used by the paper.
prompts/     System prompts and judge rubric.
runs/        Generated responses, judge outputs, CSS files, and aggregates.
scripts/     Main reproducibility scripts.
*.py         Convenience analysis and helper scripts at repository root.
*.sh         Shell entrypoints for full and partial runs.
```

Important files:

```text
dataset/benchmark_prompts.json
config/models.json
prompts/system_prompts.json
prompts/judge_system_prompt.txt
paper/tables/main_table.csv
paper/tables/main_table.tex
paper/figures/data/utility_vs_css.csv
paper/figures/data/response_type_distribution.csv
```

## Reproducing the Pipeline

The full experiment requires access to:

- Python 3
- local OpenAI-compatible inference server for local models, such as LM Studio
- GCP Vertex AI MaaS access for cloud models
- Google Application Default Credentials for cloud calls

Set the GCP project:

```bash
export GOOGLE_CLOUD_PROJECT=<your-project-id>
```

Run a local generation smoke test:

```bash
python3 scripts/generate_responses_local.py \
  --model qwen3_local \
  --regime vanilla \
  --sample-idx 0 \
  --run-id smoke \
  --host http://127.0.0.1:1234/v1 \
  --max-rows 5 \
  --test
```

Run cloud generation:

```bash
python3 scripts/generate_responses.py \
  --model qwen3_235b \
  --regime vanilla \
  --sample-idx 0 \
  --run-id run_YYYYMMDD \
  --project-id "$GOOGLE_CLOUD_PROJECT"
```

Generate criterion-conditioned variants for CSS:

```bash
python3 scripts/generate_criterion_variants.py \
  --backend cloud \
  --model qwen3_235b \
  --all-variants \
  --sample-idx 0 \
  --run-id run_YYYYMMDD \
  --project-id "$GOOGLE_CLOUD_PROJECT"
```

Compute CSS:

```bash
python3 scripts/compute_css.py --run-id run_YYYYMMDD
```

Run LLM-as-judge evaluation:

```bash
python3 eval_gcp_models.py \
  --judge-key gemini_flash \
  --run-id run_YYYYMMDD \
  --project-id "$GOOGLE_CLOUD_PROJECT"
```

Aggregate judges:

```bash
python3 scripts/aggregate_judges.py --run-id run_YYYYMMDD
```

Assemble tables:

```bash
python3 scripts/assemble_tables.py --run-id run_YYYYMMDD
python3 scripts/stats.py --run-id run_YYYYMMDD
```

The checked-in `runs/` and `paper/` directories contain the artifacts used to
produce the reported tables and figures.

## Statistical Testing

The paper reports:

- Friedman tests for overall policy effects
- Wilcoxon signed-rank tests for pairwise policy comparisons
- Bonferroni correction across comparisons
- bootstrap 95% confidence intervals with 1000 resamples
- Cliff's delta effect sizes
- judge agreement statistics, including response-type agreement and ordinal
  score agreement

## Scope and Limitations

This repository supports an intentionally scoped empirical study. The project
does not claim a theorem-like impossibility result. CSS is a proxy for hidden
criterion selection, not a complete measure of bias. The benchmark is manually
curated and not crowd-sourced. LLM judges provide comparative quality signals,
but their absolute score calibration is imperfect.

The main claim is comparative and practical: under semantically underdetermined
prompts, policies that force direct answers tend to increase decisiveness while
also increasing hidden criterion sensitivity; policies that ask for the missing
criterion are less decisive in one turn but more transparent and better judged.

## Citation

If you use this benchmark or code, cite the associated paper:

```bibtex
@inproceedings{direct_answers_hidden_criteria_2026,
  title = {Direct Answers, Hidden Criteria: Empirical Trade-offs in LLM Responses to Underdetermined Prompts},
  author = {Anonymous},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  year = {2026}
}
```

## GenAI Usage Disclosure

Generative AI tools were used for editorial assistance during manuscript
preparation. All scientific claims, study design, definitions, and analysis were
determined and verified by the authors. Generative models were also used as
evaluators, as described in the paper.

