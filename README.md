# FIE2025 Factivity Evaluation Pipeline

This repository provides a simple evaluation pipeline for factivity inference tasks based on the [FIE2025 dataset](https://github.com/UM-FAH-Yuan/FIE2025).  
It supports OpenAI-compatible LLM endpoints (e.g., Qwen, DeepSeek) via LangChain.

---

## 0. Overview

This repository provides:
- **Two-stage evaluation pipeline** for natural text (predicate-based Stage-1, full-text Stage-2).
- **Type-based evaluation pipeline** for artificial data.
- **Auto-CoT** scripts to automatically generate chain-of-thought (CoT) demonstrations and select the best few-shot prompts for Stage-2 refinement.
- A technical report describing methodology and results: [System_Report_for_CCL25_Eval_Task_4pdf.pdf](assets/System_Report_for_CCL25_Eval_Task_4pdf.pdf)

**中文版 README**: [README_FIE2025_CN.md](README_FIE2025_CN.md)

---

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Edit `models/model_configs.yaml` and add your API key (or export env variables).

---

## 2. Download Dataset

Download from: [https://github.com/UM-FAH-Yuan/FIE2025](https://github.com/UM-FAH-Yuan/FIE2025)

---

## 3. Pipeline Idea

We approach the factivity inference task in **two main steps**:

1. **Stage-1:** Predicate-only heuristic prediction (T/F/U) based purely on the factuality tendency of the verb/predicate.
2. **Stage-2:** Full-text reasoning to revise the Stage-1 decision:
   - **Upgrade:** From uncertain (U) to T/F when additional context supports a conclusion.
   - **Downgrade:** From T/F to U if negation, hypothetical, or self-uncertainty is found.

This two-stage setup allows the system to combine **linguistic prior knowledge** (predicate type) and **contextual inference**.

Additionally, we provide **Auto-CoT** scripts to generate few-shot CoT demonstrations automatically, allowing the model to produce better Stage-2 judgments without manual crafting of examples.

---

## 4. Run

### 4.1 NAT data (two-stage pipeline)

```bash
python scripts/run_two_stage_nat.py \
  --data_path data/natural.json \
  --prompt1 prompts/stage1_only_predicate.txt \
  --prompt2 prompts/stage2_read_full_upgrade.txt \
  --prompt3 prompts/stage2_read_full_downgrade.txt \
  --model Qwen-Max \
  --output_dir results
```

### 4.2 ART data (type-based pipeline)

```bash
python scripts/run_art_by_type.py \
  --data_path data/artificial.json \
  --prompt_T prompts/art_prompt_T.txt \
  --prompt_F prompts/art_prompt_F.txt \
  --model_name Qwen-Max \
  --output_dir results
```

---

## 5. Auto-CoT Prompt Optimization

### 5.1 Build CoT demonstrations

```bash
python autocot/autocot_build_demos.py \
  --data_path data/natural.json \
  --stage upgrade \
  --stage1_prompt prompts/stage1_only_predicate.txt \
  --instruction_prompt prompts/stage2_read_full_upgrade.txt \
  --gen_model Qwen-Max \
  --num_demos 8 \
  --temperature 0.5 \
  --max_workers 8 \
  --out_dir autocot/outputs
```

Outputs:
- `autocot/outputs/demos_upgrade.json`
- `autocot/outputs/fewshot_cot_upgrade.txt`

---

### 5.2 Evaluate & select best few-shot prompt

```bash
python autocot/autocot_eval_demos.py \
  --data_path data/natural.json \
  --demos_json autocot/outputs/demos_upgrade.json \
  --eval_model Qwen-Max \
  --num_pools 5 \
  --shots 8 \
  --temperature 0.2 \
  --max_workers 8 \
  --out_dir autocot/selection
```

Outputs:
- `autocot/selection/best_fewshot.txt` – best few-shot prompt

---

### 5.3 Final run with optimized few-shot

```bash
python scripts/run_two_stage_nat.py \
  --data_path data/natural.json \
  --prompt1 prompts/stage1_only_predicate.txt \
  --prompt2 autocot/selection/best_fewshot.txt \
  --prompt3 prompts/stage2_read_full_downgrade.txt \
  --model Qwen-Max \
  --output_dir results
```

---

## 6. Output

Results are saved to `results/*.json` with predicted labels for each `d_id`.

Each record contains:
- `d_id`: instance ID
- `gold`: gold label (if available)
- `pred`: predicted label
- `stage1_pred`, `stage2_pred`: intermediate stage predictions
- `raw_stage1`, `raw_stage2`: raw model outputs

---
