#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Auto-CoT (build demonstrations)

Goal:
  Automatically generate chain-of-thought (CoT) demonstrations using an LLM and
  stitch them into a few-shot prompt for Stage-2 reasoning.

Modes:
  - downgrade (TF→U): build demos only on items whose Stage-1 label ∈ {T, F}
  - upgrade   (U→TF): build demos only on items whose Stage-1 label == U

Inputs: 
  --data_path            JSON array; each item has d_id, text, hypothesis, predicate, (answer)
  --stage1_prompt        path to the Stage-1 (predicate-only) prompt
  --instruction_prompt   path to a seed instruction for this stage (e.g., your downgrade/upgrade rules)
  --gen_model            model key for generating CoT demos (mapped in models/model_configs.yaml)
  --num_demos            number of demonstrations to generate (default: 8)

Outputs:
  autocot/outputs/demos_{stage}.json         # list of demos: text/hypothesis/predicate/rationale/label/raw
  autocot/outputs/fewshot_cot_{stage}.txt    # few-shot CoT prompt for Stage-2 reasoning
"""

import os
import sys
import json
import random
import argparse
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- Ensure project root is importable: <repo_root>/src/... ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Prefer importing project utilities; otherwise raise (we want real API calls here).
from src.api.run_model import call_model
from src.utils.prompt_loader import load_prompt
from src.utils.helpers import fmt, extract_label


def stage1_labels(
    data: List[Dict[str, Any]],
    stage1_prompt: str,
    model: str,
    temperature: float = 0.2,
    max_workers: int = 8
) -> List[str]:
    """Run Stage-1 (predicate-only) on the entire dataset and return labels per item."""
    labels = [None] * len(data)

    def _one(i, it):
        p = fmt(stage1_prompt, it["text"], it["hypothesis"], it.get("predicate", ""))
        try:
            rsp = call_model(model, p, temperature).strip()
        except Exception as e:
            rsp = f"[ERROR] {e}"
        return i, extract_label(rsp)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_one, i, it) for i, it in enumerate(data)]
        for fu in tqdm(as_completed(futs), total=len(futs), desc="Stage1"):
            i, lab = fu.result()
            labels[i] = lab
    return labels


def diversity_sample(indices: List[int], items: List[Dict[str, Any]], k: int) -> List[int]:
    """
    Heuristic diversity sampling by (predicate, text-length bucket).
    Round-robin draw across buckets to avoid homogeneous demos.
    """
    if not indices:
        return []
    buckets = {}
    for i in indices:
        pred = items[i].get("predicate", "")
        length_bin = len(items[i]["text"]) // 40
        key = (pred, length_bin)
        buckets.setdefault(key, []).append(i)

    order = sorted(buckets.keys(), key=lambda x: (x[0], x[1]))
    picked = []
    while len(picked) < k and order:
        for key in list(order):
            arr = buckets.get(key, [])
            if arr:
                picked.append(arr.pop(random.randrange(len(arr))))
                if len(picked) >= k:
                    break
            if not arr:
                order.remove(key)

    while len(picked) < k:
        picked.append(random.choice(indices))
    random.shuffle(picked)
    return picked[:k]


def build_cot_for_item(
    it: Dict[str, Any],
    instruction: str,
    model: str,
    temperature: float = 0.5
) -> Dict[str, Any]:
    """
    Ask the LLM to think step-by-step and return JSON:
      {"rationale": "...", "label": "T|F|U"}
    """
    meta = f"""{instruction}

Please think step-by-step, then output ONLY one JSON object with fields:
- "rationale": a brief reasoning in Chinese
- "label": one of "T", "F", "U"

【文本】{it['text']}
【命题】{it['hypothesis']}
【谓词】{it.get('predicate','')}
"""
    try:
        rsp = call_model(model, meta, temperature=temperature).strip()
    except Exception as e:
        # Robust fallback: keep running and record the error in 'raw'
        rsp = f"[ERROR] {e}"

    try:
        obj = json.loads(rsp)
        lab = (obj.get("label", "") if isinstance(obj, dict) else "").upper()
        if lab not in {"T", "F", "U"}:
            lab = extract_label(rsp)
        rationale = obj.get("rationale", "") if isinstance(obj, dict) else ""
    except Exception:
        lab = extract_label(rsp)
        rationale = ""

    return {
        "d_id": it.get("d_id"),
        "text": it["text"],
        "hypothesis": it["hypothesis"],
        "predicate": it.get("predicate", ""),
        "rationale": rationale,
        "label": lab,
        "raw": rsp,
    }


def render_fewshot_prompt(demos: List[Dict[str, Any]], tail_instruction: str) -> str:
    """
    Stitch a few-shot prompt block using demos,
    followed by a tail instruction (e.g., "Output T / F / U only").
    """
    lines = []
    for j, d in enumerate(demos, 1):
        lines.append(f"[Example {j}]")
        lines.append(f"Text: {d['text']}")
        lines.append(f"Hypothesis: {d['hypothesis']}")
        lines.append(f"Predicate: {d.get('predicate','')}")
        if d.get("rationale"):
            lines.append(f"Reasoning: {d['rationale']}")
        lines.append(f"Conclusion: {d['label']}")
        lines.append("")
    lines.append("Follow the style above and answer the next question.")
    lines.append(tail_instruction.strip())
    return "\n".join(lines).strip()


def main():
    ap = argparse.ArgumentParser(description="Auto-CoT: build CoT demonstrations for downgrade (TF→U) or upgrade (U→TF)")
    ap.add_argument("--data_path", required=True, help="JSON array: d_id,text,hypothesis,predicate,(answer)")
    ap.add_argument("--stage", choices=["downgrade", "upgrade"], required=True, help="downgrade=TF→U, upgrade=U→TF")
    ap.add_argument("--stage1_prompt", required=True, help="Stage-1 (predicate-only) prompt file")
    ap.add_argument("--instruction_prompt", required=True, help="Seed instruction for this stage (e.g., your rules)")
    ap.add_argument("--gen_model", required=True, help="Model key for generating CoT demos")
    ap.add_argument("--num_demos", type=int, default=8, help="Number of demonstrations to build")
    ap.add_argument("--temperature", type=float, default=0.5, help="Generation temperature for CoT")
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--out_dir", default="autocot/outputs")
    args = ap.parse_args()

    data = json.load(open(args.data_path, "r", encoding="utf-8"))
    p1 = load_prompt(args.stage1_prompt)
    instruction = load_prompt(args.instruction_prompt)

    # Stage-1 to determine eligible subset
    s1 = stage1_labels(data, p1, model=args.gen_model, temperature=0.0, max_workers=args.max_workers)

    if args.stage == "downgrade":
        eligible = [i for i, lab in enumerate(s1) if lab in {"T", "F"}]
        stage_tag = "downgrade"
    else:
        eligible = [i for i, lab in enumerate(s1) if lab == "U"]
        stage_tag = "upgrade"

    if not eligible:
        print("No eligible items for the requested stage. Check dataset or Stage-1 prompt.")
        return

    pick_idx = diversity_sample(eligible, data, args.num_demos)
    items = [data[i] for i in pick_idx]

    # Build CoT demos in parallel
    demos = [None] * len(items)

    def _one(i, it):
        demo = build_cot_for_item(it, instruction, model=args.gen_model, temperature=args.temperature)
        return i, demo

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(_one, i, it) for i, it in enumerate(items)]
        for fu in tqdm(as_completed(futs), total=len(futs), desc=f"Build CoT demos [{stage_tag}]"):
            i, demo = fu.result()
            demos[i] = demo

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, f"demos_{stage_tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(demos, f, ensure_ascii=False, indent=2)

    tail_inst = "Output ONLY one uppercase letter: T / F / U"
    prompt_text = render_fewshot_prompt(demos, tail_instruction=tail_inst)
    prompt_path = os.path.join(args.out_dir, f"fewshot_cot_{stage_tag}.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)

    print(f"Saved demos to {json_path}")
    print(f"Saved few-shot CoT prompt to {prompt_path}")


if __name__ == "__main__":
    main()
