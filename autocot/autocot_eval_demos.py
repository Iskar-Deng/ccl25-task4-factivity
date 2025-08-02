#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Auto-CoT (evaluate & select)

Goal:
  Given an Auto-CoT demo set (JSON), evaluate several random few-shot subsets
  on a dev set and select the best few-shot prompt by accuracy.

Inputs:
  --data_path     JSON array (dev set): d_id,text,hypothesis,predicate,answer
  --demos_json    path to demos_{stage}.json from the build script
  --eval_model    model key for evaluation
  --num_pools     how many different random few-shot combinations to evaluate (default: 5)
  --shots         how many demonstrations per combination (default: 8)

Outputs:
  autocot/selection/pool_XX_shots{N}.txt     # each candidate few-shot prompt
  autocot/selection/selection_report.json    # accuracy per candidate
  autocot/selection/best_fewshot.txt         # the best-performing few-shot prompt
"""

import os
import sys
import json
import argparse
import random
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- Ensure project root is importable: <repo_root>/src/... ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api.run_model import call_model
from src.utils.prompt_loader import load_prompt
from src.utils.helpers import fmt, extract_label


def assemble_fewshot_prompt_block(demos: List[Dict[str, Any]]) -> str:
    """Render a few-shot block from demos."""
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
    lines.append("Output ONLY one uppercase letter: T / F / U")
    return "\n".join(lines).strip()


def eval_with_prompt_block(
    items: List[Dict[str, Any]],
    prompt_block: str,
    model: str,
    temperature: float = 0.2,
    max_workers: int = 8
) -> List[str]:
    """Evaluate one few-shot block on the entire dev set and return predicted labels."""
    labels = [None] * len(items)

    def _one(i, it):
        q = f"""{prompt_block}

[Question]
Text: {it['text']}
Hypothesis: {it['hypothesis']}
Predicate: {it.get('predicate','')}

Output ONLY one uppercase letter: T / F / U"""
        try:
            rsp = call_model(model, q, temperature).strip()
        except Exception as e:
            rsp = f"[ERROR] {e}"
        return i, extract_label(rsp)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_one, i, it) for i, it in enumerate(items)]
        for fu in tqdm(as_completed(futs), total=len(futs), desc="Eval"):
            i, lab = fu.result()
            labels[i] = lab
    return labels


def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """Simple accuracy; ignores items with missing gold or predictions."""
    valid = [(t, p) for t, p in zip(y_true, y_pred) if t is not None and p is not None]
    if not valid:
        return 0.0
    correct = sum(1 for t, p in valid if t == p)
    return correct / len(valid)


def main():
    ap = argparse.ArgumentParser(description="Auto-CoT evaluation & selection for few-shot demos")
    ap.add_argument("--data_path", required=True, help="JSON array (dev set): d_id,text,hypothesis,predicate,answer")
    ap.add_argument("--demos_json", required=True, help="Path to demos_{stage}.json produced by autocot_build_demos.py")
    ap.add_argument("--eval_model", required=True, help="Model key for evaluation")
    ap.add_argument("--num_pools", type=int, default=5, help="Number of random few-shot combinations to evaluate")
    ap.add_argument("--shots", type=int, default=8, help="Number of demonstrations per combination")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--out_dir", default="autocot/selection")
    args = ap.parse_args()

    data = json.load(open(args.data_path, "r", encoding="utf-8"))
    demos = json.load(open(args.demos_json, "r", encoding="utf-8"))
    os.makedirs(args.out_dir, exist_ok=True)

    gold = [it.get("answer") for it in data]
    report = []
    for pid in range(1, args.num_pools + 1):
        subset = random.sample(demos, min(args.shots, len(demos)))
        block = assemble_fewshot_prompt_block(subset)
        preds = eval_with_prompt_block(
            data, block, args.eval_model,
            temperature=args.temperature, max_workers=args.max_workers
        )
        acc = accuracy(gold, preds)

        save_block = os.path.join(args.out_dir, f"pool_{pid:02d}_shots{len(subset)}.txt")
        with open(save_block, "w", encoding="utf-8") as f:
            f.write(block)

        report.append({"pool": pid, "shots": len(subset), "acc": acc, "path": save_block})
        print(f"[Pool {pid}] acc={acc:.4f} -> {save_block}")

    report.sort(key=lambda x: x["acc"], reverse=True)
    best = report[0] if report else {}

    out_json = os.path.join(args.out_dir, "selection_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"candidates": report, "best": best}, f, ensure_ascii=False, indent=2)
    print(f"Selection report saved to {out_json}")

    if best:
        best_path = os.path.join(args.out_dir, "best_fewshot.txt")
        with open(best_path, "w", encoding="utf-8") as f:
            f.write(open(best["path"], "r", encoding="utf-8").read())
        print(f"Best few-shot saved to {best_path} (acc={best['acc']:.4f})")


if __name__ == "__main__":
    main()
