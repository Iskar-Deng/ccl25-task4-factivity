#!/usr/bin/env python
import os, sys, json, time, argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.api.run_model import call_model
from src.utils.prompt_loader import load_prompt
from src.utils.helpers import fmt, extract_label

def safe_call(model: str, prompt: str, temp: float) -> str:
    try:
        return call_model(model, prompt, temp).strip()
    except Exception as e:
        if "data_inspection_failed" in str(e):
            return "[BLOCKED]"
        raise

def run_two_stage_with_branching(item, p1, p2, p3, model, temp):
    out = {"d_id": item.get("d_id")}
    if "answer" in item:
        out["gold"] = item["answer"]

    t0 = time.time()
    rsp1 = safe_call(model, fmt(p1, item["text"], item["hypothesis"], item.get("predicate","")), temp)
    t1 = time.time()
    lab1 = "R" if rsp1 == "[BLOCKED]" else extract_label(rsp1)

    if rsp1 == "[BLOCKED]":
        out.update(dict(stage1_pred="R", pred="R",
                        raw_stage1=rsp1, raw_stage2="",
                        time_stage1=round(t1 - t0, 3), time_stage2=0.0))
        return out

    out.update({
        "stage1_pred": lab1,
        "raw_stage1": rsp1,
        "time_stage1": round(t1 - t0, 3)
    })

    if lab1 == "U" and p2:
        rsp2 = safe_call(model, fmt(p2, item["text"], item["hypothesis"], item.get("predicate","")), temp)
        t2 = time.time()
        lab2 = extract_label(rsp2)
    elif lab1 in {"T","F"} and p3:
        filled3 = fmt(p3, item["text"], item["hypothesis"], item.get("predicate","")).replace("{first_label}", lab1)
        rsp2 = safe_call(model, filled3, temp)
        t2 = time.time()
        try:
            lab2 = json.loads(rsp2).get("final", lab1).upper()
        except Exception:
            lab2 = lab1
    else:
        rsp2, lab2, t2 = "", lab1, t1

    out.update({
        "stage2_pred": lab2 if rsp2 else None,
        "pred": lab2,
        "raw_stage2": rsp2,
        "time_stage2": round(t2 - t1, 3) if rsp2 else 0.0
    })
    return out

def main():
    ap = argparse.ArgumentParser(description="Two-stage pipeline (nat)")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--prompt1", required=True)
    ap.add_argument("--prompt2", required=True)
    ap.add_argument("--prompt3", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output_dir", default="results")
    ap.add_argument("--output_name")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_workers", type=int, default=8)
    args = ap.parse_args()

    data = json.load(open(args.data_path, "r", encoding="utf-8"))
    p1 = load_prompt(args.prompt1)
    p2 = load_prompt(args.prompt2) if args.prompt2 else None
    p3 = load_prompt(args.prompt3) if args.prompt3 else None

    out_name = args.output_name or f"{os.path.splitext(os.path.basename(args.prompt1))[0]}__2stage__{args.model.replace('/', '_')}"
    results = [None] * len(data)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(run_two_stage_with_branching, item, p1, p2, p3, args.model, args.temperature): i
                   for i, item in enumerate(data)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="2-stage eval"):
            results[futures[fut]] = fut.result()

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"{out_name}.json")
    json.dump(results, open(save_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Simplified results saved to {save_path}")

if __name__ == "__main__":
    main()
