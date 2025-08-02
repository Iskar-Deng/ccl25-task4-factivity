#!/usr/bin/env python
import os, sys, json, time, argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.api.run_model import call_model
from src.utils.prompt_loader import load_prompt
from src.utils.helpers import fmt, extract_label

def run_single(item, prompt, model_name, temperature=0.2):
    filled_prompt = fmt(prompt, item["text"], item["hypothesis"], item.get("predicate",""), item.get("type",""))
    try:
        start = time.time()
        response = call_model(model_name, filled_prompt, temperature).strip()
        duration = round(time.time() - start, 3)
    except Exception as e:
        response = f"[ERROR] {str(e)}"
        duration = 0.0
    label = extract_label(response)
    return {"d_id": item["d_id"], "answer": label, "raw": response, "time": duration}

def main():
    parser = argparse.ArgumentParser(description="ART evaluation by type")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--prompt_T", type=str, required=True)
    parser.add_argument("--prompt_F", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_workers", type=int, default=8)
    args = parser.parse_args()

    raw_data = json.load(open(args.data_path, "r", encoding="utf-8"))
    data = [{
        "d_id": d["d_id"],
        "predicate": d.get("predicate",""),
        "text": d["text"],
        "hypothesis": d["hypothesis"],
        "type": d.get("type","")
    } for d in raw_data]

    prompt_T = load_prompt(args.prompt_T)
    prompt_F = load_prompt(args.prompt_F)

    print(f"✅ Loaded {len(data)} examples.")
    results = [None] * len(data)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for idx, item in enumerate(data):
            if item["type"] == "正叙实":
                futures[executor.submit(run_single, item, prompt_T, args.model_name, args.temperature)] = idx
            elif item["type"] == "反叙实":
                futures[executor.submit(run_single, item, prompt_F, args.model_name, args.temperature)] = idx
            elif item["type"] == "非叙实":
                results[idx] = {"d_id": item["d_id"], "answer": "U", "raw": "Skipped (非叙实, auto U)", "time": 0.0}
            else:
                raise ValueError(f"Unknown type: {item['type']} in d_id: {item['d_id']}")
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Eval"):
            results[futures[fut]] = fut.result()

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = args.output_name or "Art_run"
    save_path = os.path.join(args.output_dir, f"{output_name}.json")
    json.dump(results, open(save_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved simplified results to {save_path}")

if __name__ == "__main__":
    main()
