#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_langchain_call.py
A tiny connectivity test to verify your model config and LangChain call work.

Usage:
  python scripts/test_langchain_call.py --model Qwen-Max --prompt "测试一下"
Options:
  --temperature 0.2     # optional
  --max_tokens 256      # optional
"""
import os, sys, argparse

# Ensure project root is importable: <repo_root>/src/...
THIS = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api.run_model import call_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='Model key in models/model_configs.yaml (e.g., Qwen-Max)')
    ap.add_argument('--prompt', required=True, help='Prompt text to send')
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--max_tokens', type=int, default=None)
    args = ap.parse_args()

    print('== Request ==')
    print('Model    :', args.model)
    print('Temp     :', args.temperature)
    print('MaxTokens:', args.max_tokens)
    print('Prompt   :', args.prompt)
    print('')

    try:
        resp = call_model(args.model, args.prompt, temperature=args.temperature, max_tokens=args.max_tokens)
        print('== Response ==')
        print(resp)
    except Exception as e:
        print('!! Error calling model:', e)
        print('Hint: check models/model_configs.yaml (api_key, base_url, model_name) or environment variables.')

if __name__ == '__main__':
    main()
