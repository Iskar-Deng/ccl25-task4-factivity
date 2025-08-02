from typing import Optional, Dict, Any
import os, yaml
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

def _load_model_configs() -> Dict[str, Any]:
    here = os.path.dirname(__file__)
    cfg_path = os.path.join(here, "../../models/model_configs.yaml")
    cfg_path = os.path.normpath(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        all_cfg = yaml.safe_load(f)
    return all_cfg["models"]

def _get_chat(model_key: str, temperature: Optional[float], max_tokens: Optional[int]) -> ChatOpenAI:
    cfgs = _load_model_configs()
    if model_key not in cfgs:
        raise ValueError(f"Model '{model_key}' not found in models/model_configs.yaml")
    c = cfgs[model_key]
    api_key = c.get("api_key") or os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or ""
    base_url = c["base_url"]
    model_name = c["model_name"]
    temp = temperature if temperature is not None else c.get("temperature", 0.0)
    max_toks = max_tokens if max_tokens is not None else c.get("max_tokens", 1024)
    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name,
                      temperature=float(temp), max_tokens=int(max_toks))

def call_model(model_name: str, user_input: str,
               temperature: Optional[float]=None, max_tokens: Optional[int]=None,
               return_logprobs: bool=False) -> str:
    llm = _get_chat(model_name, temperature, max_tokens)
    msgs = [SystemMessage(content="你是一个小心谨慎的中文NLP评测助手。回答请只输出文本，不要多余格式。"),
            HumanMessage(content=user_input)]
    resp = llm.invoke(msgs)
    return (resp.content or "").strip()
