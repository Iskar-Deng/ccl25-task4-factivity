# FIE2025 事实性推理评测流程

本仓库提供了一个关于 [FIE2025](https://github.com/UM-FAH-Yuan/FIE2025) 任务的简单事实性推理评测流程。  
支持通过 LangChain 调用 OpenAI 兼容的 LLM 接口（例如 Qwen、DeepSeek）。

---

## 1. 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

编辑 `models/model_configs.yaml` 文件，填入你的 API Key。

---

## 2. 下载数据集

从 [https://github.com/UM-FAH-Yuan/FIE2025](https://github.com/UM-FAH-Yuan/FIE2025) 下载并放置到 `data/` 目录下。

---

## 3. 运行流程

### 3.1 NAT 数据集（两阶段推理）

```bash
python scripts/run_two_stage_nat.py \
  --data_path data/natural.json \
  --prompt1 prompts/stage1_only_predicate.txt \
  --prompt2 prompts/stage2_read_full_upgrade.txt \
  --prompt3 prompts/stage2_read_full_downgrade.txt \
  --model Qwen-Max \
  --output_dir results
```

### 3.2 ART 数据集（按类型直接推理）

```bash
python scripts/run_art_by_type.py \
  --data_path data/artificial.json \
  --prompt_T prompts/art_prompt_T.txt \
  --prompt_F prompts/art_prompt_F.txt \
  --model_name Qwen-Max \
  --output_dir results
```

---

## 4. Auto-CoT 提示优化流程

### 4.1 生成 CoT 演示样例

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

输出文件：
- `autocot/outputs/demos_upgrade.json`
- `autocot/outputs/fewshot_cot_upgrade.txt`

---

### 4.2 评估并选择最佳 few-shot 提示

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

输出文件：
- `autocot/selection/best_fewshot.txt` – 自动选择出的最佳 few-shot 提示

---

### 4.3 使用优化后的 few-shot 再次推理

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

## 5. 输出格式

结果保存到 `results/*.json` 文件中，每条记录包含：
- `d_id`: 样本 ID
- `gold`: 标注标签（如果有）
- `pred`: 模型最终预测
- `stage1_pred`, `stage2_pred`: 两阶段的中间预测结果
- `raw_stage1`, `raw_stage2`: 模型原始输出

---
