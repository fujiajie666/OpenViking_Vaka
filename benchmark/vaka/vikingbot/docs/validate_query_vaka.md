# validate_query_vaka.py 使用指引

## 目的

验证 `vaka_judge.csv` 中的 query 是否合理：给定完整对话上下文（session_id 1-70），大模型能否正确回答该 query 并匹配 gold answer。

如果大量 query 回答错误，说明问题本身可能不合理（上下文中没有足够证据），需要修正或剔除。

## 流程

支持两种模式（`--mode`）：

### full 模式（默认）

1. 从 `vaka_locomo.csv` 加载 session_id 1-70 的对话数据，按时间顺序拼接成完整上下文
2. 对 `vaka_judge.csv` 中每个 query，拼接 prompt：`上下文 + "基于以上对话历史，回答问题：{query}"`
3. 调用大模型生成答案（单模型）
4. 将生成答案与 `standard_answer` 对比，用 3 模型 ensemble 多数投票判断 CORRECT/WRONG
5. 结果保存到 CSV，支持断点续跑

### evidence 模式

1. 从 `vaka_locomo.csv` 加载 `complete_evidence_dialogue_with_model` 列，按 `item_id` 建立映射
2. 对 `vaka_judge.csv` 中第 i 个 query（0-based），使用 `item_id = i + 1` 对应的 evidence 作为上下文
3. 拼接 prompt：`evidence 上下文 + "基于以上对话历史，回答问题：{query}"`
4. 后续步骤同 full 模式（生成答案 → ensemble 评判）

evidence 模式使用的是每道题对应的精简证据上下文，而非全部 70 个 session 的对话，可以验证 gold answer 是否能从 evidence 中正确回答。

## 时间映射规则

- session_id 1-10 → 2025年8月1日 (Day 1)，Session 1-10
- session_id 11-20 → 2025年8月2日 (Day 2)，Session 1-10
- 以此类推：Day = (session_id - 1) // 10 + 1，Session = (session_id - 1) % 10 + 1

## 运行方式

```bash
# 基本运行（断点续跑，已有结果的不会重新处理）
uv run python benchmark/vaka/vikingbot/scripts/validate_query_vaka.py

# evidence 模式：使用每道题对应的 evidence 上下文
uv run python benchmark/vaka/vikingbot/scripts/validate_query_vaka.py --mode evidence

# 只测试某一道题（query_index 从 0 开始，对应 CSV 中的第几行）
uv run python benchmark/vaka/vikingbot/scripts/validate_query_vaka.py --query-index 0

# 只测试多道题
uv run python benchmark/vaka/vikingbot/scripts/validate_query_vaka.py --query-index 0 5 12

# 强制重新处理所有 query
uv run python benchmark/vaka/vikingbot/scripts/validate_query_vaka.py --force

# 自定义并发数
uv run python benchmark/vaka/vikingbot/scripts/validate_query_vaka.py --parallel 5

# 自定义模型
uv run python benchmark/vaka/vikingbot/scripts/validate_query_vaka.py \
  --answer-model ep-20260423162207-qfqr8 \
  --judge-models ep-20260423162207-qfqr8 ep-20260501104936-72vfz ep-20260501105042-9kp5v

# 指定输出路径
uv run python benchmark/vaka/vikingbot/scripts/validate_query_vaka.py --output /path/to/output.csv
```

## API Key 配置

与 `judge.py` 相同，优先级：

1. `--token` 参数
2. `~/.openviking_benchmark_env` 中的 `ARK_API_KEY`
3. 环境变量 `ARK_API_KEY` 或 `OPENAI_API_KEY`

## 输出

默认输出到 `benchmark/vaka/vikingbot/result/vaka_query_validation.csv`，包含以下列：

| 列名 | 说明 |
|------|------|
| query_index | 问题序号 |
| query | 原始问题 |
| standard_answer | 标准答案 |
| generated_answer | 大模型生成的答案 |
| is_correct | CORRECT 或 WRONG（3 模型 ensemble 结果） |
| is_correct_doubao | 豆包模型评判结果：CORRECT 或 WRONG |
| is_correct_glm | GLM 模型评判结果：CORRECT 或 WRONG |
| is_correct_deepseek | DeepSeek 模型评判结果：CORRECT 或 WRONG |
| reasoning | 判决理由（中文） |
| answer_input_tokens | 生成答案的输入 token 数 |
| answer_output_tokens | 生成答案的输出 token 数 |
| judge_input_tokens | 评判的输入 token 数 |
| judge_output_tokens | 评判的输出 token 数 |

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--locomo` | `data/vaka_locomo.csv` | 对话数据路径 |
| `--judge-csv` | `data/vaka_judge.csv` | 问题数据路径 |
| `--output` | `result/vaka_query_validation.csv` | 输出路径 |
| `--base-url` | `https://ark.cn-beijing.volces.com/api/v3` | API 地址 |
| `--answer-model` | `ep-20260423162207-qfqr8` | 生成答案的模型 |
| `--judge-models` | 3 个 endpoint | 评判模型（ensemble 多数投票） |
| `--parallel` | 3 | 并发请求数 |
| `--max-context-chars` | 80000 | 上下文最大字符数 |
| `--mode` | `full` | 上下文模式：`full` 使用全部 session 1-70 对话；`evidence` 使用 locomo 中的 `complete_evidence_dialogue_with_model` 列作为每题的精简上下文 |
| `--query-index` | None | 只处理指定的问题序号（0-based），如 `--query-index 0 5 12` |
| `--force` | False | 强制重新处理已有结果 |

## 如何使用结果

- **full 模式 accuracy 高**：大部分 query 能从上下文中正确回答，说明问题质量好
- **full 模式 accuracy 低**：检查 WRONG 的 query，可能原因：
  - 上下文中缺少足够证据支持该问题
  - gold answer 表述有误
  - 问题本身模糊或有歧义
- **evidence 模式**：验证 gold answer 是否能从精简 evidence 上下文中正确回答。如果 evidence 模式正确但 full 模式错误，说明 evidence 提取有效但完整对话过长导致信息遗漏
- 关注 `reasoning` 列可定位具体问题
