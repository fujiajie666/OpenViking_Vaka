# VikingBot 单题测试工具（test_per_query）

逐条测试 VikingBot 对 `wrong_answers_with_memories.csv` 中问题的回答，并可选调用 LLM 判官评估正确性。

## 工作流程

1. 按 `question_index` 从输入 CSV 加载指定问题
2. 调用 VikingBot Chat API 获取回答
3. 提取回答中命中的检索记忆（query_memory + LLM tool_result memories）
4. 可选：调用 LLM 判官对比标准答案，判定 CORRECT / WRONG
5. 将结果（回答、正确性、判官推理、检索记忆）追加/更新到结果 CSV

## 快速开始

```bash
# 最简用法：测试第 42 题（需本地运行 VikingBot，默认 http://localhost:1933）
python benchmark/vaka/vikingbot/scripts/test_per_query.py 42

# 跳过判官，仅打印回答
python benchmark/vaka/vikingbot/scripts/test_per_query.py 42 --no-judge

# 指定 VikingBot 地址和 API Key
python benchmark/vaka/vikingbot/scripts/test_per_query.py 42 \
  --openviking-url http://your-server:1933 \
  --api-key your-key

# 启用判官（需设置 ARK_API_KEY 或通过 --token 传入）
python benchmark/vaka/vikingbot/scripts/test_per_query.py 42 --token sk-xxx
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `question_index` | 位置参数 | — | 要测试的问题序号 |
| `--input` | str | `data/wrong_answers_with_memories.csv` | 输入 CSV 路径 |
| `--openviking-url` | str | `http://localhost:1933` | VikingBot 服务地址 |
| `--user-id` | str | `default` | OpenViking user_id |
| `--account` | str | `default` | OpenViking account |
| `--api-key` | str | — | VikingBot API Key |
| `--base-url` | str | `https://ark.cn-beijing.volces.com/api/v3` | 判官 LLM API 地址 |
| `--token` | str | `$ARK_API_KEY` 或 `$OPENAI_API_KEY` | 判官 API Token |
| `--model` | str | `ep-20260423162207-qfqr8` | 判官模型名称 |
| `--no-judge` | flag | — | 跳过 LLM 判官，仅输出回答 |
| `--output` | str | `analysis/question_result/temp.csv` | 结果 CSV 保存路径 |
| `--force` | flag | — | 强制覆盖已有结果（默认跳过已测试的问题） |

## 环境变量

| 变量 | 用途 |
|------|------|
| `ARK_API_KEY` | 判官 API Token（优先级高于 `OPENAI_API_KEY`） |
| `OPENAI_API_KEY` | 判官 API Token（备选） |

可通过 `~/.openviking_benchmark_env` 文件自动加载。

## 结果 CSV 字段

| 字段 | 说明 |
|------|------|
| `question_index` | 问题序号 |
| `question` | 原始问题 |
| `response` | VikingBot 回答 |
| `is_correct` | `CORRECT` / `WRONG` / 空（未判官时为空） |
| `reasoning` | 判官的简短中文解释 |
| `time_cost` | 请求耗时（秒） |
| `retrieved_memories_json` | 检索记忆 JSON（含 query_memory 和 llm_memory） |
| `retrieved_memories_text` | 检索记忆可读文本（score + uri + abstract） |
