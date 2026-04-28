# enrich_wrong_analysis.py 使用说明

为错题分析 CSV 补充或刷新两列信息：

- **`related_memory_text`** — 该题真正相关的原始对话证据
- **`standard_answer_reasonable`** — LLM 判断该题的 `standard_answer` 是否合理（REASONABLE / UNREASONABLE + 理由）

`related_memory_text` 现在只基于 `evidence` 生成，不再混入 `reference_memory`、`notes`、`reference_dialogue` 这类摘要或说明字段。

## 前置准备

### 1. 安装依赖

```bash
uv pip install -r res/scripts/requirements.txt
```

需要 `python-dotenv` 和 `openai` 两个包。前者读 `.env`，后者调 LLM。

### 2. 配置 API Key（仅 LLM 判断步骤需要）

在项目根目录创建 `.env` 文件：

```
API_KEY="你的key"
```

如果跳过 LLM 判断（`--skip-judge`），则不需要配置。

## 只更新 `related_memory_text` 列

如果你只想刷新某个 CSV 里的 `related_memory_text`，不改动别的列，使用：

```bash
python3 res/scripts/enrich_wrong_analysis.py --wrong-csv res/result_timestamp/wrong_answers_analysis.csv --skip-judge
```

这个命令会：

1. 读取 `wrong_answers_analysis.csv`
2. 按每行的 `question_index` 到 `vaka_qa_result_withs.csv` 找到对应的 `evidence`
3. 把 `evidence` 里的 `d...s...r...` 解析成 `(session_id, round)`
4. 到 `vaka_locomo.csv` 取这些轮次对应的：
   - `query`
   - `deepsearch_answer`
5. 用这些原始对话片段重写 `related_memory_text`

不会触碰 `standard_answer_reasonable` 的已有值，也不会重新做 LLM 判断。

## 只更新 `standard_answer_reasonable` 列

如果你只需修订 LLM 合理性判断，保留已有的 `related_memory_text` 不变，使用：

```bash
python3 res/scripts/enrich_wrong_analysis.py --wrong-csv res/result_timestamp/wrong_answers_analysis.csv --skip-memory
```

搭配 `--force` 可以强制重新判断所有行（包括已有结果的）：

```bash
python3 res/scripts/enrich_wrong_analysis.py --wrong-csv res/result_timestamp/wrong_answers_analysis.csv --skip-memory --force
```

## 运行

所有命令在项目根目录下执行：

```bash
cd /Users/bytedance/Desktop/vaka_viking
```

### 仅填充记忆原文，不调 LLM

```bash
python3 res/scripts/enrich_wrong_analysis.py --skip-judge
```

快速完成，不需要 API Key。只生成或刷新 `related_memory_text` 列。

### 完整运行（记忆原文 + LLM 合理性判断）

```bash
python3 res/scripts/enrich_wrong_analysis.py
```

需要 `.env` 中配置 `API_KEY`，或通过 `--token` 传入。

### 强制重新判断已有结果

```bash
python3 res/scripts/enrich_wrong_analysis.py --force
```

默认跳过已填写 `standard_answer_reasonable` 的行，`--force` 会重新判断所有行。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--wrong-csv` | `res/result_no_prefetch/stable_wrong_analysis.csv` | 待补充或刷新的错题分析 CSV |
| `--locomo-csv` | `res/result_no_prefetch/vaka_locomo.csv` | 原始对话数据 |
| `--withs-csv` | `OpenViking/benchmark/vaka/vikingbot/result/vaka_qa_result_withs.csv` | 提供 `question_index` 到 `evidence` 的映射 |
| `--base-url` | `https://ark.cn-beijing.volces.com/api/coding` | LLM API 地址 |
| `--token` | 从 `.env` 读 `API_KEY` | API Key，优先级高于 `.env` |
| `--model` | `glm-5.1` | 判断用的模型 |
| `--parallel` | `5` | 并发请求数 |
| `--skip-judge` | 否 | 跳过 LLM 合理性判断，只刷新 `related_memory_text` |
| `--skip-memory` | 否 | 跳过 `related_memory_text` 填充，只运行 LLM 合理性判断 |
| `--force` | 否 | 强制重新判断所有行（包括已有结果的） |

## 输出

脚本直接原地更新目标 CSV，不会额外生成新文件。

| 列名 | 新方式 |
|------|--------|
| `related_memory_text` | 仅包含 `evidence` 对应到的原始对话：`query` + `deepsearch_answer` |
| `standard_answer_reasonable` | 仅在未加 `--skip-judge` 时更新 |

## 数据来源说明

`related_memory_text` 的新生成方式只使用这条链路：

- `wrong_answers_analysis.csv` / `stable_wrong_analysis.csv` 中的 `question_index`
- `vaka_qa_result_withs.csv` 中对应行的 `evidence`
- `vaka_locomo.csv` 中对应 `(session_id, round)` 的 `query` 和 `deepsearch_answer`

如果 `evidence` 为空，或者映射不到任何原始对话，则 `related_memory_text` 会写成 `[无相关记忆数据]`。
