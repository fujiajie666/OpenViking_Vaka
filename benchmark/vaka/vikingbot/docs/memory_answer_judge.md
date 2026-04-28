# memory_answer_judge.py 使用说明

这个脚本会基于每道题的 `related_memory_text` 先生成一个回答，再把这个回答与 `standard_answer` 做对比判分，用来区分两类问题：

- **memory_insufficient**：记忆里本来就没有足够信息，答不出来
- **model_incapable**：记忆其实够了，但模型还是没答对

脚本会直接原地更新输入 CSV。

## 前置准备

### 1. 安装依赖

建议在项目环境里运行：

```bash
uv pip install -r res/scripts/requirements.txt
```

至少需要：

- `openai`
- `python-dotenv`

如果缺少 `openai`，脚本会提示：

```bash
请使用项目环境运行，例如: uv run python res/scripts/memory_answer_judge.py
```

### 2. 配置 API Token

脚本默认会从项目根目录 `.env` 中读取：

```env
ARK_API_KEY="你的key"
```

也可以运行时通过 `--token` 显式传入。

## 最常用运行方式

所有命令建议在项目根目录执行。

### 默认运行

```bash
uv run python res/scripts/memory_answer_judge.py
```

默认行为：

- 读取 `res/result_timestamp/stable_wrong_analysis.csv`
- 对尚未填写 `memory_judge_result` 的行进行处理
- 生成三列：
  - `memory_generated_answer`
  - `memory_judge_result`
  - `memory_judge_reasoning`
- 处理过程中持续落盘，最终原地覆盖原 CSV

### 指定输入文件

```bash
uv run python res/scripts/memory_answer_judge.py \
  --input res/result_timestamp/stable_wrong_analysis.csv
```

适合你想处理其它分析 CSV 时使用。

### 强制重跑所有行

```bash
uv run python res/scripts/memory_answer_judge.py --force
```

默认会跳过已经有 `memory_judge_result` 的行；加上 `--force` 后会把所有行重新生成和重判。

### 调整并发数

```bash
uv run python res/scripts/memory_answer_judge.py --parallel 10
```

适合：

- 提高吞吐量
- 或在接口限流时把并发调小，例如 `--parallel 2`

### 指定生成模型和评判模型

```bash
uv run python res/scripts/memory_answer_judge.py \
  --gen-model ep-xxxx \
  --judge-model ep-yyyy
```

适合分别测试：

- 一个模型负责“基于记忆回答”
- 另一个模型负责“判断回答是否与标准答案一致”

### 指定 OpenAI 兼容接口地址和 token

```bash
uv run python res/scripts/memory_answer_judge.py \
  --base-url http://ark.cn-beijing.volces.com/api/v3 \
  --token your_token
```

适合临时切换接口地址，或不想依赖 `.env` 时使用。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | `res/result_timestamp/stable_wrong_analysis.csv` | 输入 CSV 路径 |
| `--base-url` | `http://ark.cn-beijing.volces.com/api/v3` | OpenAI 兼容 API 地址 |
| `--token` | 读取 `ARK_API_KEY` | API token，也可手动传入 |
| `--gen-model` | `ep-20260423162207-qfqr8` | 生成回答时使用的模型 |
| `--judge-model` | `ep-20260423162207-qfqr8` | 评判回答时使用的模型 |
| `--parallel` | `5` | 并发请求数 |
| `--force` | 否 | 即使已有结果也重新处理 |

## 输入 CSV 需要哪些列

脚本核心会读取这些列：

- `question_index`：题号，只用于日志打印
- `question`：问题文本
- `standard_answer`：标准答案
- `related_memory_text`：相关记忆文本

其中最关键的是：

- `question`
- `standard_answer`
- `related_memory_text`

如果输入 CSV 没有这些列，脚本虽然可能能读入，但实际处理结果会异常或为空。

## 输出列说明

脚本会确保 CSV 中存在以下三列，不存在就自动补上：

- `memory_generated_answer`：模型根据 `related_memory_text` 生成的答案
- `memory_judge_result`：判定结果，可能为：
  - `CORRECT`
  - `WRONG`
  - `ERROR`
- `memory_judge_reasoning`：简短中文理由

## 处理流程

每一行大致按下面顺序执行：

1. 用 `question + related_memory_text` 构造回答提示词
2. 调用 `--gen-model` 生成 `memory_generated_answer`
3. 用 `question + standard_answer + generated_answer` 构造评判提示词
4. 调用 `--judge-model` 判断结果是否一致
5. 立即把当前结果写回 CSV，避免中途中断时全部丢失

## 日志怎么看

运行时通常会看到类似输出：

```text
Loading CSV...
  100 rows loaded
  37 rows to process
  [qi=12] Generating answer...
  [qi=12] Judging...
  [qi=12] CORRECT: 生成答案覆盖了标准答案核心信息
```

末尾会输出汇总：

- `CORRECT (有记忆能答对)`
- `WRONG   (有记忆仍答错)`
- `ERROR`
- `记忆足够率`
- `模型不足率`

这里的：

- **记忆足够率** = `CORRECT / (CORRECT + WRONG + ERROR)`
- **模型不足率** = `WRONG / (CORRECT + WRONG + ERROR)`

注意：当前脚本把 `ERROR` 也算进总计里了，所以这两个比例是“按全部已处理结果”统计的。

## 常见场景示例

### 1. 首次跑默认数据集

```bash
uv run python res/scripts/memory_answer_judge.py
```

### 2. 重新评估整份 CSV

```bash
uv run python res/scripts/memory_answer_judge.py \
  --input res/result_timestamp/stable_wrong_analysis.csv \
  --force
```

### 3. 用一个模型回答、另一个模型做 judge

```bash
uv run python res/scripts/memory_answer_judge.py \
  --gen-model ep-answer-model \
  --judge-model ep-judge-model
```

### 4. 降低并发，减少接口压力

```bash
uv run python res/scripts/memory_answer_judge.py --parallel 2
```

## 常见报错

### 没有 token

如果看到：

```text
Error: API token is required
请设置 ARK_API_KEY 或 API_KEY 环境变量，或通过 --token 参数传入
```

处理方式：

- 在 `.env` 中配置 `ARK_API_KEY`
- 或运行时加 `--token xxx`

### 没装 openai 包

如果看到：

```text
Error: openai package is required.
```

处理方式：

```bash
uv pip install -r res/scripts/requirements.txt
```

或直接使用项目环境：

```bash
uv run python res/scripts/memory_answer_judge.py
```

### Judge 返回 `ERROR`

通常表示：

- 接口调用失败
- 返回内容里没有合法 JSON
- 超时或限流

可以尝试：

- 降低并发：`--parallel 2`
- 重新跑：`--force`
- 切换 `--judge-model`
- 检查 `--base-url` 和 `--token`

## 补充说明

- 这个脚本会**原地覆盖**输入 CSV，运行前如果你想保留旧结果，先手动备份。
- 处理过程中每行结束都会写一次临时文件再替换原文件，所以中断后通常不会整份损坏。
- `BRIEF_SUFFIX` 当前为空，因此生成答案时没有额外强制“简短回答”的约束。
