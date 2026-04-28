"""Given wrong-analysis CSV with question + related_memory_text + standard_answer,
generate an answer using memory context, then judge if it matches the standard answer.

This helps distinguish:
  - memory_insufficient: the memory doesn't contain enough info to answer
  - model_incapable: memory is sufficient but the model still can't answer correctly
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv(*args, **kwargs) -> bool:
        return False

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


SCRIPT_DIR = Path(__file__).parent.resolve()
RES_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = RES_DIR.parent
DEFAULT_INPUT = str(RES_DIR / "result_timestamp" / "bad_questions.csv")

load_dotenv(PROJECT_ROOT / ".env")

BRIEF_SUFFIX = "" #\n请尽量简短作答，只回答与问题直接相关的内容，不要展开无关信息，但确保不遗漏问题要求的关键信息。


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def load_csv(path: str) -> tuple[list[dict], list[str]]:
    raise_csv_field_limit()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for col in ["memory_generated_answer", "memory_judge_result", "memory_judge_reasoning"]:
        if col not in fieldnames:
            fieldnames.append(col)
    return rows, fieldnames


def build_answer_prompt(question: str, related_memory_text: str) -> str:
    return f"""根据以下记忆内容回答问题。

记忆内容：
{related_memory_text or "[无相关记忆]"}

问题：
{question}{BRIEF_SUFFIX}"""


def build_judge_prompt(question: str, standard_answer: str, generated_answer: str) -> str:
    return f"""你是一个评估助手。请判断 GENERATED_ANSWER 是否与 GOLD_ANSWER 在关键信息上一致。

要求宽松对待措辞差异，但严格判断核心事实是否正确、是否遗漏关键信息。

如果 GENERATED_ANSWER 包含了 GOLD_ANSWER 的核心要点，且没有与 GOLD_ANSWER 矛盾的内容，判定为 CORRECT。
如果 GENERATED_ANSWER 遗漏了关键信息或包含矛盾，判定为 WRONG。

QUESTION:
{question}

GOLD_ANSWER:
{standard_answer}

GENERATED_ANSWER:
{generated_answer}

Return JSON only:
{{"is_correct": "CORRECT" or "WRONG", "reasoning": "一句简短的中文解释"}}"""


def extract_json_object(content: str) -> dict:
    start_idx = content.find("{")
    end_idx = content.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        raise ValueError(f"No JSON object found in response: {content}")
    return json.loads(content[start_idx : end_idx + 1])


async def generate_answer(
    client: AsyncOpenAI,
    *,
    model: str,
    question: str,
    related_memory_text: str,
) -> str:
    prompt = build_answer_prompt(question, related_memory_text)
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=120,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        return f"[GENERATE ERROR] {exc}"


async def judge_answer(
    client: AsyncOpenAI,
    *,
    model: str,
    question: str,
    standard_answer: str,
    generated_answer: str,
) -> tuple[str, str]:
    prompt = build_judge_prompt(question, standard_answer, generated_answer)
    system_prompt = "你是一个严格但公正的答案评估专家。"
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            timeout=60,
        )
        content = (resp.choices[0].message.content or "").strip()
        result = extract_json_object(content)
        is_correct = str(result.get("is_correct", "WRONG")).strip().upper() == "CORRECT"
        reasoning = str(result.get("reasoning", "")).strip()
        return "CORRECT" if is_correct else "WRONG", reasoning
    except Exception as exc:
        return "ERROR", f"[JUDGE ERROR] {exc}"


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate answers from memory and judge against standard answers"
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to stable_wrong_analysis.csv, default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--base-url",
        default="http://ark.cn-beijing.volces.com/api/v3",
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("ARK_API_KEY"),
        help="API token, default from ARK_API_KEY or API_KEY env var",
    )
    parser.add_argument(
        "--gen-model",
        default="ep-20260423162207-qfqr8",
        help="Model for generating answers, default: doubao-seed-2.0-pro",
    )
    parser.add_argument(
        "--judge-model",
        default="ep-20260423162207-qfqr8",
        help="Model for judging answers, default: doubao-seed-2.0-pro",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=10,
        help="Parallel request count, default: 5",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process rows even when result is already present",
    )
    args = parser.parse_args()

    if not args.token:
        print("Error: API token is required")
        print("请设置 ARK_API_KEY 或 API_KEY 环境变量，或通过 --token 参数传入")
        raise SystemExit(1)

    if AsyncOpenAI is None:
        print("Error: openai package is required.")
        print("请使用项目环境运行，例如: uv run python res/scripts/memory_answer_judge.py")
        raise SystemExit(1)

    print("Loading CSV...")
    rows, fieldnames = load_csv(args.input)
    print(f"  {len(rows)} rows loaded")

    target_rows = [
        (i, row)
        for i, row in enumerate(rows)
        if args.force or not (row.get("memory_judge_result") or "").strip()
    ]
    print(f"  {len(target_rows)} rows to process")

    if not target_rows:
        print("All rows already processed, exit")
        return

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.token)
    semaphore = asyncio.Semaphore(args.parallel)
    file_lock = asyncio.Lock()

    async def save_results() -> None:
        async with file_lock:
            temp_file = f"{args.input}.tmp"
            with open(temp_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temp_file, args.input)

    async def process_row(idx: int, row: dict) -> None:
        async with semaphore:
            qi = row.get("question_index", "")
            question = row.get("question", "")
            standard_answer = row.get("standard_answer", "")
            related_memory_text = row.get("related_memory_text", "")

            print(f"  [qi={qi}] Generating answer...")
            generated = await generate_answer(
                client,
                model=args.gen_model,
                question=question,
                related_memory_text=related_memory_text,
            )
            row["memory_generated_answer"] = generated

            print(f"  [qi={qi}] Judging...")
            result, reasoning = await judge_answer(
                client,
                model=args.judge_model,
                question=question,
                standard_answer=standard_answer,
                generated_answer=generated,
            )
            row["memory_judge_result"] = result
            row["memory_judge_reasoning"] = reasoning

            await save_results()
            print(f"  [qi={qi}] {result}: {reasoning[:60]}")

    await asyncio.gather(*(process_row(idx, row) for idx, row in target_rows))

    # Final save + summary
    temp_file = f"{args.input}.tmp"
    with open(temp_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_file, args.input)

    correct = sum(1 for r in rows if r.get("memory_judge_result") == "CORRECT")
    wrong = sum(1 for r in rows if r.get("memory_judge_result") == "WRONG")
    error = sum(1 for r in rows if r.get("memory_judge_result") == "ERROR")
    total_judged = correct + wrong + error

    print(f"\nDone. Results saved to {args.input}")
    print(f"  CORRECT (有记忆能答对): {correct}")
    print(f"  WRONG   (有记忆仍答错): {wrong}")
    print(f"  ERROR:  {error}")
    if total_judged:
        print(f"  记忆足够率: {correct / total_judged:.1%} (有记忆能答对 / 总计)")
        print(f"  模型不足率: {wrong / total_judged:.1%} (有记忆仍答错 / 总计)")


if __name__ == "__main__":
    asyncio.run(main())
