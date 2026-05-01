from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

import httpx

try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv(*args, **kwargs) -> bool:
        return False


SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_INPUT = str(SCRIPT_DIR.parent / "result_timestamp" / "wrong_answers_analysis.csv")
DEFAULT_OUTPUT = str(SCRIPT_DIR.parent / "result_timestamp" / "wrong_answers_with_memories.csv")
DEFAULT_USER_ID = "default"
DEFAULT_AGENT_ID = "default"
DEFAULT_ACCOUNT = "default"

load_dotenv(Path.home() / ".openviking_benchmark_env")


def raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


async def chat_with_bot(
    question: str,
    *,
    openviking_url: str,
    session_id: str = "default",
    user_id: str | None = None,
    account: str | None = None,
    agent_id: str | None = None,
    api_key: str | None = None,
) -> tuple[dict, float]:
    url = f"{openviking_url.rstrip('/')}/bot/v1/chat"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    if account:
        headers["X-OpenViking-Account"] = account
    if user_id:
        headers["X-OpenViking-User"] = user_id
    if agent_id:
        headers["X-OpenViking-Agent"] = agent_id

    body = {
        "message": question,
        "session_id": session_id,
        "stream": False,
    }
    if user_id:
        body["user_id"] = user_id
    if agent_id:
        body["agent_id"] = agent_id

    start_time = time.time()
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(url, json=body, headers=headers)
    time_cost = time.time() - start_time

    if response.status_code != 200:
        return {
            "message": f"[HTTP ERROR] status={response.status_code}, body={response.text[:200]}",
            "relevant_memories": "",
        }, time_cost

    try:
        data = response.json()
        if not isinstance(data, dict):
            return {
                "message": f"[INVALID RESPONSE] {str(data)[:200]}",
                "relevant_memories": "",
            }, time_cost
        return data, time_cost
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "message": f"[PARSE ERROR] {str(exc)}: {response.text[:200]}",
            "relevant_memories": "",
        }, time_cost


def load_rows(input_path: str) -> tuple[list[dict], list[str]]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raise_csv_field_limit()
    with open(input_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "retrieved_memories_json" not in fieldnames:
        fieldnames.append("retrieved_memories_json")
    if "retrieved_memories_text" not in fieldnames:
        fieldnames.append("retrieved_memories_text")

    return rows, fieldnames


def _extract_memories_from_payload(payload: object) -> list[dict]:
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return []
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return []

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):
        memories = payload.get("memories")
        if isinstance(memories, list):
            return [item for item in memories if isinstance(item, dict)]

    return []


def extract_query_memory(data: dict) -> str:
    val = data.get("relevant_memories")
    if isinstance(val, str):
        return val
    return ""


def extract_llm_memories(data: dict) -> list[dict]:
    memories: list[dict] = []

    events = data.get("events")
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict) or event.get("type") != "tool_result":
                continue
            memories.extend(_extract_memories_from_payload(event.get("data")))

    deduped: list[dict] = []
    seen: set[str] = set()
    for memory in memories:
        key = memory.get("uri")
        if not isinstance(key, str) or not key:
            key = json.dumps(memory, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(memory)

    return deduped


def build_memories_text(query_memory: str, llm_memories: list[dict]) -> str:
    parts: list[str] = []
    if query_memory.strip():
        parts.append(query_memory.strip())
    if llm_memories:
        lines: list[str] = []
        for memory in llm_memories:
            uri = str(memory.get("uri") or "")
            score = memory.get("score")
            score_text = f"{float(score):.6f}" if isinstance(score, (int, float)) else ""
            abstract = str(memory.get("abstract") or "").replace("\n", " ").strip()

            entry_parts = []
            if score_text:
                entry_parts.append(f"[{score_text}]")
            if uri:
                entry_parts.append(uri)
            if abstract:
                entry_parts.append(abstract)

            if entry_parts:
                lines.append(" | ".join(entry_parts))
        if lines:
            parts.append("\n".join(lines))

    return "\n\n".join(parts)


async def enrich_row(
    row: dict,
    *,
    openviking_url: str,
    session_prefix: str,
    index: int,
    user_id: str | None,
    account: str | None,
    agent_id: str | None,
    api_key: str | None,
    semaphore: asyncio.Semaphore,
) -> dict:
    question = (row.get("question") or "").strip()
    if not question:
        row["retrieved_memories_json"] = ""
        row["retrieved_memories_text"] = ""
        return row

    session_id = f"{session_prefix}_{index}"

    async with semaphore:
        data, _ = await chat_with_bot(
            question,
            openviking_url=openviking_url,
            session_id=session_id,
            user_id=user_id,
            account=account,
            agent_id=agent_id,
            api_key=api_key,
        )

    query_memory = extract_query_memory(data)
    llm_memories = extract_llm_memories(data)
    row["retrieved_memories_json"] = json.dumps(
        {"query_memory": query_memory, "llm_memory": llm_memories},
        ensure_ascii=False,
    )
    row["retrieved_memories_text"] = build_memories_text(query_memory, llm_memories)
    return row


async def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch retrieved memories for wrong answer CSV")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Input CSV path, default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path, default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--openviking-url",
        default="http://localhost:1933",
        help="OpenViking base URL, default: http://localhost:1933",
    )
    parser.add_argument(
        "--session-prefix",
        default="get_mem",
        help="Session id prefix, default: get_mem",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=5,
        help="Parallel request count, default: 5",
    )
    parser.add_argument(
        "--user-id",
        default=os.getenv("OPENVIKING_USER", DEFAULT_USER_ID),
        help="X-OpenViking-User / user_id, default: env OPENVIKING_USER or 'default'",
    )
    parser.add_argument(
        "--account",
        default=os.getenv("OPENVIKING_ACCOUNT", DEFAULT_ACCOUNT),
        help="X-OpenViking-Account, default: env OPENVIKING_ACCOUNT or 'default'",
    )
    parser.add_argument(
        "--agent-id",
        default=os.getenv("OPENVIKING_AGENT", DEFAULT_AGENT_ID),
        help="X-OpenViking-Agent / agent_id, default: env OPENVIKING_AGENT or 'default'",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional X-API-Key, default: empty string",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N rows, default: 0 means all rows",
    )
    args = parser.parse_args()

    rows, fieldnames = load_rows(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]
    semaphore = asyncio.Semaphore(max(args.parallel, 1))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        tasks = [
            asyncio.create_task(
                enrich_row(
                    dict(row),
                    openviking_url=args.openviking_url,
                    session_prefix=args.session_prefix,
                    index=index,
                    user_id=args.user_id,
                    account=args.account,
                    agent_id=args.agent_id,
                    api_key=args.api_key,
                    semaphore=semaphore,
                )
            )
            for index, row in enumerate(rows, start=1)
        ]

        written_rows = 0
        for task in asyncio.as_completed(tasks):
            row = await task
            writer.writerow(row)
            f.flush()
            written_rows += 1
            print(f"written_rows={written_rows}", flush=True)

    print(f"input_rows={len(rows)}")
    print(f"output={output_path}")
    print("added_columns=retrieved_memories_json,retrieved_memories_text")


if __name__ == "__main__":
    asyncio.run(main())
