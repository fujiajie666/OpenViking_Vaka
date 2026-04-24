# Vaka LoCoMo Benchmark

Evaluates long-memory recall using two datasets:

- **`data/vaka_locomo.csv`** — 405 rows of real work conversations, session 1-100. Sessions 1-70 are imported as memory; sessions 71-100 are not used directly.
- **`data/vaka_judge.csv`** — 60 evaluation questions about user preferences/behaviour patterns, each with a `standard_answer` gold label.

## Case Split

Rows are grouped by global `session_id` into cases of 10 sessions each:

- `session_id` 1-10 → `case_0001`
- `session_id` 11-20 → `case_0002`
- `session_id` 21-30 → `case_0003`
- …

## Pipeline

All commands are run from the project root (`OpenViking/`). Results are written to `benchmark/vaka/vikingbot/result/`.

### Step 1 — Import memory (session 1-70)

Default identity is `account=default`, `user_id=default`, `agent_id=default`. The `--memory-sessions` default is `1-70`.

```bash
# Session granularity (default): one OpenViking session per global_session_id
caffeinate -i uv run python benchmark/vaka/vikingbot/import_to_ov.py \
    --input benchmark/vaka/vikingbot/data/vaka_locomo.csv

# Case granularity (for comparison): all sessions in a case merged into one OpenViking session
caffeinate -i uv run python benchmark/vaka/vikingbot/import_to_ov.py \
    --input benchmark/vaka/vikingbot/data/vaka_locomo.csv \
    --ingest-mode case
```

The two modes use different success keys and do not conflict — you can run both for side-by-side comparison without clearing any records.

Import is resumable: each session is checkpointed immediately to `result/import_success.csv` and `result/.ingest_record.json`. Re-running skips already-imported sessions automatically. To force a full re-import, add `--force-ingest`.

Use a custom identity when needed:

```bash
caffeinate -i uv run python benchmark/vaka/vikingbot/import_to_ov.py \
    --input benchmark/vaka/vikingbot/data/vaka_locomo.csv \
    --user-id vaka --agent-id vaka
```

After import, verify completeness:

```bash
python3 -c "
import csv
with open('benchmark/vaka/vikingbot/result/import_success.csv') as f:
    rows = list(csv.DictReader(f))
session_rows = [r for r in rows if r['global_session_id'].isdigit()]
ids = sorted(int(r['global_session_id']) for r in session_rows)
missing = [i for i in range(1, 71) if i not in ids]
print(f'Imported: {len(ids)}, missing: {missing or \"none\"}')
"
```

### Step 2 — Generate answers

Calls OpenViking `/bot/v1/chat` for each question in `vaka_judge.csv`. The `--user-id` and `--account` defaults are `default`. Resume by re-running the same command — already-answered questions are skipped automatically.

```bash
caffeinate -i uv run python benchmark/vaka/vikingbot/run_eval.py \
    benchmark/vaka/vikingbot/data/vaka_judge.csv \
    --output benchmark/vaka/vikingbot/result/vaka_qa_result.csv \
    --parallel 3
```

### Step 3 — Judge answers

Requires `ARK_API_KEY` (Ark platform, `doubao-seed-2-0-pro-260215` model). Set it via `~/.openviking_benchmark_env`, the `ARK_API_KEY` env var, or `--token`. Re-running skips already-judged rows; add `--force` to re-judge everything.

```bash
uv run python benchmark/vaka/vikingbot/judge.py \
    --input benchmark/vaka/vikingbot/result/vaka_qa_result.csv \
    --parallel 10
```

### Step 4 — Statistics

```bash
uv run python benchmark/vaka/vikingbot/stat_judge_result.py \
    --input benchmark/vaka/vikingbot/result/vaka_qa_result.csv
```

## Notes

- **`run_full_eval.sh` has known bugs** (wrong input file and unsupported flags passed to `run_eval.py`). Run the four steps above manually until those are fixed.
- `run_eval.py` calls OpenViking live for each question — it does not reuse pre-computed answers from any CSV column.
- The `identity` used in Step 1 (`user_id`, `agent_id`, `account`) must exactly match what `run_eval.py` uses in Step 2, otherwise the bot cannot retrieve the imported memory.
- Steps 1 and 2 involve sustained network calls. On macOS, wrap with `caffeinate -i` to prevent system sleep from interrupting them.
- If `judge_standard` is present in the eval CSV, `judge.py` uses it as a scoring rubric. If `standard_answer` is present, it grades against the gold answer. If both are empty, the judge evaluates coherence with prior memory context.
