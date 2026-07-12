# Autoresearch: 100% Test Coverage

## Objective
Achieve 100% test coverage for the `dspy_lancedb_memory` package by writing comprehensive tests for all uncovered code paths.

## Metrics
- **Primary**: coverage_pct (%, higher is better) — the optimization target
- **Secondary**: test_count, test_time_s — monitoring test suite health

## How to Run
`uv run pytest tests/ -v --cov=dspy_lancedb_memory --cov-report=term-missing`

## Files in Scope
- `tests/test_store.py` — Main store tests (currently 86% covered)
- `tests/test_process_memories.py` — Process memories tests
- `tests/test_smoke.py` — Smoke tests
- `src/dspy_lancedb_memory/config.py` — Global config (46% covered)
- `src/dspy_lancedb_memory/extraction.py` — Memory extraction (38% covered)
- `src/dspy_lancedb_memory/memory.py` — Memory factory (28% covered)
- `src/dspy_lancedb_memory/models.py` — Data models (96% covered)
- `src/dspy_lancedb_memory/reranking.py` — Reranker (20% covered)
- `src/dspy_lancedb_memory/store.py` — Main store (86% covered)

## Off Limits
- Do NOT modify source code to make it more testable (no coverage cheating)
- Do NOT use `# pragma: no cover` or similar markers
- Do NOT mock internal implementation details just to hit lines

## Constraints
- All existing tests must continue to pass
- New tests should use the same `StubMemoryStore` pattern from test_store.py
- Tests must be deterministic (no real LLM/embedding calls)
- Coverage must be genuine - testing real behavior, not artificial line-hitting

## What's Been Tried
- Baseline: 73% coverage (92 tests, all passing)

## Coverage Gaps by Module

### config.py (46% → target 100%)
Missing: configure() function, get_* functions when called directly

### extraction.py (38% → target 100%)
Missing: MemoryOperationExtractor.forward(), MemoryExtractor.forward() (real paths), MemoryReconciler.forward() (real paths)

### memory.py (28% → target 100%)
Missing: configure() wrapper, Store() factory function

### reranking.py (20% → target 100%)
Missing: LiteLLMReranker._rerank(), _rerank_openrouter(), _rerank_custom_api(), _attach_fallback_scores(), rerank_vector(), rerank_fts(), rerank_hybrid()

### store.py (86% → target 100%)
Missing: Various edge cases in upsert_memories, delete_memories_by_search, _reconcile_one, _reconcile_batch_parallel

### models.py (96% → target 100%)
Missing: MemoryType._missing_() (lines 34-37)
