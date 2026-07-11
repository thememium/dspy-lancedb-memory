# Ideas for Further Coverage Improvement

## Current Status: 96% Coverage (301 tests)

### Remaining Gaps (4% = 42 lines)

#### store.py Constructor Lines (92-146)
These lines handle `embedding_lm` with `api_base`/`api_key` kwargs. The `StubMemoryStore` bypasses the constructor entirely by overriding `_embed`. To test these lines, we would need to:
1. Create a real `dspy.LM` with `api_base`/`api_key` kwargs
2. Let the constructor create a real `dspy.Embedder`
3. This requires network access or heavy mocking of dspy internals

**Recommendation**: Accept this gap - it's testing dspy library internals, not our code.

#### store.py Reconciler Edge Cases (1130-1136, 1312-1318)
These are in the `_apply_reconciliations` and `upsert_memory` paths. They require:
- `_apply_reconciliations` with "keep" action and no `existing_row` attribute
- `upsert_memory` non-semantic path with exact content match
- `upsert_memory` non-semantic path with similarity update

**Recommendation**: These can be tested with more complex test scenarios.

#### reranking.py Line 194
This is `combined = self.merge_results(vector_results, fts_results)` in `rerank_hybrid`. The LanceDB base class's `merge_results` method requires `_rowid` column which is added by LanceDB internally, not available in our test fixtures.

**Recommendation**: Accept this gap - it's testing LanceDB internals.

#### models.py Line 36, extraction.py Line 221
These are single lines in `_missing_` and `forward` methods. They may be dead code or rarely reached paths.

**Recommendation**: Investigate if these are truly unreachable or just need specific test conditions.

### Ideas for 100% Coverage (if needed)
1. **Mock dspy.Embedder** at the module level to test constructor paths without real API calls
2. **Create LanceDB tables with _rowid** column to test merge_results path
3. **Add parametrized tests** for all _semantic_match_action return paths
