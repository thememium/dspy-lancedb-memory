# Ideas for Further Coverage Improvement

## Current Status: 97% Coverage (308 tests)

### Remaining Gaps (3% = 33 lines)

#### store.py Constructor Lines (92, 116, 123, 126, 129-135, 144-146)
- Line 92: `embedding_lm = dspy.LM(...)` - only when embedding_lm is None (requires real API setup)
- Lines 123, 126: `_embed` and `_embed_many` - StubMemoryStore overrides _embed
- Lines 129-135: `_infer_embedding_dim` - StubMemoryStore sets embedding_dim directly
- Lines 144-146: `_get_vector_dim` with ListType (rare LanceDB schema)

**Note**: Lines 96, 105-106 (api_base/api_key handling) ARE covered by constructor tests.

#### store.py _pack_metadata with scope (line 214)
This is `packed[cls._SCOPE_METADATA_KEY] = scope_dict` - covered by test_pack_with_metadata_and_scope.
May be a coverage tracking artifact.

#### store.py _semantic_match_action (line 421)
`return "update" if new_is_richer or same_slot_replacement else "skip"`
Requires specific inputs to trigger same_slot_replacement without new_is_richer.

#### store.py _get_or_create_table (lines 490-491)
Table creation path with specific conditions.

#### store.py get_memory_history (lines 668, 683-684)
History chain traversal with forward references (changed loop).

#### store.py create_memories extract path (lines 792, 819)
`cast(list[Any], extractor(messages=contents).memories)` - extract=True path.

#### store.py search hybrid (line 926)
Hybrid search with reranker - complex LanceDB interaction.

#### store.py reconciler paths (lines 1130-1136, 1312-1318)
Apply reconciliations and upsert with reconciler edge cases.

#### extraction.py line 221
`reconciled.final_type = str(existing_memories[0]["type"])` - specific reconciler path.

#### models.py line 36
`return None` in `MemoryType._missing_()` - may be a coverage tracking artifact.

#### reranking.py line 194
`combined = self.merge_results(...)` - requires LanceDB _rowid column.
