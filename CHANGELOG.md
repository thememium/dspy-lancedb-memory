# Changelog

## v0.1.5 (2026-06-07)

[Compare changes](https://github.com/thememium/dspy-lancedb-memory/compare/v0.1.4...v0.1.5)

### ✅ Tests

- add smoke test for package import and metadata ([483f704](https://github.com/thememium/dspy-lancedb-memory/commit/483f704c3e0d68c35cbe10f74eec238954fb2144))

### 🤖 CI

- **publish.yml**: add workflow to publish Python package on release ([b5ff782](https://github.com/thememium/dspy-lancedb-memory/commit/b5ff78214c46c05785feaf2dafc68f7e90e849c8))

### Contributors

- Edward Boswell <thememium@gmail.com>

## v0.1.4 (2026-06-07)

[Compare changes](https://github.com/thememium/dspy-lancedb-memory/compare/v0.1.3...v0.1.4)

### ✅ Tests

- add smoke test for package import and metadata ([483f704](https://github.com/thememium/dspy-lancedb-memory/commit/483f704c3e0d68c35cbe10f74eec238954fb2144))

### 🤖 CI

- **publish.yml**: add workflow to publish Python package on release ([b5ff782](https://github.com/thememium/dspy-lancedb-memory/commit/b5ff78214c46c05785feaf2dafc68f7e90e849c8))

### Contributors

- Edward Boswell <thememium@gmail.com>

## v0.1.3 (2026-06-07)

[Compare changes](https://github.com/thememium/dspy-lancedb-memory/compare/v0.1.2...v0.1.3)

### 🚀 Enhancements

- **memory**: add api_base and api_key support to LiteLLMReranker ([5bdd33f](https://github.com/thememium/dspy-lancedb-memory/commit/5bdd33f971dbf007855064e2b43c839208cf4abc))
- **reranking**: add custom API support and provider filtering for LiteLLM reranker ([db2f346](https://github.com/thememium/dspy-lancedb-memory/commit/db2f34699229d0a457d6474218f1a35ecae66814))
- **dspy_lancedb_memory**: dynamic embed dim inference ([b085a9b](https://github.com/thememium/dspy-lancedb-memory/commit/b085a9bfda8fe478fd9f8d654c98f526234b9d4b))
- **memory**: add DSPy Memory library with config, extraction, storage, reranker ([b0d2bb3](https://github.com/thememium/dspy-lancedb-memory/commit/b0d2bb3fc00631961624cfd4a396495d69491c69))
- **example**: add deletion example to process_memories ([c757edc](https://github.com/thememium/dspy-lancedb-memory/commit/c757edc3d68672e944daa7a6d02fcb96ac746271))
- **dspy_memory**: expose MemoryOperation API in __init__ ([04904db](https://github.com/thememium/dspy-lancedb-memory/commit/04904db0e5d28b092d197cbe31e62597927b60c3))
- **dspy_memory**: add memory operation extraction ([49b7163](https://github.com/thememium/dspy-lancedb-memory/commit/49b71637f6a6a56d0c01e7e730d01b849a1ab606))
- **models**: add MemoryOperation and MemoryOperations models ([a74624c](https://github.com/thememium/dspy-lancedb-memory/commit/a74624ca36063356f70cb0c279085f464b0076db))
- **memory**: add delete_memories_by_search and process_memories to store ([784d87c](https://github.com/thememium/dspy-lancedb-memory/commit/784d87c79bc4befbbc0ad55f03f8620de1b4febc))
- **store**: add relevance filtering to search_memories ([66f3bf7](https://github.com/thememium/dspy-lancedb-memory/commit/66f3bf791eff696550d3c87c7c497cb8fe098ba0))
- **memory**: add relevance_score to Memory and update store conversion ([6a604d7](https://github.com/thememium/dspy-lancedb-memory/commit/6a604d70600e18cd105597819ee934ef20ce0b36))
- **dspy_memory**: expose MemoryReconciler and ReconciledMemory ([97b5a3a](https://github.com/thememium/dspy-lancedb-memory/commit/97b5a3a4d95af1e64b8459946f3dead7ce663df6))
- **dspy_memory**: add reconciler support ([5ececa5](https://github.com/thememium/dspy-lancedb-memory/commit/5ececa54447ccaf69d89607b396aed0bd54362c8))
- **models**: add ReconciledMemory model ([fb2d8a7](https://github.com/thememium/dspy-lancedb-memory/commit/fb2d8a73ca763cfb9bbf7a1e4f8c4535a09c2dbe))
- **memory**: add ReconcileMemory and MemoryReconciler for deduplication and update logic ([7d14b8a](https://github.com/thememium/dspy-lancedb-memory/commit/7d14b8a778de0ed761163dfa66ed8c896c4843ca))
- **memory**: add semantic upsert operations and API refinements ([f64b2bc](https://github.com/thememium/dspy-lancedb-memory/commit/f64b2bc7bd106e772ca9686bd0ce61289b66e852))

### 🩹 Fixes

- **memory-store**: set default minRelevanceScore and simplify fetch logic ([862a008](https://github.com/thememium/dspy-lancedb-memory/commit/862a0080a73049b64cb102b3471d123c65d76256))

### 💅 Refactors

- **dspy_lancedb_memory/config.py**: allow optional embedding dimension in config ([2511006](https://github.com/thememium/dspy-lancedb-memory/commit/2511006e6bf121c90298791182cc518b5dbbcc97))
- **example**: update example to use dspy-lancedb-memory SDK ([5743fc0](https://github.com/thememium/dspy-lancedb-memory/commit/5743fc0704204bbc26463e3e1886255f972690f8))
- **pyproject.toml**: rename package to dspy-lancedb-memory and update script names ([4544278](https://github.com/thememium/dspy-lancedb-memory/commit/45442783986fe8a5d0288c8366e7cfc6478d360d))
- **test**: update imports to dspy_lancedb_memory ([438a1e1](https://github.com/thememium/dspy-lancedb-memory/commit/438a1e1b01c56b22dbb305d54f194297296a3f61))
- **dspy_memory/store**: add semantic upsert deduplication and filtering logic ([22a92d9](https://github.com/thememium/dspy-lancedb-memory/commit/22a92d9fba99117863862ac08796bf1c5c3828a9))

### 📖 Documentation

- **readme**: update example scenario to bookmark organizer ([e828b66](https://github.com/thememium/dspy-lancedb-memory/commit/e828b665de76f8754f86e09903671caa26d44948))
- add bug report issue template ([1d16b11](https://github.com/thememium/dspy-lancedb-memory/commit/1d16b11b586e7520ffd217141ca28dab8c473466))
- add contributing guide for new contributors ([15a19c1](https://github.com/thememium/dspy-lancedb-memory/commit/15a19c121e48cfe4f11b15c121d6b90af284fe66))
- add MIT license file ([16bc017](https://github.com/thememium/dspy-lancedb-memory/commit/16bc0173f3ef8466bcc080631ef27714578d73ab))
- add SECURITY.md for vulnerability reporting and security guidelines ([c2ca4e6](https://github.com/thememium/dspy-lancedb-memory/commit/c2ca4e6e87d78101cd263bca78d064200025b223))
- **README**: rename dspy‑memory to dspy‑lancedb‑memory in docs ([dc6cb14](https://github.com/thememium/dspy-lancedb-memory/commit/dc6cb1440a20ac4ac1a52c56cc58a624f1bb8b37))
- **readme**: fix table separator line to match column count ([342d00a](https://github.com/thememium/dspy-lancedb-memory/commit/342d00ad34b0efc180d33326db3b5cbc6709846a))

### 🏡 Chore

- **deps**: bump usechange from 0.1.28 to 0.1.35 ([1c1f417](https://github.com/thememium/dspy-lancedb-memory/commit/1c1f4177610c57cfac39243314ffcef72b1ac0ca))
- **.gitignore**: add .omo to ignore ([3d4efaf](https://github.com/thememium/dspy-lancedb-memory/commit/3d4efafb67fa5fbfb789bb604b0566bb30564b01))

### ✅ Tests

- add comprehensive tests for memory operation processing ([e5c894e](https://github.com/thememium/dspy-lancedb-memory/commit/e5c894ebbb2f0586cc31f632dd885416f14e5efe))
- add reconciler integration tests and deterministic stub ([bb6dbdd](https://github.com/thememium/dspy-lancedb-memory/commit/bb6dbddd2000139536a9d911426538f91c739bce))

### Other Changes

- Merge pull request #2 from thememium/eboswell/feat/variable-dim-size-and-local-models (#2) ([d79f8ab](https://github.com/thememium/dspy-lancedb-memory/commit/d79f8ab23451d4401e61d1c6c8292ec043de02c7))
- Merge pull request #1 from thememium/eboswell/feat/add-extract-operation-with-delete (#1) ([531264c](https://github.com/thememium/dspy-lancedb-memory/commit/531264c9fe040728ffd37867ccaeca82f4bec928))

### Contributors

- Edward Boswell <thememium@gmail.com>

## v0.1.2 (2026-05-05)

[Compare changes](https://github.com/thememium/dspy-memory/compare/v0.1.1...v0.1.2)

### 🚀 Enhancements

- **reranking**: add OpenRouter support to LiteLLMReranker ([e4b83b8](https://github.com/thememium/dspy-memory/commit/e4b83b8e8f2f9bbb9d85c8d494c9a0ec2c4db1bb))
- **dspy_memory/reranking**: add LM support and robust error handling ([0e921be](https://github.com/thememium/dspy-memory/commit/0e921bec36c8d0a6e8c14354502301d4ac425419))
- **dspy_memory**: add session_id to memory schema ([6042027](https://github.com/thememium/dspy-memory/commit/6042027817f9142bdf93a7e1a043ec0c21ad19d1))
- **dspy_memory**: add support for custom extraction Signature ([bfa6945](https://github.com/thememium/dspy-memory/commit/bfa69450323de2bac6dea8c07c139b604657f56c))
- **dspy_memory/extraction.py**: allow configurable Signature for MemoryExtractor ([bdff86a](https://github.com/thememium/dspy-memory/commit/bdff86a51865392f5d70703b4236e14d7a9cac4a))
- **memory**: add signature support for memory extraction ([01a7966](https://github.com/thememium/dspy-memory/commit/01a7966d90c33d60e734dc94b2912314ba542fa2))
- **memory**: allow custom memory types in memory_type_from_string ([03ec472](https://github.com/thememium/dspy-memory/commit/03ec472d227bce89a74bcc4e9e62dec0b1e3f8fd))
- **memory**: add factory API for configuring DSPy memory ([0cd8254](https://github.com/thememium/dspy-memory/commit/0cd82545889f1e11c4b88bfd2d0493d22e5b3d22))

### 🩹 Fixes

- **memory.py**: correct OpenRouterReranker initialization ([4947220](https://github.com/thememium/dspy-memory/commit/494722048c9ad1dfb117d3c55b34db40f297ec06))

### 💅 Refactors

- **dspy_memory/store.py**: convert embed output to native floats and add bulk embedding support ([aa0847a](https://github.com/thememium/dspy-memory/commit/aa0847a80e8d91915c75b54962c12fe3efb047ea))
- **dspy_memory**: rename reranker_model to reranker_lm and update API ([aa0151f](https://github.com/thememium/dspy-memory/commit/aa0151fe94500780e615001fe529fe3ed8acbd97))
- **reranking**: replace OpenRouterReranker with LiteLLMReranker and simplify config ([9578226](https://github.com/thememium/dspy-memory/commit/95782266393792d2808895e66f74d32956bcd2eb))
- **dspy_memory**: rename lm to extraction_lm ([1117872](https://github.com/thememium/dspy-memory/commit/1117872ca1fe5312d23e2d33230d8fdee66e1d87))
- **memory**: use dspy.LM for config and fix reranker model path ([53e4ad6](https://github.com/thememium/dspy-memory/commit/53e4ad6de074c3da8f623aebe51fbeef1f223cf0))
- **dspy_memory/config.py**: rename embedding and reranker configs from ([0d98ba2](https://github.com/thememium/dspy-memory/commit/0d98ba2608023c60c917f7fec7612f89b6989214))
- **memory.py**: rename embedding and reranker args to dspy.LM ([1f5b7d3](https://github.com/thememium/dspy-memory/commit/1f5b7d3dd68dd28c021c6685a4fb4757b78c6e57))
- **dspymemory**: replace embedding_model with embedding_lm and add default LM instance ([4725414](https://github.com/thememium/dspy-memory/commit/47254148aa6277c61e61c90f242f26b03c103ab3))
- rename messages argument to contents in create_memories ([d6fdfea](https://github.com/thememium/dspy-memory/commit/d6fdfea5f9b879bec94e95255e5df552c05adfa5))
- **store**: rename `messages` parameter to `contents` in LanceDSPyMemoryStore ([75bd2ea](https://github.com/thememium/dspy-memory/commit/75bd2ea33cd5634db860cac8c4c8df58b5ed104b))
- **pyproject.toml**: update dependencies formatting and dev script ([82562c3](https://github.com/thememium/dspy-memory/commit/82562c32992caef3601a2e51ca538431ae8d45ab))
- **dspy_memory/store.py**: use ExtractMemory with signature and support string memory types ([0907249](https://github.com/thememium/dspy-memory/commit/09072493af73026f24313e4a4928f040ec4192b4))
- **dspy_memory/extraction.py**: remove automatic runtime config, require explicit memory.configure() call ([feca0ea](https://github.com/thememium/dspy-memory/commit/feca0ea2239ee38549a9c189e2b57711028235b3))
- **dspy_memory**: expose memory module and clean up __init__ ([135d4d4](https://github.com/thememium/dspy-memory/commit/135d4d42312c9dc1f3d8224f89695f6e60fce7bb))
- **config**: add comprehensive global configuration API ([06452ee](https://github.com/thememium/dspy-memory/commit/06452eeeb3ebb876236974c21c242fc255c7f67a))

### 📖 Documentation

- replace OpenRouterReranker with LiteLLMReranker in README ([3e104aa](https://github.com/thememium/dspy-memory/commit/3e104aaea026d1d2ddde2c09bb9ed96386d01419))
- **readme**: add session/conversation scoping, fix create_memories param ([41b62ba](https://github.com/thememium/dspy-memory/commit/41b62baf7800d263fc7d08e3f705f7262b1b0d6a))
- **example**: add filtering examples by user_id, session_id, conversation_id ([facf2be](https://github.com/thememium/dspy-memory/commit/facf2bee6387b33492b5b95eacab1bc5b15c0fdd))
- **README**: update example configs to use LM objects and dspy import ([138de65](https://github.com/thememium/dspy-memory/commit/138de657af14f751c39da5b2edc1f22ac09c15a5))
- **example**: add example usage of dspy-memory SDK ([ab9dd85](https://github.com/thememium/dspy-memory/commit/ab9dd85e8989504d8cf259e0becacc6c00458447))
- **readme**: switch to memory.configure() and memory.Store() usage ([0c23e0d](https://github.com/thememium/dspy-memory/commit/0c23e0da20c413a7c51b2225538eb5eeb4bd012a))
- **README**: add comprehensive project README with usage, installation, examples ([6f26058](https://github.com/thememium/dspy-memory/commit/6f26058c20c5e806941f4a93a6166aa0884da7d2))

### 🏡 Chore

- **deps**: add httpx dependency ([ff25014](https://github.com/thememium/dspy-memory/commit/ff2501473661f45e81ad3939973510c25f34a5b8))
- **pyproject**: remove httpx, add litellm dependency ([b205343](https://github.com/thememium/dspy-memory/commit/b205343f21d0161a8461d3fb4752aab447174edf))
- **.gitignore**: add scratch folder to ignore list ([d862d47](https://github.com/thememium/dspy-memory/commit/d862d47c2f95dbdda5da01ca45fa9f44ab6fa666))
- **.gitignore**: ignore .lancedb* files ([2290122](https://github.com/thememium/dspy-memory/commit/229012294da1a87ee67db03f46fbb940c88df32e))

### 🎨 Styles

- **README**: fix indentation of reranker_lm configuration line ([50afd16](https://github.com/thememium/dspy-memory/commit/50afd1653019bd2fc4d036505f325f96831846c2))

### Contributors

- Edward Boswell <thememium@gmail.com>

## v0.1.1 (2026-05-05)

### 🚀 Enhancements

- **dspy_memory**: add LanceDSPyMemoryStore for persistent vector memories ([11d94e1](https://github.com/thememium/dspy-memory/commit/11d94e11b20db61421025b335bb173f4dabdd070))
- **dspy_memory**: add OpenRouterReranker for Cohere‑compatible reranking ([08694dd](https://github.com/thememium/dspy-memory/commit/08694ddb6dfa27afbfabab340e481fc6aed6c105))
- **dspy_memory**: add MemoryType enum, MemoryItem model, and type parser ([43da998](https://github.com/thememium/dspy-memory/commit/43da998bbe82c78e1daa5d1bf346c4709f88ab23))
- **dspy_memory**: add memory extraction module ([ed765a3](https://github.com/thememium/dspy-memory/commit/ed765a34103ff861d123d1459e05e05bc57dbc41))
- **dspy_memory**: add configuration helper for dspy LM runtime ([c338769](https://github.com/thememium/dspy-memory/commit/c338769c0ac87dc470922d8dc4e0f771191fbf6f))
- **dependencies**: add required runtime dependencies ([88b9351](https://github.com/thememium/dspy-memory/commit/88b9351c01e89787a400add5193156be6e2f22e4))

### 💅 Refactors

- **dspy_memory**: expose public API and clean up __init__.py ([4a0dba4](https://github.com/thememium/dspy-memory/commit/4a0dba4142cb61288821279466a8375e71aaa349))

### 🏡 Chore

- **deps**: add pyarrow>=24.0.0 for Arrow support ([f2c4cc2](https://github.com/thememium/dspy-memory/commit/f2c4cc2f77720db964f3547f1cdb9d9839baee80))

### Contributors

- Edward Boswell <thememium@gmail.com>
