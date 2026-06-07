<a name="readme-top"></a>

<div align="center">
  <h3 align="center">DSPy LanceDB Memory</h3>

  <p align="center">
    Persistent vector memory store for DSPy-powered AI agents — extract, store, and recall structured memories from conversation.
    <br />
    <a href="#table-of-contents"><strong>Explore the Documentation »</strong></a>
    <br />
    <a href="https://github.com/thememium/dspy-lancedb-memory/issues">Report Bug</a>
    ·
    <a href="https://github.com/thememium/dspy-lancedb-memory/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->

<a name="table-of-contents"></a>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li><a href="#quick-start">Quick Start</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#memory-taxonomy">Memory Taxonomy</a></li>
    <li><a href="#development">Development</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

<!-- ABOUT -->

## About

DSPy Memory is a persistent vector memory store for DSPy-powered AI agents. It uses DSPy signatures to extract structured, categorized memories from conversation turns and stores them in LanceDB for efficient semantic retrieval.

- **Method-based SDK** — Single ``memory.configure()`` entry point for extraction LM, embedding model, and reranker
- **DSPy-native extraction** — Uses `ChainOfThought` with a typed `ExtractMemory` signature to pull salient information from conversations
- **Structured memory taxonomy** — Six memory categories (preference, semantic, episodic, procedural, summary, artifact) for fine-grained organization
- **Persistent vector storage** — LanceDB-backed with automatic text embeddings via the DSPy `Embedder`
- **Semantic search** — Query memories by user ID, session ID, conversation ID, memory type, or natural language
- **Optional reranking** — ``LiteLLMReranker`` wraps ``litellm.rerank()`` for cross-encoder reranking via Cohere, Jina, and any LiteLLM-compatible provider
- **Full CRUD** — Create, search, update, and delete individual memories or batch-extract from conversations

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- QUICK START -->

## Quick Start

### Prerequisites

Set the API key environment variable for your chosen provider. LiteLLM routes to the correct key automatically based on the model prefix:

```bash
# Examples — set whichever matches your provider
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export CO_API_KEY="..."            # Cohere reranker
export JINA_API_KEY="..."          # Jina reranker
```

If you use a proxy or gateway (e.g. OpenRouter, LiteLLM proxy), set the base URL and key:

```bash
export OPENAI_API_BASE="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="sk-or-v1-..."
```

### Install

Install dspy-lancedb-memory with uv (recommended)

```bash
uv add dspy-lancedb-memory
```

Install with pip (alternative)

```bash
pip install dspy-lancedb-memory
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE -->

## Usage

### Basic Usage

```python
from dspy_lancedb_memory import memory
import dspy

# Configure: extraction LM, embedding LM, and optional reranker LM
memory.configure(
    extraction_lm=dspy.LM("openai/gpt-4o-mini"),
    embedding_lm=dspy.LM("openai/text-embedding-3-small"),
    reranker_lm=dspy.LM("cohere/rerank-4-fast"),   # optional — omit to disable
)

# Create a store — picks up all defaults from configure()
store = memory.Store()

# Store a single memory
store.create_memory(
    user_id="user_123",
    content="Edward prefers DSPy signatures over ad-hoc prompts.",
    memory_type="preference",
)

# Store with session and conversation scoping
store.create_memory(
    user_id="user_123",
    session_id="session_abc",
    conversation_id="conv_456",
    content="Remember this from our conversation about RAG pipelines.",
    memory_type="episodic",
)

# Search memories (with reranking)
results = store.search_memories(
    user_id="user_123",
    session_id="session_abc",   # optional — narrow to a session
    query="What does Edward prefer?",
    use_reranker=True,
)
print(results)
```

### Extract Memories from Conversation

The real power is automatic extraction. Pass a conversation turn and the LLM extracts all salient memories, each categorized by type.

```python
from dspy_lancedb_memory import memory
import dspy

memory.configure(extraction_lm=dspy.LM("openai/gpt-4o-mini"))
store = memory.Store()

messages = [
    {
        "role": "user",
        "content": "I really like using DSPy signatures instead of writing prompts by hand. "
                   "I'm building a CLI tool to organize my digital bookmarks by topic. "
                   "The PR is at github.com/example/bookmark-organizer/pull/42.",
    },
]

created = store.create_memories(
    user_id="user_123",
    contents=messages,
    extract=True,  # default; uses DSPy ChainOfThought to extract memories
)
# Returns multiple MemoryItems categorized automatically:
#   preference: "User prefers DSPy signatures over writing prompts by hand"
#   semantic: "User is building a CLI bookmark organizer"
#   procedural: "User is organizing bookmarks by topic"
#   artifact: "github.com/example/bookmark-organizer/pull/42"

for m in created:
    print(f"[{m['memory_type']}] {m['content']}")
```

### Search with Memory Type and Reranking

```python
from dspy_lancedb_memory import memory
import dspy

memory.configure(
    extraction_lm=dspy.LM("openai/gpt-4o-mini"),
    reranker_lm=dspy.LM("cohere/rerank-4-fast"),
)
store = memory.Store()

# Filter by memory type and optionally use reranking for better results
results = store.search_memories(
    user_id="user_123",
    query="What are Edward's tool preferences?",
    memory_type="preference",  # optional filter
    limit=5,
    use_reranker=True,         # uses configured reranker endpoint
)
```

### Filtering by Session and Conversation

Every memory can be scoped to a ``session_id`` and ``conversation_id``. Both are optional and can be used independently or together.

```python
# Scope to a specific session
session_memories = store.search_memories(
    user_id="user_123",
    session_id="session_abc",
    query="What did we discuss last time?",
)

# Scope to a specific conversation
conversation_memories = store.search_memories(
    user_id="user_123",
    conversation_id="conv_456",
    query="RAG pipeline details",
)

# Combine session + conversation for maximum precision
precise = store.search_memories(
    user_id="user_123",
    session_id="session_abc",
    conversation_id="conv_456",
    query="specific topic",
)

# Omit both to search across all sessions and conversations
all_results = store.search_memories(
    user_id="user_123",
    query="anything",
)
```

### Raw Store (No Extraction)

Store content verbatim without LLM extraction.

```python
store.create_memories(
    user_id="user_123",
    contents=[{"role": "user", "content": "A raw fact worth storing."}],
    extract=False,
    memory_type="semantic",
)
```

### Update and Delete

```python
# Update memory content (re-embeds automatically)
store.update_memory(
    memory_id="some-uuid",
    content="Updated memory text",
)

# Delete a memory
store.delete_memory(memory_id="some-uuid")
```

### Upsert — Insert, Update, or Skip

``upsert_memory`` uses semantic similarity to decide what to do:

1. **Exact match** — same ``content`` string exists → skip (no-op)
2. **Semantic match** — similar content found (cosine similarity ≥ threshold) → **update** it
3. **No match** — nothing close enough → **insert** a new memory

```python
# First insert
store.upsert_memory(
    user_id="user_123",
    content="Edward is building a RAG pipeline for climate modeling.",
    memory_type="semantic",
)

# Same content string → skip (returns existing row unchanged)
store.upsert_memory(
    user_id="user_123",
    content="Edward is building a RAG pipeline for climate modeling.",
)

# Semantically similar content → update that memory in place
store.upsert_memory(
    user_id="user_123",
    content="Edward is designing a RAG pipeline for climate data analysis.",
    similarity_threshold=0.8,  # lower = more aggressive updates
)

# Completely different content → insert a new row
store.upsert_memory(
    user_id="user_123",
    content="Edward prefers DSPy signatures over raw prompts.",
    memory_type="preference",
)
```

The ``similarity_threshold`` (default ``0.85``) controls how close two
memories must be to consider them the same.  Higher values make upsert
more conservative (mostly inserts); lower values make it more aggressive
(mostly updates).

### Batch Upsert with Extraction

``upsert_memories`` mirrors ``create_memories`` exactly — same parameters,
same DSPy extraction — but each extracted memory goes through the upsert
decision instead of a blind insert.

```python
from dspy_lancedb_memory import memory
import dspy

memory.configure(extraction_lm=dspy.LM("openai/gpt-4o-mini"))
store = memory.Store()

messages = [
    {
        "role": "user",
        "content": "I really like using DSPy signatures instead of writing prompts by hand. "
                   "I'm working on a RAG pipeline for my thesis on climate modeling. "
                   "The PR is at github.com/example/climate-rag/pull/42.",
    },
]

upserted = store.upsert_memories(
    user_id="user_123",
    contents=messages,
    extract=True,
)

for m in upserted:
    # Each extracted memory was independently upserted:
    #   - exact match → skip
    #   - semantic match → update
    #   - no match → insert
    print(f"[{m['memory_type']}] {m['content']}")
```

### Using the Reranker

The easiest way — configure via ``memory.configure(reranker_lm=...)`` with a ``dspy.LM`` and `memory.Store()` picks it up automatically:

```python
from dspy_lancedb_memory import memory
import dspy

memory.configure(
    extraction_lm=dspy.LM("openai/gpt-4o-mini"),
    reranker_lm=dspy.LM("cohere/rerank-4-fast"),
)
store = memory.Store()  # LiteLLMReranker auto-created from reranker_lm
```

You can also pass a plain model string instead of a ``dspy.LM``:

```python
memory.configure(reranker_lm="cohere/rerank-english-v3.0")
```

For full control (custom column, top_n, etc.), build a ``LiteLLMReranker`` and pass it to ``Store()``:

```python
from dspy_lancedb_memory import LiteLLMReranker

reranker = LiteLLMReranker(
    model="cohere/rerank-english-v3.0",
    column="content",               # LanceDB column to rerank against
    top_n=20,                       # optional: limit reranked candidates
)

store = memory.Store(reranker=reranker)
```

The model string uses the same ``provider/model`` format as ``dspy.LM`` — e.g.
``"cohere/rerank-english-v3.0"``, ``"jina/jina-reranker-v2-base-multilingual"``.
LiteLLM handles the routing.

### Custom API Base and Key

When running behind a proxy, gateway, or self-hosted endpoint, pass ``api_base`` and ``api_key`` directly to ``LiteLLMReranker``:

```python
from dspy_lancedb_memory import LiteLLMReranker

reranker = LiteLLMReranker(
    model="my-provider/rerank-model",
    api_base="https://my-gateway.example.com/v1",
    api_key="sk-my-secret",
    column="content",
    top_n=20,
)

store = memory.Store(reranker=reranker)
```

For embeddings behind a custom endpoint, pass the same ``api_base``/``api_key`` via a ``dspy.LM``:

```python
import dspy

memory.configure(
    embedding_lm=dspy.LM(
        "openai/text-embedding-3-small",
        api_base="https://my-gateway.example.com/v1",
        api_key="sk-my-secret",
    ),
)
```

LiteLLM automatically routes to the correct provider based on the model prefix. If your
provider is not in LiteLLM's built-in list, ``LiteLLMReranker`` falls back to calling a
Cohere-compatible ``/rerank`` endpoint on your ``api_base``.

### Custom Configuration

Everything — including LanceDB defaults — in one call:

```python
from dspy_lancedb_memory import memory
import dspy

memory.configure(
    extraction_lm=dspy.LM("anthropic/claude-sonnet-4-20250514"),           # extraction LM
    embedding_lm=dspy.LM("openai/text-embedding-3-small"),              # embedding LM
    embedding_dim=1536,                                                             # must match
    reranker_lm=dspy.LM("cohere/rerank-4-fast"),                         # reranker model
    uri=".my_memories",                                                              # LanceDB path
    table_name="user_memories",                                                      # LanceDB table
)

store = memory.Store()  # everything inherited from configure()
```

Override individual fields on ``Store()`` when you need something different:

```python
store = memory.Store(uri="./scratch", reranker=None)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Memory Taxonomy

Every extracted memory is categorized into one of six types:

| Type | Description | Example |
|---|---|---|
| `preference` | User tastes, likes/dislikes, preferred formats, tone, tools | `"User prefers DSPy signatures over ad-hoc prompts"` |
| `semantic` | Facts, biographical data, stable knowledge about the user | `"User is a PhD student researching climate modeling"` |
| `episodic` | Events, tasks, decisions, or outcomes from a specific interaction | `"User decided to use LanceDB over Chroma for persistence"` |
| `procedural` | Learned rules, workflows, steps, patterns, or how-to knowledge | `"User's RAG pipeline uses hybrid search with reranking"` |
| `summary` | Compressed conversation or task summaries capturing the gist | `"User discussed their thesis work on climate RAG pipelines"` |
| `artifact` | Links, paths, IDs, or references to files, PRs, docs, outputs | `"github.com/example/climate-rag/pull/42"` |

When storing directly (without extraction), the default type is `semantic`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Available API

| API | Description |
|---|---|
| [`memory`](#basic-usage) | SDK module — ``memory.configure()`` and ``memory.Store()`` |
| [`MemoryExtractor`](#extract-memories-from-conversation) | DSPy `ChainOfThought` module for memory extraction |
| [`LiteLLMReranker`](#using-the-reranker) | Cross-encoder reranker via ``litellm.rerank()`` — supports Cohere, Jina, and any LiteLLM-compatible provider |
| [`MemoryType`](#memory-taxonomy) | Enum of the six memory categories |
| [`MemoryItem`](#extract-memories-from-conversation) | Pydantic model for extracted memories |
| [`upsert_memory`](#upsert--insert-update-or-skip) | Semantic upsert — insert, update, or skip based on content similarity |
| `session_id` / `conversation_id` | Optional scoping fields on ``create_memory``, ``create_memories``, ``search_memories``, and ``upsert_memory`` |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DEVELOPMENT -->

## Development

### Code Quality

This project uses several tools to maintain code quality:

- **Ruff:** Linting and formatting
- **isort:** Import sorting
- **pytest:** Testing framework
- **deptry:** Dependency checking
- **ty:** Type checking (based on pyright)

**Available commands:**

```sh
# Run all quality checks
uv run poe clean-full

# Individual checks
uv run poe lint          # Ruff linting
uv run poe format        # Ruff formatting
uv run poe sort          # Import sorting
```

### Testing

Run tests using pytest:

```sh
# Run all tests
uv run pytest

# Run specific test
uv run pytest path/to/test.py::test_name
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributing

Quick workflow:

1. Fork and branch: `git checkout -b feature/name`
2. Make changes
3. Run checks: `uv run poe clean-full`
4. Commit and push
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## License

MIT (as declared in `pyproject.toml`).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<div align="center">
  <p>
    <sub>Built by <a href="https://github.com/thememium">thememium</a></sub>
  </p>
</div>
