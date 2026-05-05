<a name="readme-top"></a>

<div align="center">
  <h3 align="center">DSPy Memory</h3>

  <p align="center">
    Persistent vector memory store for DSPy-powered AI agents — extract, store, and recall structured memories from conversation.
    <br />
    <a href="#table-of-contents"><strong>Explore the Documentation »</strong></a>
    <br />
    <a href="https://github.com/thememium/dspy-memory/issues">Report Bug</a>
    ·
    <a href="https://github.com/thememium/dspy-memory/issues">Request Feature</a>
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
- **Semantic search** — Query memories by user ID, conversation ID, memory type, or natural language
- **Optional reranking** — Plug in the `OpenRouterReranker` for Cohere-compatible reranking over vector search results
- **Full CRUD** — Create, search, update, and delete individual memories or batch-extract from conversations

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- QUICK START -->

## Quick Start

### Prerequisites

Set the `OPENROUTER_API_KEY` environment variable if you plan to use the OpenRouter reranker:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### Install

Install dspy-memory with uv (recommended)

```bash
uv add dspy-memory
```

Install with pip (alternative)

```bash
pip install dspy-memory
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE -->

## Usage

### Basic Usage

```python
from dspy_memory import memory

# Configure: extraction LM, embedding model, and optional reranker
memory.configure(
    model="openrouter/openai/gpt-4o-mini",
    embedding_model="openrouter/openai/text-embedding-3-small",
    reranker_model="cohere/rerank-4-fast",   # optional — omit to disable reranking
)

# Create a store — picks up all defaults from configure()
store = memory.Store()

# Store a single memory
store.create_memory(
    user_id="user_123",
    content="Edward prefers DSPy signatures over ad-hoc prompts.",
    memory_type="preference",
)

# Search memories (with reranking)
results = store.search_memories(
    user_id="user_123",
    query="What does Edward prefer?",
    use_reranker=True,
)
print(results)
```

### Extract Memories from Conversation

The real power is automatic extraction. Pass a conversation turn and the LLM extracts all salient memories, each categorized by type.

```python
from dspy_memory import memory

memory.configure(model="openrouter/openai/gpt-4o-mini")
store = memory.Store()

messages = [
    {
        "role": "user",
        "content": "I really like using DSPy signatures instead of writing prompts by hand. "
                   "I'm working on a RAG pipeline for my thesis on climate modeling. "
                   "The PR is at github.com/example/climate-rag/pull/42.",
    },
]

created = store.create_memories(
    user_id="user_123",
    messages=messages,
    extract=True,  # default; uses DSPy ChainOfThought to extract memories
)
# Returns multiple MemoryItems categorized automatically:
#   preference: "User prefers DSPy signatures over writing prompts by hand"
#   semantic: "User is researching climate modeling for their thesis"
#   procedural: "User is building a RAG pipeline"
#   artifact: "github.com/example/climate-rag/pull/42"

for m in created:
    print(f"[{m['memory_type']}] {m['content']}")
```

### Search with Memory Type and Reranking

```python
from dspy_memory import memory

memory.configure(
    model="openrouter/openai/gpt-4o-mini",
    reranker_model="cohere/rerank-4-fast",
)
store = memory.Store()

# Filter by memory type and optionally use reranking for better results
results = store.search_memories(
    user_id="user_123",
    query="What are Edward's tool preferences?",
    memory_type="preference",  # optional filter
    limit=5,
    use_reranker=True,         # uses OpenRouter /rerank endpoint
)
```

### Raw Store (No Extraction)

Store content verbatim without LLM extraction.

```python
store.create_memories(
    user_id="user_123",
    messages=[{"role": "user", "content": "A raw fact worth storing."}],
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

### Using OpenRouter Reranker

The easiest way — configure via ``memory.configure(reranker_model=...)`` and `memory.Store()` picks it up automatically:

```python
from dspy_memory import memory

memory.configure(
    model="openrouter/openai/gpt-4o-mini",
    reranker_model="cohere/rerank-4-fast",
)
store = memory.Store()  # reranker auto-created from configure()
```

For full control (custom column, top_n, etc.), build an ``OpenRouterReranker`` and pass it to ``Store()``:

```python
from dspy_memory import OpenRouterReranker

reranker = OpenRouterReranker(
    model="cohere/rerank-4-fast",  # default
    column="content",               # LanceDB column to rerank against
    top_n=20,                       # optional: limit reranked candidates
)

store = memory.Store(reranker=reranker)
```

Requires `OPENROUTER_API_KEY` environment variable or pass `api_key=...` to the constructor.

> **Note:** `dspy.LM` does not have a built-in reranker interface — the ``OpenRouterReranker`` calls OpenRouter's ``/rerank`` endpoint directly via HTTP.

### Custom Configuration

Everything — including LanceDB defaults — in one call:

```python
from dspy_memory import memory

memory.configure(
    model="openrouter/anthropic/claude-sonnet-4-20250514",                  # extraction LM
    embedding_model="openrouter/openai/text-embedding-3-small",              # embedding model
    embedding_dim=1536,                                                      # must match
    reranker_model="cohere/rerank-4-fast",                                    # reranker model
    uri=".my_memories",                                                       # LanceDB path
    table_name="user_memories",                                               # LanceDB table
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
|---|---|---|
| [`memory`](#basic-usage) | SDK module — ``memory.configure()`` and ``memory.Store()`` |
| [`MemoryExtractor`](#extract-memories-from-conversation) | DSPy `ChainOfThought` module for memory extraction |
| [`OpenRouterReranker`](#using-openrouter-reranker) | LanceDB-compatible reranker via OpenRouter |
| [`MemoryType`](#memory-taxonomy) | Enum of the six memory categories |
| [`MemoryItem`](#extract-memories-from-conversation) | Pydantic model for extracted memories |

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
