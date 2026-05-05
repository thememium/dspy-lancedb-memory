"""Example usage of the dspy-memory SDK."""

from __future__ import annotations

import dspy
from pydantic import BaseModel

from dspy_memory import memory

# ===========================================================================
# 1. Basic usage
# ===========================================================================

memory.configure(
    model="openrouter/openai/gpt-4o-mini",
    embedding_model="openrouter/openai/text-embedding-3-small",
    reranker_model="cohere/rerank-4-fast",
)

store = memory.Store()

# Store a single memory
store.create_memory(
    user_id="user_123",
    content="Edward prefers DSPy signatures over ad-hoc prompts.",
    memory_type="preference",
)

# Search with reranking
results = store.search_memories(
    user_id="user_123",
    query="What does Edward prefer?",
    use_reranker=True,
)
print("Basic search:", results)

# ===========================================================================
# 2. Extract memories from conversation
# ===========================================================================

messages = [
    {
        "role": "user",
        "content": (
            "I really like using DSPy signatures instead of writing prompts by hand. "
            "I'm working on a RAG pipeline for my thesis on climate modeling. "
            "The PR is at github.com/example/climate-rag/pull/42."
        ),
    },
]

created = store.create_memories(
    user_id="user_123",
    messages=messages,
    extract=True,
)

for m in created:
    print(f"  [{m['memory_type']}] {m['content']}")

# ===========================================================================
# 3. Custom memory types
# ===========================================================================

# Any string works as a memory type — not just the predefined enum values.
store.create_memory(
    user_id="user_123",
    content="Edward is a PhD student at Stanford.",
    memory_type="biographical",
)

custom_results = store.search_memories(
    user_id="user_123",
    query="Edward's education",
    memory_type="biographical",
)
print("Custom type search:", custom_results)

# ===========================================================================
# 4. Custom extraction signature
# ===========================================================================


class CodeMemory(BaseModel):
    content: str
    type: str


class ExtractCodeReferences(dspy.Signature):
    """Extract code-related memories from a conversation turn.

    Instructions
    ------------
    1. Read the messages carefully.
    2. Identify ALL code references, tool preferences, and technical facts.
    3. For each memory output a ``CodeMemory`` with:
       * ``content`` — concise, self-contained technical fact.
       * ``type`` — one of:
         **tool** (preferred tools / editors / CLIs),
         **repo**   (repository links, PRs, branches),
         **pattern** (coding style, architecture, conventions).
    """

    messages: list[dict[str, str]] = dspy.InputField()
    memories: list[CodeMemory] = dspy.OutputField(
        desc="Extracted code-related memories."
    )


code_store = memory.Store(uri=".lancedb", signature=ExtractCodeReferences)

tech_messages = [
    {
        "role": "user",
        "content": (
            "I use ruff for linting and uv for package management. "
            "My open-source repo is at github.com/user/my-tool."
        ),
    },
]

code_memories = code_store.create_memories(
    user_id="dev_1",
    messages=tech_messages,
    extract=True,
)

for m in code_memories:
    print(f"  [{m['memory_type']}] {m['content']}")

# ===========================================================================
# 5. Per-store overrides
# ===========================================================================

# Override individual fields without affecting other stores:
scratch = memory.Store(
    uri="./scratch",
    reranker=None,  # disable reranking for this store
    table_name="scratchpad",
)
scratch.create_memory(user_id="anon", content="scratch data")

# ===========================================================================
# 6. Update and delete
# ===========================================================================

mem = store.create_memory(
    user_id="user_123",
    content="Old text",
    memory_type="semantic",
)
store.update_memory(memory_id=mem["id"], content="Updated text")
store.delete_memory(memory_id=mem["id"])
print("Update and delete: OK")
