"""Example usage of the dspy-lancedb-memory SDK."""

from __future__ import annotations

import dspy
from pydantic import BaseModel

from dspy_lancedb_memory import memory

# ===========================================================================
# 1. Basic usage
# ===========================================================================

memory.configure(
    extraction_lm=dspy.LM("openrouter/openai/gpt-4o-mini"),
    embedding_lm=dspy.LM("openrouter/openai/text-embedding-3-small"),
    reranker_lm=dspy.LM("openrouter/cohere/rerank-4-fast"),
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
    contents=messages,
    extract=True,
)

for m in created:
    print(f"  [{m.memory_type}] {m.content}")

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
    contents=tech_messages,
    extract=True,
)

for m in code_memories:
    print(f"  [{m.memory_type}] {m.content}")

# ===========================================================================
# 5. Filtering by user_id, session_id, and conversation_id
# ===========================================================================

# All three ID fields can be used alone or in combination
store.create_memory(
    user_id="user_123",
    session_id="session_abc",
    content="Session-specific memory.",
    memory_type="episodic",
)

# Scope search to a specific session
session_results = store.search_memories(
    user_id="user_123",
    session_id="session_abc",
    query="Session-specific",
)
print(f"Session-filtered: {len(session_results)} result(s)")

# Combine session_id with conversation_id
store.create_memory(
    user_id="user_123",
    session_id="session_abc",
    conversation_id="conv_456",
    content="Deeply scoped memory.",
    memory_type="semantic",
)

combined = store.search_memories(
    user_id="user_123",
    session_id="session_abc",
    conversation_id="conv_456",
    query="scoped",
)
print(f"Combined filter: {len(combined)} result(s)")

# ===========================================================================
# 6. Per-store overrides
# ===========================================================================

# Override individual fields without affecting other stores:
scratch = memory.Store(
    uri="./.lancedb/scratch",
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
store.update_memory(memory_id=mem.id, content="Updated text")
store.delete_memory(memory_id=mem.id)
print("Update and delete: OK")

# ===========================================================================
# 7. Upsert — insert, update, or skip based on semantic similarity
# ===========================================================================

# First insert
first = store.upsert_memory(
    user_id="user_123",
    content="Edward is building a RAG pipeline for climate modeling.",
    memory_type="semantic",
)
print(f"Created: {first.id[:8]}… — {first.content}")

# Same content → skip (no-op, returns same row)
skip = store.upsert_memory(
    user_id="user_123",
    content="Edward is building a RAG pipeline for climate modeling.",
    memory_type="semantic",
)
print(f"Skipped (same id? {skip.id == first.id}): {skip.content}")

# Semantically similar content → update
updated = store.upsert_memory(
    user_id="user_123",
    content="Edward is designing a RAG pipeline for climate data analysis.",
    memory_type="semantic",
    similarity_threshold=0.8,
)
print(f"Updated (same id? {updated.id == first.id}): {updated.content}")

# Completely different content → insert
new = store.upsert_memory(
    user_id="user_123",
    content="Edward prefers DSPy signatures over raw prompts.",
    memory_type="preference",
)
print(f"New: {new.id[:8]}… — [{new.memory_type}] {new.content}")

# ===========================================================================
# 8. Upsert memories — batch upsert with extraction
# ===========================================================================

# Exactly the same as store.create_memories(...), but each extracted
# memory goes through the upsert decision instead of a blind insert.
upserted = store.upsert_memories(
    user_id="user_123",
    contents=[
        {
            "role": "user",
            "content": (
                "I really like using DSPy signatures instead of writing prompts by hand. "
                "I'm working on a RAG pipeline for my thesis on climate modeling. "
                "The PR is at github.com/example/climate-rag/pull/42."
            ),
        },
    ],
    extract=True,
)

for m in upserted:
    # Each extracted memory was independently upserted:
    #   - exact match -> skip
    #   - semantic match -> update
    #   - no match -> insert
    print(f"  [{m.memory_type}] {m.content}")

# ===========================================================================
# 9. Process memories with deletion intent
# ===========================================================================

# If a user asks to delete or remove a memory, the process_memories method
# can detect that intent and actually delete the matching memory.
created, deleted = store.process_memories(
    user_id="user_123",
    contents=[
        {
            "role": "user",
            "content": "Please forget that I mentioned the climate RAG pipeline. Delete that memory.",
        },
    ],
    extract=True,
)

print(f"Created: {len(created)} memories, Deleted: {len(deleted)} memories")
for d in deleted:
    print(f"  Deleted: [{d.memory_type}] {d.content}")
