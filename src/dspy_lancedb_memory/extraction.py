import dspy

from .models import (
    MemoryItem,
    MemoryOperation,
    MemoryType,
    ReconciledMemory,
    memory_type_from_string,
)


class ExtractMemoryOperations(dspy.Signature):
    """
    Analyze a conversation turn and extract any memory operations the user wants to perform.

    Instructions
    ------------
    1. Read the provided messages carefully.
    2. Determine if the user is requesting any changes to their stored memories.
    3. For each detected intent, output a MemoryOperation:
       * ``action``      — one of: **create**, **update**, **delete**
       * ``content``     — for create/update: the new memory text; for delete: the
                         specific text of the memory to remove (if provided verbatim)
       * ``search_query``— for delete: a natural-language description of the memory
                         to delete (used when no exact text is given)
       * ``memory_type`` — the memory category (e.g., preference, semantic, episodic).
                         Use ``""`` if the type is not specified or unclear.

    Guidelines
    ----------
    - Create: user introduces a new fact, preference, or information explicitly.
    - Update: user corrects, refines, or changes an existing piece of information.
    - Delete: user explicitly asks to forget, remove, or delete a memory.
              Set ``content`` to the exact text if stated, otherwise use
              ``search_query`` to describe what should be removed.
    - If no memory-related operations are found, return an empty list.
    """

    messages: list[dict[str, str]] = dspy.InputField()
    operations: list[MemoryOperation] = dspy.OutputField(
        desc="All memory operations extracted from the conversation. Empty list if none."
    )


class MemoryOperationExtractor(dspy.Module):
    """Wraps a DSPy Signature in ChainOfThought for reliable memory
    operation extraction (create/update/delete intents)."""

    def __init__(self, signature=ExtractMemoryOperations):
        super().__init__()
        self.extract = dspy.ChainOfThought(signature)

    def forward(self, messages: list[dict[str, str]]) -> list[MemoryOperation]:
        prediction: dspy.Prediction = self.extract(messages=messages)
        ops: list[MemoryOperation] = prediction.operations
        if not isinstance(ops, list):
            ops = [ops] if ops else []

        cleaned: list[MemoryOperation] = []
        for op in ops:
            action = (op.action or "").strip().lower()
            if action not in ("create", "update", "delete"):
                continue
            op.action = action
            cleaned.append(op)

        return cleaned


class ExtractMemory(dspy.Signature):
    """
    Extract every salient memory from a conversation turn.

    Instructions
    ------------
    1. Read the provided messages carefully.
    2. Identify **all** distinct pieces of information worth remembering.
    3. For each memory output a ``MemoryItem`` with:
       * ``content`` – concise, self-contained text summarising the takeaway.
       * ``type``    – one of:
         **preference** (user tastes, likes/dislikes, tone, tools),
         **semantic**   (stable facts / biographical data),
         **episodic**   (a specific event, task, decision, or outcome),
         **procedural** (a rule, workflow, step, pattern, or how-to),
         **summary**    (compressed conversation / task gist),
         **artifact**   (links, paths, IDs, references to files / PRs / docs).
    """

    messages: list[dict[str, str]] = dspy.InputField()
    memories: list[MemoryItem] = dspy.OutputField(
        desc="All extracted memories as structured MemoryItem objects."
    )


class MemoryExtractor(dspy.Module):
    """Wraps a DSPy Signature in ChainOfThought for reliable memory extraction.

    Parameters
    ----------
    signature :
        A DSPy ``Signature`` class whose output field is a ``list[Model]``
        where each model has ``.content`` (str) and ``.type`` (str).
        Defaults to :class:`ExtractMemory`.
    """

    def __init__(self, signature=ExtractMemory):
        super().__init__()
        self.extract = dspy.ChainOfThought(signature)

    def forward(
        self, messages: list[dict[str, str]]
    ) -> list[tuple[str, MemoryType | str]]:
        prediction: dspy.Prediction = self.extract(messages=messages)
        items: list[MemoryItem] = prediction.memories
        if not isinstance(items, list):
            items = [items]

        cleaned: list[tuple[str, MemoryType | str]] = []
        for item in items:
            content = item.content.strip()
            if not content:
                continue
            cleaned.append((content, memory_type_from_string(item.type)))

        return cleaned


class ReconcileMemory(dspy.Signature):
    """
    Compare a newly-extracted memory with existing stored memories and
    decide the right action:

    1. **keep** — the new memory is a duplicate or subset of existing
       information. No change needed.
    2. **update** — the new memory adds, refines, or corrects an existing
       memory. Synthesize a unified version that preserves the best of both.
    3. **create** — the new memory is genuinely distinct and should become a
       new stored row.

    Instructions
    ------------
    - If the existing memory covers the same subject/topic and the new memory
      adds details (e.g., first name → full name, general food → specific
      food), produce an **update** with a synthesized combined version.
    - If the new memory is just a restatement or paraphrase with no new info,
      produce **keep**.
    - Only produce **create** when the new memory is clearly about a different
      subject or fact.
    - Always return ``final_content`` as a single concise sentence.
    - For **keep** and **update**, always include the correct ``memory_id``.
    """

    new_memory_content: str = dspy.InputField(desc="The newly extracted memory text.")
    new_memory_type: str = dspy.InputField(
        desc="The type/category of the new memory (e.g., semantic, preference)."
    )
    existing_memories: list[dict] = dspy.InputField(
        desc="List of existing memories that are semantically similar. Each dict "
        "has 'id' (str), 'content' (str), and 'type' (str)."
    )
    reconciled: ReconciledMemory = dspy.OutputField(
        desc="The decision: action, memory_id, final_content, final_type."
    )


class MemoryReconciler(dspy.Module):
    """Wraps a DSPy Signature in ChainOfThought for reliable memory
    reconciliation.

    Parameters
    ----------
    signature :
        A DSPy ``Signature`` class for memory reconciliation.
        Defaults to :class:`ReconcileMemory`.
    """

    def __init__(self, signature=ReconcileMemory):
        super().__init__()
        self.reconcile = dspy.ChainOfThought(signature)

    def forward(
        self,
        *,
        new_memory_content: str,
        new_memory_type: str,
        existing_memories: list[dict],
    ) -> ReconciledMemory:
        prediction: dspy.Prediction = self.reconcile(
            new_memory_content=new_memory_content,
            new_memory_type=new_memory_type,
            existing_memories=existing_memories,
        )
        reconciled: ReconciledMemory = prediction.reconciled

        # Normalize and guard the action field
        action = reconciled.action.strip().lower()
        if action.startswith("keep"):
            action = "keep"
        elif action.startswith("update"):
            action = "update"
        elif action.startswith("create"):
            action = "create"
        else:
            # Fallback — if the LLM is uncertain, default to create so we
            # never silently discard novel information.
            action = "create"

        reconciled.action = action

        # Ensure memory_id is set for keep / update actions
        if action in ("keep", "update") and not reconciled.memory_id:
            if existing_memories:
                reconciled.memory_id = str(existing_memories[0]["id"])

        # Ensure final_content is set
        if not reconciled.final_content:
            if action == "create":
                reconciled.final_content = new_memory_content
            elif existing_memories:
                reconciled.final_content = str(existing_memories[0]["content"])
            else:
                reconciled.final_content = new_memory_content

        # Ensure final_type is set
        if not reconciled.final_type:
            reconciled.final_type = (
                new_memory_type
                if action == "create"
                else str(existing_memories[0]["type"])
                if existing_memories
                else new_memory_type
            )

        return reconciled
