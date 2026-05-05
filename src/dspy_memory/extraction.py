import dspy

from .config import configure_runtime
from .models import MemoryItem, MemoryType, memory_type_from_string


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
    """Wraps ExtractMemory in ChainOfThought for reliable extraction."""

    def __init__(self):
        super().__init__()
        configure_runtime()
        self.extract = dspy.ChainOfThought(ExtractMemory)

    def forward(self, messages: list[dict[str, str]]) -> list[tuple[str, MemoryType]]:
        prediction: dspy.Prediction = self.extract(messages=messages)
        items: list[MemoryItem] = prediction.memories
        if not isinstance(items, list):
            items = [items]

        cleaned: list[tuple[str, MemoryType]] = []
        for item in items:
            content = item.content.strip()
            if not content:
                continue
            cleaned.append((content, memory_type_from_string(item.type)))

        return cleaned
