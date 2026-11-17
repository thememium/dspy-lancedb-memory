import dspy

from ..models import MemoryItem, MemoryType, memory_type_from_string
from .signatures import ExtractMemory


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

    def forward(self, messages: list[dict[str, str]]) -> dspy.Prediction:
        prediction: dspy.Prediction = self.extract(messages=messages)
        items: list[MemoryItem] = prediction.memories
        if not isinstance(items, list):
            items = [items]

        cleaned: list[tuple[str, MemoryType | str, dict]] = []
        for item in items:
            content = item.content.strip()
            if not content:
                continue
            cleaned.append(
                (
                    content,
                    memory_type_from_string(item.type),
                    dict(getattr(item, "metadata", {}) or {}),
                )
            )

        return dspy.Prediction(memories=cleaned)
