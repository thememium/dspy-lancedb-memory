import dspy

from ..models import MemoryOperation
from .signatures import ExtractMemoryOperations


class MemoryOperationExtractor(dspy.Module):
    """Wraps a DSPy Signature in ChainOfThought for reliable memory
    operation extraction (create/update/delete intents)."""

    def __init__(self, signature=ExtractMemoryOperations):
        super().__init__()
        self.extract = dspy.ChainOfThought(signature)

    def forward(self, messages: list[dict[str, str]]) -> dspy.Prediction:
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

        return dspy.Prediction(operations=cleaned)
