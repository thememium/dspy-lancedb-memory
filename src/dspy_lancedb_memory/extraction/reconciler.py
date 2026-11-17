import dspy

from ..models import ReconciledMemory
from .signatures import ReconcileMemory


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
    ) -> dspy.Prediction:
        prediction: dspy.Prediction = self.reconcile(
            new_memory_content=new_memory_content,
            new_memory_type=new_memory_type,
            existing_memories=existing_memories,
        )
        reconciled: ReconciledMemory = prediction.reconciled

        action = reconciled.action.strip().lower()
        if action.startswith("keep"):
            action = "keep"
        elif action.startswith("update"):
            action = "update"
        elif action.startswith("create"):
            action = "create"
        else:
            action = "create"

        reconciled.action = action

        if (
            action in ("keep", "update")
            and not reconciled.memory_id
            and existing_memories
        ):
            reconciled.memory_id = str(existing_memories[0]["id"])

        if not reconciled.final_content:
            if action == "create":
                reconciled.final_content = new_memory_content
            elif existing_memories:
                reconciled.final_content = str(existing_memories[0]["content"])
            else:
                reconciled.final_content = new_memory_content

        if not reconciled.final_type:
            reconciled.final_type = (
                new_memory_type
                if action == "create"
                else str(existing_memories[0]["type"])
                if existing_memories
                else new_memory_type
            )

        return dspy.Prediction(reconciled=reconciled)
