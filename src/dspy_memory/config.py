import dspy

DEFAULT_LM_MODEL = "openrouter/openai/gpt-oss-120b"

_lm: dspy.LM | None = None


def configure_runtime(*, model: str = DEFAULT_LM_MODEL) -> dspy.LM:
    global _lm

    if _lm is None or _lm.model != model:
        _lm = dspy.LM(model=model)
        dspy.configure(lm=_lm)

    return _lm
