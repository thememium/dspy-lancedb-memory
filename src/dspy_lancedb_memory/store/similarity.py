"""Text-normalisation and duplicate-detection heuristics for upserts."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_SEMANTIC_DUPLICATE_STOPWORDS = frozenset(
    {"a", "an", "the", "their", "his", "her", "our", "my", "your"}
)


def distance_to_similarity(distance: float) -> float:
    """Map LanceDB's L2 distance to a bounded ``0..1`` relevance score."""
    return 1.0 / (1.0 + max(float(distance), 0.0))


def normalize_semantic_content(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s]+", " ", text.lower())
    tokens = [
        token
        for token in normalized.split()
        if token and token not in _SEMANTIC_DUPLICATE_STOPWORDS
    ]
    return " ".join(tokens)


def shared_prefix_length(left: list[str], right: list[str]) -> int:
    shared = 0
    for left_token, right_token in zip(left, right, strict=False):
        if left_token != right_token:
            break
        shared += 1
    return shared


def shared_suffix_length(left: list[str], right: list[str]) -> int:
    shared = 0
    for left_token, right_token in zip(reversed(left), reversed(right), strict=False):
        if left_token != right_token:
            break
        shared += 1
    return shared


def semantic_match_action(
    *,
    new_content: str,
    existing_content: str,
    distance: float,
    similarity_threshold: float,
    skip_threshold: float = 0.85,
) -> str | None:
    """Classify a new memory against an existing one as skip, update, or new."""
    new_normalized = normalize_semantic_content(new_content)
    existing_normalized = normalize_semantic_content(existing_content)

    if not new_normalized or not existing_normalized:
        return None

    if new_normalized == existing_normalized:
        return "skip"

    new_tokens = new_normalized.split()
    existing_tokens = existing_normalized.split()
    new_set = set(new_tokens)
    existing_set = set(existing_tokens)
    union = new_set | existing_set
    jaccard = len(new_set & existing_set) / len(union) if union else 0.0
    prefix = shared_prefix_length(new_tokens, existing_tokens)
    suffix = shared_suffix_length(new_tokens, existing_tokens)
    sequence_ratio = SequenceMatcher(None, new_normalized, existing_normalized).ratio()
    contains_other = (
        new_normalized in existing_normalized or existing_normalized in new_normalized
    )
    cosine_similarity = distance_to_similarity(distance)
    same_slot_replacement = prefix >= max(
        2, min(len(new_tokens), len(existing_tokens)) - 1
    )
    new_is_richer = len(new_tokens) > len(existing_tokens) or (
        existing_normalized in new_normalized and new_normalized != existing_normalized
    )

    # --- Skip-threshold gate ------------------------------------------------
    # When cosine similarity is very high, the vectors are near-identical.
    # Skip the write unless the new content is genuinely richer or replaces
    # a specific slot value (e.g. "color is blue" → "color is red").
    # This prevents runaway duplicate memories across repeated upsert runs.
    if (
        cosine_similarity >= skip_threshold
        and not new_is_richer
        and not same_slot_replacement
    ):
        return "skip"

    same_fact = (
        (prefix >= 2 and jaccard >= 0.5)
        or (suffix >= 2 and jaccard >= 0.5)
        or (contains_other and (prefix >= 1 or suffix >= 1))
        or (same_slot_replacement and sequence_ratio >= 0.88)
    )
    if not same_fact or cosine_similarity < similarity_threshold:
        return None

    if sequence_ratio >= 0.94 and not new_is_richer and not same_slot_replacement:
        return "skip"

    return "update" if new_is_richer or same_slot_replacement else "skip"
