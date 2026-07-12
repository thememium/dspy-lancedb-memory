"""Tests for dspy_lancedb_memory.reranking — covering LiteLLMReranker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

from dspy_lancedb_memory.reranking import LiteLLMReranker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_table():
    """Create a sample PyArrow table for reranking tests."""
    return pa.table(
        {
            "content": ["test content 1", "test content 2", "test content 3"],
            "id": ["id1", "id2", "id3"],
            "_distance": [0.1, 0.2, 0.3],
        }
    )


@pytest.fixture
def sample_fts_table():
    """Create a sample FTS result table."""
    return pa.table(
        {
            "content": ["test content 1", "test content 2", "test content 3"],
            "id": ["id1", "id2", "id3"],
            "_score": [0.9, 0.8, 0.7],
        }
    )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestLiteLLMRerankerInit:
    def test_default_values(self):
        reranker = LiteLLMReranker()
        assert reranker.model == "cohere/rerank-english-v3.0"
        assert reranker.column == "text"
        assert reranker.top_n is None
        assert reranker.api_base is None
        assert reranker.api_key is None

    def test_custom_values(self):
        reranker = LiteLLMReranker(
            model="custom/model",
            column="content",
            top_n=5,
            api_base="http://custom.api",
            api_key="test-key",
        )
        assert reranker.model == "custom/model"
        assert reranker.column == "content"
        assert reranker.top_n == 5
        assert reranker.api_base == "http://custom.api"
        assert reranker.api_key == "test-key"


# ---------------------------------------------------------------------------
# _attach_fallback_scores
# ---------------------------------------------------------------------------


class TestAttachFallbackScores:
    def test_with_distance_column(self, sample_table):
        reranker = LiteLLMReranker()
        result = reranker._attach_fallback_scores(sample_table)

        assert "_relevance_score" in result.column_names
        scores = result["_relevance_score"].to_pylist()
        # 1.0 / (1.0 + d) for each distance
        assert scores[0] == pytest.approx(1.0 / 1.1)
        assert scores[1] == pytest.approx(1.0 / 1.2)
        assert scores[2] == pytest.approx(1.0 / 1.3)

    def test_without_distance_column(self):
        reranker = LiteLLMReranker()
        table = pa.table(
            {
                "content": ["test 1", "test 2"],
                "id": ["id1", "id2"],
            }
        )
        result = reranker._attach_fallback_scores(table)

        assert "_relevance_score" in result.column_names
        scores = result["_relevance_score"].to_pylist()
        assert scores == [0.0, 0.0]


# ---------------------------------------------------------------------------
# _rerank with mocked API calls
# ---------------------------------------------------------------------------


class TestRerank:
    def test_rerank_with_litellm_and_api_base_and_key(self, sample_table):
        reranker = LiteLLMReranker(
            model="cohere/rerank-english-v3.0",
            column="content",
            api_base="http://custom.api",
            api_key="test-key",
        )

        mock_response = {
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 1, "relevance_score": 0.85},
                {"index": 2, "relevance_score": 0.75},
            ]
        }

        with patch(
            "dspy_lancedb_memory.reranking.rerank", return_value=mock_response
        ) as mock_rerank:
            result = reranker._rerank(sample_table, "test query")

        # Should pass api_base and api_key to litellm.rerank
        mock_rerank.assert_called_once()
        call_kwargs = mock_rerank.call_args[1]
        assert call_kwargs["api_base"] == "http://custom.api"
        assert call_kwargs["api_key"] == "test-key"
        assert "_relevance_score" in result.column_names

    def test_rerank_with_openrouter_model(self, sample_table):
        reranker = LiteLLMReranker(
            model="openrouter/cohere/rerank-4-fast", column="content"
        )

        mock_response = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.8},
                {"index": 1, "relevance_score": 0.7},
            ]
        }

        with patch.object(reranker, "_rerank_openrouter", return_value=mock_response):
            result = reranker._rerank(sample_table, "test query")

        assert "_relevance_score" in result.column_names

    def test_rerank_with_custom_api(self, sample_table):
        reranker = LiteLLMReranker(
            model="huggingface/model",
            api_base="http://custom.api",
            column="content",
        )

        mock_response = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.8},
                {"index": 2, "relevance_score": 0.7},
            ]
        }

        with patch.object(reranker, "_rerank_custom_api", return_value=mock_response):
            result = reranker._rerank(sample_table, "test query")

        assert "_relevance_score" in result.column_names

    def test_rerank_returns_fallback_on_exception(self, sample_table):
        reranker = LiteLLMReranker(model="cohere/rerank-english-v3.0", column="content")

        with patch(
            "dspy_lancedb_memory.reranking.rerank",
            side_effect=Exception("API Error"),
        ):
            result = reranker._rerank(sample_table, "test query")

        # Should have fallback scores
        assert "_relevance_score" in result.column_names
        scores = result["_relevance_score"].to_pylist()
        assert all(s > 0 for s in scores)

    def test_rerank_handles_empty_table(self):
        reranker = LiteLLMReranker(column="content")
        empty_table = pa.table({"content": [], "id": []})

        result = reranker._rerank(empty_table, "test query")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# _rerank_openrouter
# ---------------------------------------------------------------------------


class TestRerankOpenrouter:
    def test_rerank_openrouter_success(self):
        reranker = LiteLLMReranker(model="openrouter/cohere/rerank-4-fast")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.8},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch(
            "dspy_lancedb_memory.reranking.os.environ.get", return_value="test-key"
        ):
            with patch(
                "dspy_lancedb_memory.reranking.httpx.post", return_value=mock_response
            ) as mock_post:
                result = reranker._rerank_openrouter("test query", ["doc1", "doc2"])

        assert result["results"][0]["relevance_score"] == 0.9
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "openrouter.ai/api/v1/rerank" in call_args[0][0]

    def test_rerank_openrouter_with_top_n(self):
        reranker = LiteLLMReranker(model="openrouter/cohere/rerank-4-fast", top_n=5)

        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch(
            "dspy_lancedb_memory.reranking.os.environ.get", return_value="test-key"
        ):
            with patch(
                "dspy_lancedb_memory.reranking.httpx.post", return_value=mock_response
            ) as mock_post:
                reranker._rerank_openrouter("query", ["doc1"])

        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["top_n"] == 5

    def test_rerank_openrouter_raises_without_api_key(self):
        reranker = LiteLLMReranker(model="openrouter/cohere/rerank-4-fast")

        with patch("dspy_lancedb_memory.reranking.os.environ.get", return_value=None):
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                reranker._rerank_openrouter("query", ["doc1"])


# ---------------------------------------------------------------------------
# _rerank_custom_api
# ---------------------------------------------------------------------------


class TestRerankCustomApi:
    def test_rerank_custom_api_success(self):
        reranker = LiteLLMReranker(
            model="huggingface/model",
            api_base="http://custom.api",
            api_key="test-key",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.9}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch(
            "dspy_lancedb_memory.reranking.httpx.post", return_value=mock_response
        ) as mock_post:
            result = reranker._rerank_custom_api("query", ["doc1"])

        assert result["results"][0]["relevance_score"] == 0.9
        call_args = mock_post.call_args
        assert "custom.api/rerank" in call_args[0][0]
        assert call_args[1]["headers"]["Authorization"] == "Bearer test-key"

    def test_rerank_custom_api_without_api_key(self):
        reranker = LiteLLMReranker(
            model="huggingface/model",
            api_base="http://custom.api",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch(
            "dspy_lancedb_memory.reranking.httpx.post", return_value=mock_response
        ) as mock_post:
            reranker._rerank_custom_api("query", ["doc1"])

        call_args = mock_post.call_args
        assert "Authorization" not in call_args[1]["headers"]

    def test_rerank_custom_api_with_top_n(self):
        reranker = LiteLLMReranker(
            model="huggingface/model",
            api_base="http://custom.api",
            top_n=3,
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch(
            "dspy_lancedb_memory.reranking.httpx.post", return_value=mock_response
        ) as mock_post:
            reranker._rerank_custom_api("query", ["doc1"])

        call_args = mock_post.call_args
        assert call_args[1]["json"]["top_n"] == 3

    def test_rerank_custom_api_strips_provider_prefix(self):
        reranker = LiteLLMReranker(
            model="huggingface/my-model",
            api_base="http://custom.api",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch(
            "dspy_lancedb_memory.reranking.httpx.post", return_value=mock_response
        ) as mock_post:
            reranker._rerank_custom_api("query", ["doc1"])

        call_args = mock_post.call_args
        assert call_args[1]["json"]["model"] == "my-model"


# ---------------------------------------------------------------------------
# rerank_vector, rerank_fts, rerank_hybrid
# ---------------------------------------------------------------------------


class TestRerankInterface:
    def test_rerank_vector_drops_distance(self, sample_table):
        reranker = LiteLLMReranker(return_score="relevance", column="content")

        with patch.object(reranker, "_rerank") as mock_rerank:
            # Return a table with both _distance and _relevance_score
            reranked = sample_table.append_column(
                "_relevance_score", pa.array([0.9, 0.8, 0.7], type=pa.float32())
            )
            mock_rerank.return_value = reranked
            result = reranker.rerank_vector("query", sample_table)

        assert "_distance" not in result.column_names
        assert "_relevance_score" in result.column_names

    def test_rerank_vector_keeps_distance_when_score_all(self, sample_table):
        reranker = LiteLLMReranker(return_score="all", column="content")

        with patch.object(reranker, "_rerank") as mock_rerank:
            reranked = sample_table.append_column(
                "_relevance_score", pa.array([0.9, 0.8, 0.7], type=pa.float32())
            )
            mock_rerank.return_value = reranked
            result = reranker.rerank_vector("query", sample_table)

        assert "_distance" in result.column_names

    def test_rerank_fts_drops_score(self, sample_fts_table):
        reranker = LiteLLMReranker(return_score="relevance", column="content")

        with patch.object(reranker, "_rerank") as mock_rerank:
            reranked = sample_fts_table.append_column(
                "_relevance_score", pa.array([0.9, 0.8, 0.7], type=pa.float32())
            )
            mock_rerank.return_value = reranked
            result = reranker.rerank_fts("query", sample_fts_table)

        assert "_score" not in result.column_names
        assert "_relevance_score" in result.column_names

    def test_rerank_fts_keeps_score_when_score_all(self, sample_fts_table):
        reranker = LiteLLMReranker(return_score="all", column="content")

        with patch.object(reranker, "_rerank") as mock_rerank:
            reranked = sample_fts_table.append_column(
                "_relevance_score", pa.array([0.9, 0.8, 0.7], type=pa.float32())
            )
            mock_rerank.return_value = reranked
            result = reranker.rerank_fts("query", sample_fts_table)

        assert "_score" in result.column_names

    def test_rerank_hybrid_with_relevance_score(self, sample_table, sample_fts_table):
        reranker = LiteLLMReranker(return_score="relevance", column="content")

        # Mock both the merge and rerank methods
        with patch.object(reranker, "merge_results") as mock_merge:
            with patch.object(reranker, "_rerank") as mock_rerank:
                merged = pa.table(
                    {
                        "content": [
                            "test content 1",
                            "test content 2",
                            "test content 3",
                        ],
                        "id": ["id1", "id2", "id3"],
                        "_distance": [0.1, 0.2, 0.3],
                    }
                )
                mock_merge.return_value = merged

                reranked = merged.append_column(
                    "_relevance_score",
                    pa.array([0.95, 0.85, 0.75], type=pa.float32()),
                )
                mock_rerank.return_value = reranked

                with patch.object(reranker, "_keep_relevance_score") as mock_keep:
                    final_table = pa.table(
                        {
                            "content": [
                                "test content 1",
                                "test content 2",
                                "test content 3",
                            ],
                            "id": ["id1", "id2", "id3"],
                            "_relevance_score": [0.95, 0.85, 0.75],
                        }
                    )
                    mock_keep.return_value = final_table
                    _result = reranker.rerank_hybrid(
                        "query", sample_table, sample_fts_table
                    )

        # When score="relevance", merge_results should be called (not _merge_and_keep_scores)
        mock_merge.assert_called_once()
        # _keep_relevance_score should be called
        mock_keep.assert_called_once()
