"""Tests for text embedding utilities."""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from src.text_encoder import encode_text, get_embedding_model


class TestGetEmbeddingModel:
    """Tests for get_embedding_model function."""

    def test_model_caching(self):
        """Test that models are cached and reused."""
        # Clear cache first
        from src.text_encoder import _model_cache

        _model_cache.clear()

        model1 = get_embedding_model()
        model2 = get_embedding_model()

        # Should return the same instance (cached)
        assert model1 is model2

    def test_different_models(self):
        """Test that different model names return different models."""
        from src.text_encoder import _model_cache

        _model_cache.clear()

        # Use a smaller model for testing
        model1 = get_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        model2 = get_embedding_model("sentence-transformers/all-mpnet-base-v2")

        # Should be different instances
        assert model1 is not model2

    def test_model_type(self):
        """Test that function returns SentenceTransformer instance."""
        from sentence_transformers import SentenceTransformer

        model = get_embedding_model()
        assert isinstance(model, SentenceTransformer)


class TestEncodeText:
    """Tests for encode_text function."""

    def test_single_text_returns_1d_array(self):
        """Test that encoding a single text returns a 1D numpy array."""
        text = "This is a test description for SMT benchmarks."
        embedding = encode_text(text, show_progress=False)

        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
        assert embedding.shape[0] > 0  # Should have embedding dimension

    def test_multiple_texts_returns_2d_array(self):
        """Test that encoding multiple texts returns a 2D numpy array."""
        texts = [
            "First description",
            "Second description",
            "Third description",
        ]
        embeddings = encode_text(texts, show_progress=False)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim == 2
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] > 0  # Should have embedding dimension

    def test_single_vs_list_consistency(self):
        """Test that single text and list with one text produce same embedding."""
        text = "A single test description"
        embedding_single = encode_text(text, show_progress=False)
        embedding_list = encode_text([text], show_progress=False)

        # Should be the same (list version is 2D, single is 1D)
        assert np.array_equal(embedding_single, embedding_list[0])

    def test_empty_list_returns_empty_array(self):
        """Test that encoding an empty list returns an empty array."""
        embeddings = encode_text([], show_progress=False)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (0,)

    def test_normalize_parameter(self):
        """Test that normalize parameter works correctly."""
        text = "Test description"
        embedding_normalized = encode_text(text, normalize=True, show_progress=False)
        embedding_not_normalized = encode_text(
            text, normalize=False, show_progress=False
        )

        # Normalized embedding should have unit length
        norm = np.linalg.norm(embedding_normalized)
        assert np.isclose(norm, 1.0, atol=1e-6)

        # Non-normalized might not have unit length
        norm_not = np.linalg.norm(embedding_not_normalized)
        # It might still be close to 1, but we just check it's different behavior
        # or that normalize=True actually normalizes

    def test_batch_size_parameter(self):
        """Test that batch_size parameter is accepted."""
        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
        # Should not raise an error with custom batch size
        embeddings = encode_text(texts, batch_size=2, show_progress=False)
        assert embeddings.shape[0] == len(texts)

    def test_different_texts_produce_different_embeddings(self):
        """Test that different texts produce different embeddings."""
        text1 = "This is a description about linear arithmetic"
        text2 = "This is a description about bit vectors"
        embedding1 = encode_text(text1, show_progress=False)
        embedding2 = encode_text(text2, show_progress=False)

        # Embeddings should be different
        assert not np.array_equal(embedding1, embedding2)

    def test_similar_texts_produce_similar_embeddings(self):
        """Test that similar texts produce similar embeddings."""
        text1 = "Linear arithmetic solver for SMT"
        text2 = "Linear arithmetic solving in SMT"
        embedding1 = encode_text(text1, show_progress=False)
        embedding2 = encode_text(text2, show_progress=False)

        # Compute cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        cosine_sim = dot_product / (norm1 * norm2)

        # Similar texts should have high cosine similarity (> 0.7)
        assert cosine_sim > 0.7

    def test_custom_model_name(self):
        """Test that custom model name parameter works."""
        # Use a smaller model for faster testing
        text = "Test description"
        embedding = encode_text(
            text,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            show_progress=False,
        )
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1

    def test_show_progress_parameter(self):
        """Test that show_progress parameter doesn't break functionality."""
        texts = ["Text 1", "Text 2", "Text 3"]
        # Should work with both True and False
        embeddings_false = encode_text(texts, show_progress=False)
        embeddings_true = encode_text(texts, show_progress=True)

        assert np.array_equal(embeddings_false, embeddings_true)

    def test_whitespace_handling(self):
        """Test that texts with extra whitespace are handled."""
        text1 = "Normal text"
        text2 = "  Normal   text  "
        embedding1 = encode_text(text1, show_progress=False)
        embedding2 = encode_text(text2, show_progress=False)

        # Should produce similar embeddings (model handles whitespace)
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        cosine_sim = dot_product / (norm1 * norm2)
        assert cosine_sim > 0.9  # Very similar

    def test_long_text_truncation(self):
        """Test that long texts are handled (truncated by model)."""
        # Create a very long text
        long_text = " ".join(["word"] * 1000)
        embedding = encode_text(long_text, show_progress=False)

        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
