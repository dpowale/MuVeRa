import numpy as np
import pytest
from muvera import MuVeRA, MultiVectorDocument, SearchResult

EMBEDDING_DIM = 128

# Generate dummy data
def generate_vectors(num_vectors=10, dim=EMBEDDING_DIM):
    return np.random.rand(num_vectors, dim).astype(np.float32)

def generate_documents(n=5):
    return [
        (f"doc_{i}", generate_vectors(5), {"title": f"Document {i}"})
        for i in range(n)
    ]

@pytest.fixture
def muvera():
    return MuVeRA(embedding_dim=EMBEDDING_DIM)

def test_add_documents(muvera):
    documents = generate_documents(3)
    muvera.add_documents(documents)
    stats = muvera.get_stats()
    assert stats["num_documents"] == 3
    assert stats["embedding_dimension"] == EMBEDDING_DIM

def test_search_returns_results(muvera):
    documents = generate_documents(5)
    muvera.add_documents(documents)

    query = generate_vectors(5)
    results = muvera.search(query, top_k=3)

    assert isinstance(results, list)
    assert len(results) == 3
    for result in results:
        assert isinstance(result, SearchResult)
        assert result.score >= 0.0
        assert isinstance(result.doc_id, str)

def test_search_with_rerank_false(muvera):
    documents = generate_documents(5)
    muvera.add_documents(documents)

    query = generate_vectors(5)
    results = muvera.search(query, top_k=2, rerank=False)
    assert len(results) == 2

def test_search_empty(muvera):
    query = generate_vectors(5)
    results = muvera.search(query)
    assert results == []

def test_duplicate_ids(muvera):
    doc = generate_vectors(5)
    documents = [("duplicate_id", doc, {"meta": 1}), ("duplicate_id", doc, {"meta": 2})]
    muvera.add_documents(documents)
    stats = muvera.get_stats()
    assert stats["num_documents"] == 1  # second one overwrites
