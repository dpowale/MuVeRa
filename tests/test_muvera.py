import numpy as np
from muvera import MuVeRA

def test_search_results():
    muvera = MuVeRA(embedding_dim=128)
    docs = [("doc1", np.random.rand(5, 128).astype(np.float32), {}),
            ("doc2", np.random.rand(7, 128).astype(np.float32), {})]
    muvera.add_documents(docs)
    query = np.random.rand(4, 128).astype(np.float32)
    results = muvera.search(query, top_k=2)
    assert len(results) == 2
    assert all(hasattr(r, "doc_id") and hasattr(r, "score") for r in results)
