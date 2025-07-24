import threading
import numpy as np
import faiss
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Dict, Any
from .encoder import FixedDimensionalEncoder
from .similarity import ChamferSimilarity
from .document import MultiVectorDocument, SearchResult

class MuVeRAIndex:
    def __init__(self, embedding_dim: int, fde_dim: int = 1024, num_partitions: int = 256, random_seed: int = 42):
        self.encoder = FixedDimensionalEncoder(embedding_dim, fde_dim, num_partitions, random_seed)
        self.documents: Dict[str, MultiVectorDocument] = {}
        self.document_fdes: List[np.ndarray] = []
        self.doc_ids: List[str] = []
        self.lock = threading.Lock()
        self.faiss_index = faiss.IndexFlatIP(fde_dim)

    def add_documents(self, docs: List[Tuple[str, np.ndarray, Optional[Dict[str, Any]]]]):
        new_fdes = []
        new_ids = []
        with self.lock:
            for doc_id, vectors, metadata in docs:
                vectors = vectors.astype(np.float32, copy=False)
                fde = self.encoder.encode_document(vectors)
                self.documents[doc_id] = MultiVectorDocument(doc_id, vectors, metadata)
                new_ids.append(doc_id)
                new_fdes.append(fde)
            self.doc_ids.extend(new_ids)
            self.document_fdes.extend(new_fdes)
            self._refresh_index()

    def _refresh_index(self):
        if not self.document_fdes:
            return
        matrix = np.vstack(self.document_fdes).astype(np.float32)
        self.faiss_index.reset()
        self.faiss_index.add(matrix)

    def search(self, query_vectors: np.ndarray, top_k: int = 10, candidate_multiplier: int = 10, rerank: bool = True) -> List[SearchResult]:
        query_fde = self.encoder.encode_query(query_vectors.astype(np.float32)).reshape(1, -1)
        candidate_count = min(top_k * candidate_multiplier, len(self.doc_ids))

        if candidate_count == 0:
            return []

        scores, indices = self.faiss_index.search(query_fde, candidate_count)
        candidates = indices[0]

        if rerank:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(ChamferSimilarity.compute, query_vectors, self.documents[self.doc_ids[idx]].vectors)
                    for idx in candidates
                ]
                results = [(self.doc_ids[candidates[i]], future.result()) for i, future in enumerate(futures)]
                results.sort(key=lambda x: x[1], reverse=True)
                return [
                    SearchResult(doc_id, score, self.documents[doc_id].metadata)
                    for doc_id, score in results[:top_k]
                ]
        else:
            return [
                SearchResult(self.doc_ids[idx], float(scores[0][i]), self.documents[self.doc_ids[idx]].metadata)
                for i, idx in enumerate(candidates[:top_k])
            ]

    def get_stats(self) -> Dict[str, Any]:
        return {
            'num_documents': len(self.documents),
            'fde_dimension': self.encoder.fde_dim,
            'num_partitions': self.encoder.num_partitions,
            'embedding_dimension': self.encoder.embedding_dim
        }
