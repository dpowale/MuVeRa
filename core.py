from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from .index import MuVeRAIndex
from .document import SearchResult

class MuVeRA:
    def __init__(self, embedding_dim: int, fde_dim: int = 1024, num_partitions: int = 256, random_seed: int = 42):
        self.index = MuVeRAIndex(embedding_dim, fde_dim, num_partitions, random_seed)

    def add_documents(self, documents: List[Tuple[str, np.ndarray, Optional[Dict[str, Any]]]]):
        self.index.add_documents(documents)

    def search(self, query_vectors: np.ndarray, **kwargs) -> List[SearchResult]:
        return self.index.search(query_vectors, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        return self.index.get_stats()
