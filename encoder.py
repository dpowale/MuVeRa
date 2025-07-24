import numpy as np

class FixedDimensionalEncoder:
    def __init__(self, embedding_dim: int, fde_dim: int = 1024, num_partitions: int = 256, random_seed: int = 42):
        self.embedding_dim = embedding_dim
        self.fde_dim = fde_dim
        self.num_partitions = num_partitions
        np.random.seed(random_seed)
        self.hyperplanes = np.random.randn(num_partitions, embedding_dim).astype(np.float32)
        self.hyperplanes /= np.linalg.norm(self.hyperplanes, axis=1, keepdims=True)
        self.coords_per_partition = max(1, fde_dim // num_partitions)

    def _partition_vectors(self, vectors: np.ndarray) -> np.ndarray:
        projections = np.dot(vectors, self.hyperplanes.T)
        binary_codes = projections > 0
        hashes = np.packbits(binary_codes.astype(np.uint8), axis=1)
        return np.sum(hashes, axis=1) % self.num_partitions

    def _encode(self, vectors: np.ndarray, mode: str = 'sum') -> np.ndarray:
        fde = np.zeros(self.fde_dim, dtype=np.float32)
        partitions = self._partition_vectors(vectors)
        for partition_idx in np.unique(partitions):
            partition_vectors = vectors[partitions == partition_idx]
            if partition_vectors.size == 0:
                continue
            aggregated = np.sum(partition_vectors, axis=0) if mode == 'sum' else np.mean(partition_vectors, axis=0)
            start = (partition_idx * self.coords_per_partition) % self.fde_dim
            end = min(start + self.coords_per_partition, self.fde_dim)
            fde[start:end] = aggregated[:end - start]
        return fde

    def encode_query(self, query_vectors: np.ndarray) -> np.ndarray:
        return self._encode(query_vectors, mode='sum')

    def encode_document(self, document_vectors: np.ndarray) -> np.ndarray:
        return self._encode(document_vectors, mode='mean')
