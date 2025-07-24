import numpy as np

class ChamferSimilarity:
    @staticmethod
    def compute(query_vectors: np.ndarray, document_vectors: np.ndarray) -> float:
        if query_vectors.size == 0 or document_vectors.size == 0:
            return 0.0
        similarity_matrix = np.dot(query_vectors, document_vectors.T)
        return float(np.sum(np.max(similarity_matrix, axis=1)))
