from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

@dataclass
class MultiVectorDocument:
    id: str
    vectors: np.ndarray
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SearchResult:
    doc_id: str
    score: float
    metadata: Optional[Dict[str, Any]] = None
