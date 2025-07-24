# MuVeRA: Multi-Vector Retrieval with Fixed-Dimensional Encodings

[![arXiv](https://img.shields.io/badge/arXiv-2405.19504-b31b1b.svg)](https://arxiv.org/abs/2405.19504)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

MuVeRA is a fast, scalable **multi-vector document retrieval** system inspired by the research paper:  
**[MuVeRA: Making Multi-Vector Retrieval as Fast as Single-Vector Search](https://arxiv.org/abs/2405.19504)**.

It converts multi-vector representations from queries or documents into fixed-dimensional encodings (FDEs) to enable traditional single-vector search systems like **FAISS**. Thus, significantly improving retrieval speed and scalability.

---

## 🧭 Implementation Approach

MuVeRA bridges the gap between expressive multi-vector representations and fast single-vector retrieval. The key steps are:

1. **Multi-Vector to Fixed-Dimensional Encoding (FDE):**  
   Each query and document—originally represented by multiple dense vectors—is encoded into a single, fixed-length vector. This FDE is designed to approximate the pairwise similarity between the original multi-vector sets.

2. **Indexing and Retrieval via FAISS:**  
   The FDEs are indexed using **FAISS**, a high-performance library for similarity search. MuVeRA leverages FAISS’s optimized **Maximum Inner Product Search (MIPS)** to perform fast, large-scale retrieval.

3. **Re-ranking with Chamfer Similarity (optional):**  
   While FDEs enable fast coarse retrieval, MuVeRA optionally re-ranks the top candidates using the full **Chamfer similarity** computed between the original vector sets. This step improves accuracy by capturing the finer structure of multi-vector relationships.

This hybrid approach combines the **speed of single-vector search** with the **accuracy of multi-vector similarity**.

---

## 🔍 Key Features

- 🔁 Fixed-length encoding for multi-vector documents  
- 🚀 FAISS-based inner product retrieval  
- 🧮 Chamfer similarity re-ranking  
- 🔌 REST API, Airflow DAG, Docker support  (todo)
- 🧪 Unit tests and usage examples  


---

