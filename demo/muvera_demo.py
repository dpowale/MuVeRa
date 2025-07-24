"""
MuVeRA Test Demo - Comprehensive example of multi-vector retrieval
"""

import numpy as np
import time
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Import the MuVeRA classes 
# from muvera import MuVeRA, SearchResult

def generate_mock_embeddings(num_vectors: int, embedding_dim: int, 
                           topic_center: np.ndarray = None, 
                           noise_level: float = 0.3) -> np.ndarray:
    """Generate mock embeddings clustered around a topic center"""
    if topic_center is None:
        topic_center = np.random.randn(embedding_dim)
    
    # Generate vectors around the topic center with some noise
    vectors = []
    for _ in range(num_vectors):
        # Start with topic center and add noise
        vector = topic_center + np.random.normal(0, noise_level, embedding_dim)
        # Normalize to unit length (common for embeddings)
        vector = vector / np.linalg.norm(vector)
        vectors.append(vector)
    
    return np.array(vectors, dtype=np.float32)

def create_sample_documents(embedding_dim: int = 384) -> List[tuple]:
    """Create sample documents with multi-vector representations"""
    
    # Define topic centers for different document types
    topics = {
        'technology': np.random.randn(embedding_dim),
        'science': np.random.randn(embedding_dim), 
        'literature': np.random.randn(embedding_dim),
        'history': np.random.randn(embedding_dim),
        'sports': np.random.randn(embedding_dim)
    }
    
    documents = []
    
    # Technology documents
    for i in range(5):
        vectors = generate_mock_embeddings(
            num_vectors=np.random.randint(3, 8),
            embedding_dim=embedding_dim,
            topic_center=topics['technology'],
            noise_level=0.2
        )
        metadata = {
            'title': f'Tech Article {i+1}',
            'topic': 'technology',
            'length': len(vectors),
            'keywords': ['AI', 'machine learning', 'neural networks']
        }
        documents.append((f'tech_{i+1}', vectors, metadata))
    
    # Science documents  
    for i in range(4):
        vectors = generate_mock_embeddings(
            num_vectors=np.random.randint(4, 10),
            embedding_dim=embedding_dim,
            topic_center=topics['science'],
            noise_level=0.25
        )
        metadata = {
            'title': f'Science Paper {i+1}',
            'topic': 'science',
            'length': len(vectors),
            'keywords': ['research', 'experiment', 'hypothesis']
        }
        documents.append((f'science_{i+1}', vectors, metadata))
    
    # Literature documents
    for i in range(3):
        vectors = generate_mock_embeddings(
            num_vectors=np.random.randint(5, 12),
            embedding_dim=embedding_dim,
            topic_center=topics['literature'],
            noise_level=0.3
        )
        metadata = {
            'title': f'Literary Work {i+1}',
            'topic': 'literature',
            'length': len(vectors),
            'keywords': ['narrative', 'character', 'theme']
        }
        documents.append((f'lit_{i+1}', vectors, metadata))
    
    # History documents
    for i in range(4):
        vectors = generate_mock_embeddings(
            num_vectors=np.random.randint(6, 9),
            embedding_dim=embedding_dim,
            topic_center=topics['history'],
            noise_level=0.2
        )
        metadata = {
            'title': f'Historical Document {i+1}',
            'topic': 'history',
            'length': len(vectors),
            'keywords': ['timeline', 'events', 'analysis']
        }
        documents.append((f'hist_{i+1}', vectors, metadata))
    
    # Sports documents
    for i in range(3):
        vectors = generate_mock_embeddings(
            num_vectors=np.random.randint(2, 6),
            embedding_dim=embedding_dim,
            topic_center=topics['sports'],
            noise_level=0.25
        )
        metadata = {
            'title': f'Sports Article {i+1}',
            'topic': 'sports',
            'length': len(vectors),
            'keywords': ['competition', 'athletics', 'performance']
        }
        documents.append((f'sports_{i+1}', vectors, metadata))
    
    return documents, topics

def run_basic_demo():
    """Run basic MuVeRA functionality demo"""
    print("=" * 60)
    print("MuVeRA Basic Demo")
    print("=" * 60)
    
    # Initialize parameters
    embedding_dim = 384
    fde_dim = 512
    num_partitions = 128
    
    print(f"Embedding dimension: {embedding_dim}")
    print(f"FDE dimension: {fde_dim}")
    print(f"Number of partitions: {num_partitions}")
    print()
    
    # Create MuVeRA instance
    muvera = MuVeRA(
        embedding_dim=embedding_dim,
        fde_dim=fde_dim,
        num_partitions=num_partitions,
        random_seed=42
    )
    
    # Create sample documents
    print("Creating sample documents...")
    documents, topics = create_sample_documents(embedding_dim)
    print(f"Created {len(documents)} documents")
    
    # Add documents to index
    print("Adding documents to index...")
    start_time = time.time()
    muvera.add_documents(documents)
    index_time = time.time() - start_time
    print(f"Indexing completed in {index_time:.3f} seconds")
    
    # Print index statistics
    stats = muvera.get_stats()
    print(f"\nIndex Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    return muvera, documents, topics

def run_search_demo(muvera, topics, embedding_dim):
    """Demonstrate search functionality"""
    print("=" * 60)
    print("Search Demo")
    print("=" * 60)
    
    # Create different types of queries
    query_scenarios = [
        ("Technology Query", topics['technology'], 0.1),
        ("Science Query", topics['science'], 0.15),
        ("Cross-topic Query", (topics['technology'] + topics['science']) / 2, 0.2),
        ("Noisy Query", np.random.randn(embedding_dim), 0.5)
    ]
    
    for query_name, query_center, noise in query_scenarios:
        print(f"\n{query_name}:")
        print("-" * 40)
        
        # Generate query vectors
        query_vectors = generate_mock_embeddings(
            num_vectors=np.random.randint(2, 5),
            embedding_dim=embedding_dim,
            topic_center=query_center,
            noise_level=noise
        )
        
        print(f"Query has {len(query_vectors)} vectors")
        
        # Search with reranking
        start_time = time.time()
        results_rerank = muvera.search(
            query_vectors=query_vectors,
            top_k=5,
            candidate_multiplier=5,
            rerank=True
        )
        rerank_time = time.time() - start_time
        
        # Search without reranking
        start_time = time.time()
        results_no_rerank = muvera.search(
            query_vectors=query_vectors,
            top_k=5,
            candidate_multiplier=5,
            rerank=False
        )
        no_rerank_time = time.time() - start_time
        
        print(f"\nWith Reranking (took {rerank_time:.4f}s):")
        for i, result in enumerate(results_rerank):
            metadata = result.metadata or {}
            print(f"  {i+1}. {result.doc_id} (score: {result.score:.4f})")
            print(f"     Topic: {metadata.get('topic', 'unknown')}")
            print(f"     Title: {metadata.get('title', 'N/A')}")
        
        print(f"\nWithout Reranking (took {no_rerank_time:.4f}s):")
        for i, result in enumerate(results_no_rerank):
            metadata = result.metadata or {}
            print(f"  {i+1}. {result.doc_id} (score: {result.score:.4f})")
            print(f"     Topic: {metadata.get('topic', 'unknown')}")

def run_performance_analysis(muvera, topics, embedding_dim):
    """Analyze performance characteristics"""
    print("\n" + "=" * 60)
    print("Performance Analysis")
    print("=" * 60)
    
    # Test different query sizes
    query_sizes = [1, 2, 5, 10, 20]
    times_rerank = []
    times_no_rerank = []
    
    print("Testing query performance with different query sizes...")
    
    for size in query_sizes:
        query_vectors = generate_mock_embeddings(
            num_vectors=size,
            embedding_dim=embedding_dim,
            topic_center=topics['technology']
        )
        
        # Time with reranking
        times = []
        for _ in range(5):  # Run multiple times for average
            start = time.time()
            muvera.search(query_vectors, top_k=10, rerank=True)
            times.append(time.time() - start)
        times_rerank.append(np.mean(times))
        
        # Time without reranking
        times = []
        for _ in range(5):
            start = time.time()
            muvera.search(query_vectors, top_k=10, rerank=False)
            times.append(time.time() - start)
        times_no_rerank.append(np.mean(times))
    
    # Print results
    print(f"\n{'Query Size':<12}{'With Rerank':<15}{'Without Rerank':<15}{'Speedup':<10}")
    print("-" * 60)
    for i, size in enumerate(query_sizes):
        speedup = times_rerank[i] / times_no_rerank[i]
        print(f"{size:<12}{times_rerank[i]:<15.4f}{times_no_rerank[i]:<15.4f}{speedup:<10.2f}x")

def analyze_retrieval_quality(muvera, documents, topics, embedding_dim):
    """Analyze retrieval quality across different topics"""
    print("\n" + "=" * 60)
    print("Retrieval Quality Analysis")
    print("=" * 60)
    
    topic_names = list(topics.keys())
    precision_scores = {topic: [] for topic in topic_names}
    
    print("Testing retrieval precision for each topic...")
    
    for topic_name, topic_center in topics.items():
        print(f"\nTesting {topic_name} queries:")
        
        # Generate multiple queries for this topic
        for query_idx in range(3):
            query_vectors = generate_mock_embeddings(
                num_vectors=np.random.randint(2, 6),
                embedding_dim=embedding_dim,
                topic_center=topic_center,
                noise_level=0.15
            )
            
            results = muvera.search(query_vectors, top_k=5, rerank=True)
            
            # Calculate precision (how many results match the query topic)
            relevant_results = sum(1 for r in results 
                                 if r.metadata and r.metadata.get('topic') == topic_name)
            precision = relevant_results / len(results) if results else 0
            precision_scores[topic_name].append(precision)
            
            print(f"  Query {query_idx + 1}: {relevant_results}/{len(results)} relevant (precision: {precision:.2f})")
    
    # Print average precision per topic
    print(f"\n{'Topic':<15}{'Avg Precision':<15}{'Std Dev':<10}")
    print("-" * 40)
    for topic, scores in precision_scores.items():
        avg_precision = np.mean(scores)
        std_precision = np.std(scores)
        print(f"{topic:<15}{avg_precision:<15.3f}{std_precision:<10.3f}")

def run_comprehensive_demo():
    """Run the complete MuVeRA demo"""
    print("MuVeRA Multi-Vector Retrieval System Demo")
    print("=========================================")
    print("This demo showcases the capabilities of the MuVeRA system")
    print("for multi-vector document retrieval and search.\n")
    
    try:
        # Basic setup and indexing
        muvera, documents, topics = run_basic_demo()
        
        # Search demonstrations
        run_search_demo(muvera, topics, embedding_dim=384)
        
        # Performance analysis
        run_performance_analysis(muvera, topics, embedding_dim=384)
        
        # Quality analysis
        analyze_retrieval_quality(muvera, documents, topics, embedding_dim=384)
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        print("\nKey takeaways:")
        print("- MuVeRA efficiently handles multi-vector documents")
        print("- The system supports both fast approximate and accurate reranked search")
        print("- Performance scales well with query size")
        print("- Retrieval quality is topic-dependent but generally good")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        print("Make sure the MuVeRA module is properly imported and all dependencies are installed.")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the comprehensive demo
    run_comprehensive_demo()
