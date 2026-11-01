"""
Semantic Memory Retriever Module
Provides top-k dense vector retrieval, similarity scoring, and ranking metrics:
- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)
"""

from typing import List, Dict, Any, Optional
import numpy as np

from src.memory.schema import MemoryEntry
from src.memory.store import PersistentMemoryStore


class SemanticRetriever:
    """Retriever orchestrator for persistent memory queries."""
    def __init__(self, store: PersistentMemoryStore, default_top_k: int = 5):
        self.store = store
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[MemoryEntry]:
        k = top_k or self.default_top_k
        return self.store.query(query_text=query, top_k=k, where_filter=where_filter)


def compute_retrieval_metrics(
    queries: List[str],
    ground_truth_relevant_ids: List[List[str]],
    retriever: SemanticRetriever,
    k: int = 5
) -> Dict[str, float]:
    """
    Computes standard IR metrics: Recall@K, Precision@K, and Mean Reciprocal Rank (MRR).
    """
    if not queries or not ground_truth_relevant_ids:
        return {"recall@k": 0.0, "precision@k": 0.0, "mrr": 0.0, "k": k}

    recalls = []
    precisions = []
    reciprocal_ranks = []

    for q, target_ids in zip(queries, ground_truth_relevant_ids):
        if not target_ids:
            continue
        retrieved_entries = retriever.retrieve(q, top_k=k)
        retrieved_ids = [m.memory_id for m in retrieved_entries]
        
        target_set = set(target_ids)
        retrieved_set = set(retrieved_ids)
        hits = target_set.intersection(retrieved_set)

        # Recall@K: proportion of relevant items retrieved
        rec = len(hits) / len(target_set) if target_set else 0.0
        recalls.append(rec)

        # Precision@K: proportion of retrieved items that are relevant
        prec = len(hits) / k if k > 0 else 0.0
        precisions.append(prec)

        # MRR: reciprocal rank of the first relevant item
        rr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in target_set:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return {
        "recall@k": float(np.mean(recalls)) if recalls else 0.0,
        "precision@k": float(np.mean(precisions)) if precisions else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "k": k,
        "num_queries_evaluated": len(recalls)
    }
