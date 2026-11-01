"""
Persistent Vector Memory Store using ChromaDB
Provides insertion, retrieval, metadata filtering, reset, and provenance preservation.
"""

import os
import json
from typing import List, Optional, Dict, Any
from src.memory.schema import MemoryEntry, TrustTier, ContentType
from src.nlp.embeddings import get_embedding_model


class PersistentMemoryStore:
    """
    ChromaDB Persistent Memory Storage with metadata and provenance fields.
    """
    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "agent_persistent_memory",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.embedder = get_embedding_model(embedding_model_name)
        self._client = None
        self._collection = None

    def _init_db(self):
        if self._collection is None:
            import chromadb
            from chromadb.config import Settings
            os.makedirs(self.persist_directory, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

    def reset(self):
        """Clears all records from the collection."""
        self._init_db()
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_memories(self, memories: List[MemoryEntry], batch_size: int = 64):
        """Batch inserts MemoryEntry objects into ChromaDB."""
        if not memories:
            return
        self._init_db()

        for i in range(0, len(memories), batch_size):
            batch = memories[i:i + batch_size]
            ids = [m.memory_id for m in batch]
            documents = [m.content for m in batch]
            embeddings = self.embedder.encode(documents, normalize=True).tolist()
            metadatas = [
                {
                    "source_id": m.source_id,
                    "source_dataset": m.source_dataset,
                    "source_channel": m.source_channel,
                    "trust_tier": int(m.trust_tier),
                    "is_poisoned": int(m.is_poisoned),
                    "attack_type": m.attack_type,
                    "domain": m.domain,
                    "difficulty": m.difficulty,
                    "content_type": m.content_type.value,
                    "dormancy_days": m.dormancy_days,
                    "synthetic_timestamp": m.synthetic_timestamp,
                    "taint_chain_json": json.dumps(m.taint_chain)
                }
                for m in batch
            ]
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[MemoryEntry]:
        """Semantic top-k search returning fully populated MemoryEntry objects."""
        self._init_db()
        query_emb = self.embedder.encode([query_text], normalize=True).tolist()

        kwargs: Dict[str, Any] = {
            "query_embeddings": query_emb,
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self._collection.query(**kwargs)

        retrieved: List[MemoryEntry] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return retrieved

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for rank, (mem_id, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances), start=1):
            sim_score = max(0.0, 1.0 - float(dist))  # Cosine distance to similarity
            taint_chain = json.loads(meta.get("taint_chain_json", "[]"))
            
            entry = MemoryEntry(
                memory_id=mem_id,
                content=doc,
                source_id=meta.get("source_id", ""),
                source_dataset=meta.get("source_dataset", "unknown"),
                source_channel=meta.get("source_channel", "unknown"),
                trust_tier=TrustTier(meta.get("trust_tier", 0)),
                synthetic_timestamp=meta.get("synthetic_timestamp", ""),
                dormancy_days=int(meta.get("dormancy_days", 0)),
                taint_chain=taint_chain,
                is_poisoned=bool(meta.get("is_poisoned", False)),
                attack_type=meta.get("attack_type", "none"),
                domain=meta.get("domain", "general"),
                difficulty=meta.get("difficulty", "easy"),
                content_type=ContentType(meta.get("content_type", "FACT")),
                retrieval_rank=rank,
                similarity_score=sim_score
            )
            retrieved.append(entry)

        return retrieved

    def count(self) -> int:
        self._init_db()
        return self._collection.count()
