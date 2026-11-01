"""
Build Memory Script
Indexes normalized memory entries into ChromaDB with local sentence embeddings.
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.memory.schema import MemoryEntry
from src.memory.store import PersistentMemoryStore


def main():
    print("=" * 60)
    print("PHASE 5 & 6: BUILDING CHROMADB PERSISTENT MEMORY STORE")
    print("=" * 60)

    input_file = "data/processed/normalized_memories.jsonl"
    if not os.path.exists(input_file):
        print(f"Error: Normalized memories file not found at {input_file}. Run scripts/preprocess_data.py first.")
        return

    memories = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                memories.append(MemoryEntry.model_validate_json(line))

    print(f"Loaded {len(memories)} memory entries from {input_file}.")
    print("Initializing ChromaDB collection 'agent_persistent_memory'...")

    store = PersistentMemoryStore(
        persist_directory="data/chroma_db",
        collection_name="agent_persistent_memory",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    store.reset()

    print("Batch indexing memories into ChromaDB with sentence-transformers...")
    store.add_memories(memories, batch_size=128)

    indexed_count = store.count()
    print(f"\nSuccessfully indexed {indexed_count} memories into persistent ChromaDB store at data/chroma_db.")


if __name__ == "__main__":
    main()
