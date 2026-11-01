"""
Sampling and Stratification Module for Agent Memory Datasets
"""

import random
from typing import List, Dict, Optional
from collections import defaultdict

from src.memory.schema import MemoryEntry, TrustTier


def sample_stratified_memories(
    memories: List[MemoryEntry],
    target_count: int = 100,
    random_seed: int = 42
) -> List[MemoryEntry]:
    """
    Stratified sampling across attack_type, domain, and is_poisoned label.
    """
    random.seed(random_seed)
    if len(memories) <= target_count:
        return memories

    # Group by (is_poisoned, attack_type)
    groups: Dict[str, List[MemoryEntry]] = defaultdict(list)
    for m in memories:
        key = f"{m.is_poisoned}_{m.attack_type}_{m.domain}"
        groups[key].append(m)

    # Allocate quota per group
    samples_per_group = max(1, target_count // len(groups))
    sampled: List[MemoryEntry] = []

    for key, group_mems in groups.items():
        sampled.extend(random.sample(group_mems, min(len(group_mems), samples_per_group)))

    # If shortfall or excess, adjust
    if len(sampled) < target_count:
        remaining = [m for m in memories if m not in sampled]
        shortfall = target_count - len(sampled)
        if remaining:
            sampled.extend(random.sample(remaining, min(len(remaining), shortfall)))
    elif len(sampled) > target_count:
        sampled = random.sample(sampled, target_count)

    return sampled
