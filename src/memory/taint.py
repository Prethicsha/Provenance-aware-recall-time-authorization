"""
Taint Propagation and Multi-Hop Provenance Module
Ensures that when memories are summarized, derived, or transformed by an LLM,
the taint chain is preserved and effective trust is strictly computed as min(parent_trusts).
"""

from typing import List, Optional
from datetime import datetime, timezone
from .schema import MemoryEntry, TrustTier, ContentType


def get_effective_trust(memories: List[MemoryEntry]) -> TrustTier:
    """
    Computes the effective trust tier when multiple memories influence a context.
    Rule: effective_trust = minimum trust tier among all contributing memories.
    If empty, returns TrustTier.UNTRUSTED_EXTERNAL as a fail-safe.
    """
    if not memories:
        return TrustTier.UNTRUSTED_EXTERNAL
    min_tier_val = min(int(m.trust_tier) for m in memories)
    return TrustTier(min_tier_val)


def propagate_taint(parent_memories: List[MemoryEntry], new_hop_id: str, transform_type: str = "agent_summary") -> List[str]:
    """
    Merges all parent taint chains and appends the new derivation hop.
    """
    merged_chain = []
    seen = set()
    for pm in parent_memories:
        for hop in pm.taint_chain:
            if hop not in seen:
                merged_chain.append(hop)
                seen.add(hop)
    
    # Calculate parent minimum trust
    parent_min_tier = get_effective_trust(parent_memories)
    new_hop = f"{transform_type}:{new_hop_id} [inherited_tier={int(parent_min_tier)}]"
    merged_chain.append(new_hop)
    return merged_chain


def derive_memory(
    parent_memories: List[MemoryEntry],
    new_content: str,
    new_memory_id: str,
    source_channel: str = "derived_summary",
    content_type: ContentType = ContentType.FACT,
    attack_type: Optional[str] = None,
    domain: Optional[str] = None,
    difficulty: Optional[str] = None,
    synthetic_timestamp: Optional[str] = None,
    dormancy_days: int = 0
) -> MemoryEntry:
    """
    Derives a new persistent memory entry from one or more parent memories (e.g. LLM summary).
    Inherits the minimum trust tier of parents (taint cannot be washed away by summarization).
    Inherits is_poisoned if ANY parent memory was poisoned.
    """
    if not parent_memories:
        raise ValueError("Cannot derive memory without parent memories.")

    effective_tier = get_effective_trust(parent_memories)
    inherited_poisoned = any(pm.is_poisoned for pm in parent_memories)
    
    # Resolve attack_type / domain from poisoned parent if applicable
    resolved_attack_type = attack_type
    if resolved_attack_type is None:
        poisoned_parents = [pm.attack_type for pm in parent_memories if pm.is_poisoned and pm.attack_type != "none"]
        resolved_attack_type = poisoned_parents[0] if poisoned_parents else "none"

    resolved_domain = domain or parent_memories[0].domain
    resolved_diff = difficulty or parent_memories[0].difficulty
    ts = synthetic_timestamp or datetime.now(timezone.utc).isoformat()
    
    new_taint_chain = propagate_taint(parent_memories, new_memory_id, transform_type="agent_summary")

    return MemoryEntry(
        memory_id=new_memory_id,
        content=new_content,
        source_id=",".join(pm.memory_id for pm in parent_memories),
        source_dataset="derived_multi_hop",
        source_channel=source_channel,
        trust_tier=effective_tier,
        synthetic_timestamp=ts,
        dormancy_days=dormancy_days,
        taint_chain=new_taint_chain,
        is_poisoned=inherited_poisoned,
        attack_type=resolved_attack_type,
        domain=resolved_domain,
        difficulty=resolved_diff,
        content_type=content_type
    )
