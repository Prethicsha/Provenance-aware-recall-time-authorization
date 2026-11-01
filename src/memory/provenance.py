"""
Provenance Tracking and Metadata Assignment Module
Provides deterministic provenance tracking and origin channel mapping.
"""

from typing import List, Optional
from datetime import datetime, timezone
from .schema import MemoryEntry, TrustTier, ContentType


def map_channel_to_trust_tier(source_channel: str, default_tier: Optional[TrustTier] = None) -> TrustTier:
    """
    Deterministically maps input channel sources to explicit trust tiers.
    """
    if default_tier is not None:
        return default_tier

    channel_lower = (source_channel or "").lower().strip()
    
    # Tier 3: System Root
    if any(k in channel_lower for k in ["system", "root", "hardcoded_policy", "security_kernel"]):
        return TrustTier.TRUSTED_SYSTEM_ROOT

    # Tier 2: Supervisor / Admin
    if any(k in channel_lower for k in ["admin", "supervisor", "policy_update", "authorized_staff"]):
        return TrustTier.SUPERVISOR_ADMIN

    # Tier 1: Authenticated User
    if any(k in channel_lower for k in ["user_prompt", "authenticated_user", "customer_session"]):
        return TrustTier.AUTHENTICATED_USER

    # Tier 0: Untrusted External
    # Defaults to UNTRUSTED_EXTERNAL for web, email, external chat, 3rd party tools
    return TrustTier.UNTRUSTED_EXTERNAL


def create_memory(
    memory_id: str,
    content: str,
    source_id: str,
    source_dataset: str,
    source_channel: str,
    is_poisoned: bool,
    trust_tier: Optional[TrustTier] = None,
    attack_type: str = "none",
    domain: str = "general",
    difficulty: str = "easy",
    content_type: ContentType = ContentType.FACT,
    dormancy_days: int = 0,
    synthetic_timestamp: Optional[str] = None
) -> MemoryEntry:
    """
    Factory function to create a validated MemoryEntry with initial provenance and taint chain.
    """
    effective_tier = trust_tier if trust_tier is not None else map_channel_to_trust_tier(source_channel)
    ts = synthetic_timestamp or datetime.now(timezone.utc).isoformat()
    
    initial_taint = [f"{source_channel}:{source_id} [tier={int(effective_tier)}]"]

    return MemoryEntry(
        memory_id=memory_id,
        content=content,
        source_id=source_id,
        source_dataset=source_dataset,
        source_channel=source_channel,
        trust_tier=effective_tier,
        synthetic_timestamp=ts,
        dormancy_days=dormancy_days,
        taint_chain=initial_taint,
        is_poisoned=is_poisoned,
        attack_type=attack_type,
        domain=domain,
        difficulty=difficulty,
        content_type=content_type
    )
