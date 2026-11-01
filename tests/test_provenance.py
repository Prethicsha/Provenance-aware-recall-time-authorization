"""
Unit Tests for Provenance and Trust Hierarchy
"""

import pytest
from src.memory.schema import TrustTier, ContentType, MemoryEntry
from src.memory.provenance import map_channel_to_trust_tier, create_memory


def test_channel_to_trust_tier_mapping():
    assert map_channel_to_trust_tier("system_security_kernel") == TrustTier.TRUSTED_SYSTEM_ROOT
    assert map_channel_to_trust_tier("admin_policy_console") == TrustTier.SUPERVISOR_ADMIN
    assert map_channel_to_trust_tier("user_prompt_session") == TrustTier.AUTHENTICATED_USER
    assert map_channel_to_trust_tier("external_web_crawler") == TrustTier.UNTRUSTED_EXTERNAL
    assert map_channel_to_trust_tier("untrusted_incoming_email") == TrustTier.UNTRUSTED_EXTERNAL


def test_create_memory_orthogonal_poisoning():
    # Guideline 2: is_poisoned and trust_tier are independent
    mem1 = create_memory(
        memory_id="M001",
        content="Benign text from untrusted web.",
        source_id="S01",
        source_dataset="web",
        source_channel="external_web",
        is_poisoned=False,
        trust_tier=TrustTier.UNTRUSTED_EXTERNAL
    )
    assert mem1.trust_tier == TrustTier.UNTRUSTED_EXTERNAL
    assert mem1.is_poisoned is False
    assert len(mem1.taint_chain) == 1

    mem2 = create_memory(
        memory_id="M002",
        content="Poisoned text from supervisor console.",
        source_id="S02",
        source_dataset="admin",
        source_channel="admin_console",
        is_poisoned=True,
        trust_tier=TrustTier.SUPERVISOR_ADMIN
    )
    assert mem2.trust_tier == TrustTier.SUPERVISOR_ADMIN
    assert mem2.is_poisoned is True
