"""
Unit Tests for Taint Propagation and Multi-Hop Derivation
"""

import pytest
from src.memory.schema import TrustTier, ContentType
from src.memory.provenance import create_memory
from src.memory.taint import get_effective_trust, derive_memory, propagate_taint


def test_effective_trust_calculation():
    m_untrusted = create_memory("M1", "Text 1", "S1", "web", "external_web", is_poisoned=True, trust_tier=TrustTier.UNTRUSTED_EXTERNAL)
    m_user = create_memory("M2", "Text 2", "S2", "chat", "user_prompt", is_poisoned=False, trust_tier=TrustTier.AUTHENTICATED_USER)
    m_admin = create_memory("M3", "Text 3", "S3", "admin", "admin_console", is_poisoned=False, trust_tier=TrustTier.SUPERVISOR_ADMIN)

    # Minimum trust among {0, 1} is 0
    assert get_effective_trust([m_untrusted, m_user]) == TrustTier.UNTRUSTED_EXTERNAL
    # Minimum trust among {1, 2} is 1
    assert get_effective_trust([m_user, m_admin]) == TrustTier.AUTHENTICATED_USER
    # Empty list fail-safes to Tier 0
    assert get_effective_trust([]) == TrustTier.UNTRUSTED_EXTERNAL


def test_multi_hop_taint_preservation():
    # Hop 1: Untrusted external memory
    m1 = create_memory("M1", "Poisoned policy update", "SRC1", "web", "external_web", is_poisoned=True, trust_tier=TrustTier.UNTRUSTED_EXTERNAL)
    
    # Hop 2: Agent summarizes M1 into M2
    m2 = derive_memory(
        parent_memories=[m1],
        new_content="Summary: policy updated to deny all refund requests.",
        new_memory_id="M2"
    )

    # Verify M2 inherits taint and untrusted tier
    assert m2.trust_tier == TrustTier.UNTRUSTED_EXTERNAL
    assert m2.is_poisoned is True
    assert len(m2.taint_chain) == 2
    assert "external_web:SRC1" in m2.taint_chain[0]
    assert "agent_summary:M2" in m2.taint_chain[1]
