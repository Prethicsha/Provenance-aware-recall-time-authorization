"""
Memory, Trust, and Security Schema Definitions
Strict Pydantic models for persistent memory, provenance metadata, taint tracking, and security gate audits.
"""

from enum import Enum, IntEnum
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator


class TrustTier(IntEnum):
    """
    Explicit Trust Hierarchy:
    0 = UNTRUSTED_EXTERNAL (Web, incoming untrusted email, 3rd party tool outputs)
    1 = AUTHENTICATED_USER (Direct user prompt input, session context)
    2 = SUPERVISOR_ADMIN (Organizational supervisor, security policy administrator)
    3 = TRUSTED_SYSTEM_ROOT (Hardcoded baseline system instructions, core knowledge base)
    """
    UNTRUSTED_EXTERNAL = 0
    AUTHENTICATED_USER = 1
    SUPERVISOR_ADMIN = 2
    TRUSTED_SYSTEM_ROOT = 3


class ToolSensitivity(IntEnum):
    """
    Sensitivity tiers for agent tool actions:
    LOW = 1: Read-only operations, public knowledge queries, internal scratchpad
    MEDIUM = 2: Draft communication, local caching, non-destructive state changes
    HIGH = 3: Financial transactions (payments/refunds), account modifications, data exfiltration, system config
    """
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class ContentType(str, Enum):
    FACT = "FACT"
    DESCRIPTION = "DESCRIPTION"
    INSTRUCTION = "INSTRUCTION"
    POLICY_CHANGE = "POLICY_CHANGE"
    RECOMMENDATION = "RECOMMENDATION"
    ACTION_REQUEST = "ACTION_REQUEST"
    POTENTIAL_BEHAVIOR_CHANGE = "POTENTIAL_BEHAVIOR_CHANGE"


class IntentType(str, Enum):
    KNOWLEDGE_QUERY = "KNOWLEDGE_QUERY"
    COMMUNICATION = "COMMUNICATION"
    DATA_ACCESS = "DATA_ACCESS"
    FINANCIAL_ACTION = "FINANCIAL_ACTION"
    ACCOUNT_CHANGE = "ACCOUNT_CHANGE"
    CONFIGURATION_CHANGE = "CONFIGURATION_CHANGE"
    OTHER = "OTHER"


class GateDecision(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


class TaintChainEntry(BaseModel):
    hop_index: int
    entity_id: str
    entity_type: str  # 'external_source', 'raw_memory', 'agent_summary', 'derived_memory'
    trust_tier: TrustTier
    timestamp: str


class MemoryEntry(BaseModel):
    """
    Normalized Memory Entry stored in persistent vector memory.
    IMPORTANT: is_poisoned (behavioral attack label) and trust_tier (origin channel)
    are strictly orthogonal and independently maintained.
    """
    memory_id: str = Field(..., description="Unique memory identifier, e.g., MEM-00123")
    content: str = Field(..., description="Natural language memory text")
    source_id: str = Field(..., description="ID from raw dataset or originating process")
    source_dataset: str = Field(..., description="Name of source dataset, e.g., memshield, bipia, bitext, enron, synthetic")
    source_channel: str = Field(..., description="Channel origin: external_web, email, customer_chat, admin_console, system")
    trust_tier: TrustTier = Field(..., description="Explicit source trust tier (0 to 3)")
    synthetic_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp for synthetic temporal simulations"
    )
    dormancy_days: int = Field(default=0, description="Synthetic simulated dormancy in days (0, 7, 30)")
    taint_chain: List[str] = Field(
        default_factory=list,
        description="Audit trace of provenance origins and derivation hops"
    )
    is_poisoned: bool = Field(..., description="Ground truth behavioral poisoning label")
    attack_type: str = Field(default="none", description="Attack type (e.g. T1_instruction_override, none)")
    domain: str = Field(default="general", description="Application domain (software_development, healthcare, customer_service, etc.)")
    difficulty: str = Field(default="easy", description="Attack difficulty: easy, medium, hard")
    content_type: ContentType = Field(default=ContentType.FACT, description="Linguistic/pragmatic content type")

    # Optional / Retrieval & Runtime Fields
    user_intent: Optional[IntentType] = None
    retrieval_rank: Optional[int] = None
    similarity_score: Optional[float] = None
    poisoning_score: Optional[float] = None
    conflict_detected: Optional[bool] = None
    proposed_tool: Optional[str] = None
    tool_sensitivity: Optional[ToolSensitivity] = None
    expected_decision: Optional[GateDecision] = None
    gate_decision: Optional[GateDecision] = None
    gate_reason: Optional[str] = None

    @property
    def is_tainted(self) -> bool:
        """A memory is tainted if its origin trust tier is UNTRUSTED_EXTERNAL (0) or if any derivation inherited tier 0."""
        return (self.trust_tier == TrustTier.UNTRUSTED_EXTERNAL) or any("UNTRUSTED" in h or "inherited_tier=0" in h for h in self.taint_chain)

    @field_validator("content")
    def content_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Memory content must not be empty.")
        return v.strip()


class AuditRecord(BaseModel):
    """
    Forensic security audit record generated for every recall-time gate evaluation.
    """
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    scenario_id: str
    user_query: str
    user_intent: IntentType
    retrieved_memory_ids: List[str]
    effective_trust: TrustTier
    proposed_tool: str
    tool_sensitivity: ToolSensitivity
    gate_decision: GateDecision
    reason: str
    taint_chain: List[str]
    is_adversarial_scenario: bool
    ground_truth_poisoned: bool
    semantic_detector_flagged: bool
    policy_violated: bool
