"""
LangGraph Agent State Definitions
"""

from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field
from src.memory.schema import (
    MemoryEntry,
    TrustTier,
    ToolSensitivity,
    IntentType,
    GateDecision,
)


class AgentState(BaseModel):
    """Execution state passed across the LangGraph agent graph."""
    query: str = Field(..., description="User query / prompt")
    user_intent: IntentType = Field(default=IntentType.OTHER)
    retrieved_memories: List[MemoryEntry] = Field(default_factory=list)
    effective_trust: TrustTier = Field(default=TrustTier.UNTRUSTED_EXTERNAL)
    taint_chain: List[str] = Field(default_factory=list)
    poisoning_scores: List[float] = Field(default_factory=list)
    conflicts_detected: List[str] = Field(default_factory=list)
    
    # LLM Proposal
    proposed_tool: Optional[str] = None
    proposed_tool_args: Dict[str, Any] = Field(default_factory=dict)
    tool_sensitivity: ToolSensitivity = Field(default=ToolSensitivity.LOW)
    reasoning_summary: str = Field(default="")
    memories_used_ids: List[str] = Field(default_factory=list)

    # Security Gate Output
    gate_decision: Optional[GateDecision] = None
    gate_reason: Optional[str] = None

    # Execution & Final Output
    tool_execution_result: Optional[str] = None
    final_response: str = Field(default="")
