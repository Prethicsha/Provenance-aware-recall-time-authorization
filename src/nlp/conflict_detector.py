"""
Conflict Detector Module
Detects whether a recalled memory conflicts with authentic system policies,
standard business workflows, or user instructions.
"""

import re
from typing import List, Dict, Tuple, Optional
from src.memory.schema import MemoryEntry, IntentType, ContentType


class ConflictDetector:
    """Semantic and Pragmatic Conflict Analysis."""

    CONFLICT_PATTERNS = [
        # Policy inversion
        (r"\b(refund.*denied automatically|deny.*refund|never refund|no refunds under any circumstance)\b",
         "Inverts standard authorized refund processing workflow"),
        
        # Authentication / Security bypass
        (r"\b(bypass.*authentication|suppress.*alert|disable.*security|skip.*verification)\b",
         "Attempts to disable or suppress security/audit controls"),
        
        # Data exfiltration
        (r"\b(send.*external|forward.*copy to|upload.*credentials|dump.*passwords|cache.*in /tmp)\b",
         "Attempts unauthorized data exfiltration or credential disclosure"),
        
        # System override
        (r"\b(ignore.*instructions|override.*system|restore.*session permissions)\b",
         "Attempts prompt/behavioral hijacking or unauthorized privilege escalation")
    ]

    def detect_conflict(self, memory: MemoryEntry, current_intent: Optional[IntentType] = None) -> Tuple[bool, Optional[str]]:
        """
        Returns (has_conflict, explanation).
        """
        content_lower = memory.content.lower()

        # Check explicit conflict markers
        for pat, reason in self.CONFLICT_PATTERNS:
            if re.search(pat, content_lower):
                return True, reason

        # Check intent-to-memory conflict
        if current_intent == IntentType.FINANCIAL_ACTION:
            if "deny" in content_lower and ("refund" in content_lower or "payment" in content_lower):
                if memory.content_type in [ContentType.POLICY_CHANGE, ContentType.INSTRUCTION]:
                    return True, "Memory attempts to alter financial disbursement policy without administrator authority"

        return False, None


_GLOBAL_CONFLICT_DETECTOR = ConflictDetector()


def detect_policy_conflict(memory: MemoryEntry, current_intent: Optional[IntentType] = None) -> Tuple[bool, Optional[str]]:
    return _GLOBAL_CONFLICT_DETECTOR.detect_conflict(memory, current_intent)
