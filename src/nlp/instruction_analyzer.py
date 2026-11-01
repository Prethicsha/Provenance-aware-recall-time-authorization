"""
Instruction and Fact Analyzer Module
Analyzes whether a natural language memory is declarative (fact/description)
or directive/normative (instruction, policy change, action request, behavior modification).
"""

import re
from typing import Dict, Any
from src.memory.schema import ContentType


class InstructionAnalyzer:
    """Linguistic and Pragmatic Content Classifier."""

    INSTRUCTION_MARKERS = [
        r"\b(must|should|always|never|do not|don't|ensure|require|override|execute|run|perform|send|deny|grant)\b",
        r"^(please\s+|make sure to\s+|you must\s+|always\s+|never\s+)",
        r"\b(ignore previous|disregard|system prompt|new rule|instructions:)\b"
    ]

    POLICY_CHANGE_MARKERS = [
        r"\b(policy has changed|policy update|new policy|updated policy|effective immediately|from now on|as per updated rule)\b",
        r"\b(all refund requests should now be|all requests must be denied|new authorization required)\b"
    ]

    BEHAVIOR_CHANGE_MARKERS = [
        r"\b(whenever the user asks|if the user requests|when a clinical alert|when session restore is present|backdoor)\b",
        r"\b(trigger|suppress|bypass|substitute|route all|redirect)\b"
    ]

    def analyze(self, text: str) -> ContentType:
        if not text:
            return ContentType.FACT

        text_lower = text.lower().strip()

        # Check for Policy Changes
        for pat in self.POLICY_CHANGE_MARKERS:
            if re.search(pat, text_lower):
                return ContentType.POLICY_CHANGE

        # Check for Potential Behavior Changes / Triggers
        for pat in self.BEHAVIOR_CHANGE_MARKERS:
            if re.search(pat, text_lower):
                return ContentType.POTENTIAL_BEHAVIOR_CHANGE

        # Check for Directives / Instructions
        for pat in self.INSTRUCTION_MARKERS:
            if re.search(pat, text_lower):
                return ContentType.INSTRUCTION

        # Default declarative content
        return ContentType.FACT


_GLOBAL_INSTRUCTION_ANALYZER = InstructionAnalyzer()


def analyze_content_type(text: str) -> ContentType:
    return _GLOBAL_INSTRUCTION_ANALYZER.analyze(text)
