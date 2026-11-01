"""
User Intent Classifier Module
Classifies natural language requests into structured IntentType categories:
- KNOWLEDGE_QUERY
- COMMUNICATION
- DATA_ACCESS
- FINANCIAL_ACTION
- ACCOUNT_CHANGE
- CONFIGURATION_CHANGE
- OTHER
"""

import re
from typing import Optional, Dict
from src.memory.schema import IntentType


class IntentClassifier:
    """Hybrid Rule and Keyword Intent Classifier."""

    INTENT_PATTERNS = {
        IntentType.FINANCIAL_ACTION: [
            r"\b(refund|payment|pay|transfer|transaction|chargeback|wire|invoice|bill|rebate|disburse)\b",
            r"\b(send money|process payment|issue refund|deny refund|cancel payment)\b"
        ],
        IntentType.ACCOUNT_CHANGE: [
            r"\b(account|password|profile|role|permission|privilege|access level|credentials|login)\b",
            r"\b(switch account|change password|reset password|update account|grant access|revoke access)\b"
        ],
        IntentType.CONFIGURATION_CHANGE: [
            r"\b(config|configure|configuration|setting|parameter|override|disable check|security policy|flag)\b",
            r"\b(change setting|modify config|update rule|bypass security)\b"
        ],
        IntentType.COMMUNICATION: [
            r"\b(email|mail|message|notify|notification|draft email|send email|contact|alert|reply)\b"
        ],
        IntentType.DATA_ACCESS: [
            r"\b(export|dump|extract|download|read file|fetch data|database|exfiltrate|get records)\b"
        ],
        IntentType.KNOWLEDGE_QUERY: [
            r"\b(what is|who is|where is|how does|why is|explain|tell me about|information on|definition of)\b",
            r"\b(search for|lookup|look up|read knowledge|query)\b"
        ]
    }

    def classify(self, text: str) -> IntentType:
        if not text:
            return IntentType.OTHER
        
        text_lower = text.lower().strip()
        
        # Priority order for security sensitivity
        priority_order = [
            IntentType.FINANCIAL_ACTION,
            IntentType.CONFIGURATION_CHANGE,
            IntentType.ACCOUNT_CHANGE,
            IntentType.DATA_ACCESS,
            IntentType.COMMUNICATION,
            IntentType.KNOWLEDGE_QUERY,
        ]

        for intent in priority_order:
            patterns = self.INTENT_PATTERNS.get(intent, [])
            for pat in patterns:
                if re.search(pat, text_lower):
                    return intent

        return IntentType.OTHER


_GLOBAL_INTENT_CLASSIFIER = IntentClassifier()


def classify_user_intent(query: str) -> IntentType:
    return _GLOBAL_INTENT_CLASSIFIER.classify(query)
