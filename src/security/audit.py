"""
Security Audit Logging Module
Logs every recall-time authorization decision to structured JSONL for auditability and forensic compliance.
"""

import os
import json
from typing import Optional
from src.memory.schema import AuditRecord


class SecurityAuditLogger:
    """Forensic JSONL Audit Logger."""
    def __init__(self, log_filepath: str = "logs/security_audit.jsonl"):
        self.log_filepath = log_filepath
        os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

    def log_decision(self, record: AuditRecord):
        """Appends audit record to JSONL."""
        with open(self.log_filepath, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

    def read_records(self, limit: Optional[int] = None) -> list:
        """Reads recent audit records."""
        if not os.path.exists(self.log_filepath):
            return []
        records = []
        with open(self.log_filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        if limit:
            return records[-limit:]
        return records


_GLOBAL_AUDIT_LOGGER = None


def get_audit_logger(log_filepath: str = "logs/security_audit.jsonl") -> SecurityAuditLogger:
    global _GLOBAL_AUDIT_LOGGER
    if _GLOBAL_AUDIT_LOGGER is None:
        _GLOBAL_AUDIT_LOGGER = SecurityAuditLogger(log_filepath)
    return _GLOBAL_AUDIT_LOGGER
