"""
Dataset Loaders Module for Memory Poisoning & Agent Security Benchmarks
Loads and converts MemShield-Bench, BIPIA, Bitext, and Enron into standardized MemoryEntry objects.
Preserves independence between ground-truth is_poisoned and origin trust_tier.
"""

import os
import json
import zipfile
import pandas as pd
from typing import List, Optional
from datetime import datetime, timezone

from src.memory.schema import MemoryEntry, TrustTier, ContentType
from src.memory.provenance import create_memory
from src.data.preprocessing import clean_text, is_valid_text


def load_memshield_dataset(
    split: str = "full",
    base_dir: str = "data/raw/memshield"
) -> List[MemoryEntry]:
    """
    Loads MemShield-Bench (Primary Benchmark).
    Splits available: 'full', 'train', 'validation', 'test'.
    """
    filename = f"{split}.jsonl" if not split.endswith(".jsonl") else split
    filepath = os.path.join(base_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"MemShield dataset split not found at: {filepath}")

    memories: List[MemoryEntry] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            raw_content = item.get("content", "")
            if not is_valid_text(raw_content):
                continue

            cleaned_content = clean_text(raw_content)
            is_poisoned = (item.get("label", "").lower() == "poisoned")
            attack_type = item.get("attack_type", "none")
            domain = item.get("domain", "general")
            difficulty = item.get("difficulty", "easy")
            mem_id = item.get("id", f"MEM-SHIELD-{split.upper()}-{idx:05d}")

            # Assign origin source channel and explicit trust tier independently of is_poisoned
            # Provenance represents channel trust origin; is_poisoned represents adversarial content.
            source_raw = str(item.get("source", "")).lower()
            source_dataset = "memshield"
            
            if any(k in source_raw for k in ["agentpoison", "web", "external", "injection"]):
                source_channel = "external_web_source"
                trust_tier = TrustTier.UNTRUSTED_EXTERNAL
            elif any(k in source_raw for k in ["bulk_t", "realistic_t"]):
                # Adversarial dataset submissions tagged as external unverified submissions
                source_channel = "external_unverified_submission"
                trust_tier = TrustTier.UNTRUSTED_EXTERNAL
            elif any(k in source_raw for k in ["domain", "clean", "bulk_clean"]):
                source_channel = "corporate_knowledge_base"
                trust_tier = TrustTier.TRUSTED_SYSTEM_ROOT
            else:
                # Default external untrusted channel
                source_channel = "external_document_feed"
                trust_tier = TrustTier.UNTRUSTED_EXTERNAL

            # Determine content type based on linguistic cues
            if any(k in attack_type for k in ["instruction", "override"]):
                ctype = ContentType.INSTRUCTION
            elif any(k in attack_type for k in ["policy", "configuration"]):
                ctype = ContentType.POLICY_CHANGE
            elif "backdoor" in attack_type or "behavior" in attack_type:
                ctype = ContentType.POTENTIAL_BEHAVIOR_CHANGE
            else:
                ctype = ContentType.FACT

            mem = create_memory(
                memory_id=mem_id,
                content=cleaned_content,
                source_id=item.get("id", f"raw-{idx}"),
                source_dataset=source_dataset,
                source_channel=source_channel,
                is_poisoned=is_poisoned,
                trust_tier=trust_tier,
                attack_type=attack_type,
                domain=domain,
                difficulty=difficulty,
                content_type=ctype,
                dormancy_days=0,
                synthetic_timestamp=datetime.now(timezone.utc).isoformat()
            )
            memories.append(mem)

    return memories


def load_bipia_dataset(
    filepath: str = "data/raw/dataset_for_huggingface.jsonl",
    max_samples: int = 2000
) -> List[MemoryEntry]:
    """
    Loads BIPIA + GPT-4o-mini prompt injection benchmark for indirect injection evaluation.
    NOTE: Benign and indirect injection examples are synthetic/generated benchmark datasets.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"BIPIA dataset not found at: {filepath}")

    memories: List[MemoryEntry] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if len(memories) >= max_samples:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            content = clean_text(item.get("context", ""))
            if not is_valid_text(content):
                continue

            is_poisoned = (int(item.get("label", 0)) == 1)
            source = item.get("source", "BIPIA")
            mem_id = f"MEM-BIPIA-{idx:05d}"
            
            # Channel is external web/code repository (Tier 0) regardless of poisoning status
            source_channel = "external_web_repo"
            trust_tier = TrustTier.UNTRUSTED_EXTERNAL

            mem = create_memory(
                memory_id=mem_id,
                content=content,
                source_id=f"bipia-{idx}",
                source_dataset="bipia_synthetic",
                source_channel=source_channel,
                is_poisoned=is_poisoned,
                trust_tier=trust_tier,
                attack_type="T1_instruction_override" if is_poisoned else "none",
                domain="software_development",
                difficulty="medium" if is_poisoned else "easy",
                content_type=ContentType.INSTRUCTION if is_poisoned else ContentType.FACT
            )
            memories.append(mem)

    return memories


def load_bitext_dataset(
    filepath: Optional[str] = None,
    max_samples: int = 2000
) -> List[MemoryEntry]:
    """
    Loads Bitext Customer Support Dataset for benign corporate interactions, intent analysis, and false positive evaluation.
    Supports reading from both CSV and XLSX formats with pandas/openpyxl.
    """
    if filepath is None:
        # Check for CSV or XLSX in data/raw
        possible_paths = [
            "data/raw/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv",
            "data/raw/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.xlsx",
            "data/raw/bitext.csv",
            "data/raw/bitext.xlsx"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                filepath = p
                break

    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"Bitext dataset file not found at: {filepath}")

    if filepath.endswith(".xlsx") or filepath.endswith(".xls"):
        df = pd.read_excel(filepath, engine="openpyxl")
    else:
        try:
            df = pd.read_csv(filepath)
        except Exception:
            df = pd.read_excel(filepath, engine="openpyxl")

    df = df.dropna(subset=["response", "instruction"]).head(max_samples)

    memories: List[MemoryEntry] = []
    for idx, row in df.iterrows():
        # Store customer support knowledge / past resolution memory
        content = clean_text(f"Customer Support Resolution for [{row['category']} - {row['intent']}]: {row['response']}")
        if not is_valid_text(content):
            continue

        mem_id = f"MEM-BITEXT-{idx:05d}"
        mem = create_memory(
            memory_id=mem_id,
            content=content,
            source_id=f"bitext-{idx}",
            source_dataset="bitext",
            source_channel="customer_support_kb",
            is_poisoned=False,  # Bitext is authentic benign support data
            trust_tier=TrustTier.SUPERVISOR_ADMIN,  # Verified corporate KB
            attack_type="none",
            domain="customer_service",
            difficulty="easy",
            content_type=ContentType.FACT
        )
        memories.append(mem)

    return memories


def load_enron_sample(
    zip_path: str = "data/raw/enron email.zip",
    max_samples: int = 500
) -> List[MemoryEntry]:
    """
    Loads realistic corporate emails from Enron dataset via ZIP streaming without extracting full archive.
    NOTE: Provenance channels ('internal_email_archive') and trust tiers ('AUTHENTICATED_USER') are
    synthetic experimental metadata configured for evaluation, not native dataset labels.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Enron ZIP dataset not found at: {zip_path}")

    memories: List[MemoryEntry] = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for fname in z.namelist():
            if fname.endswith("emails.csv"):
                with z.open(fname) as ef:
                    # Stream chunks from CSV inside zip
                    chunk_iter = pd.read_csv(ef, chunksize=100)
                    for chunk in chunk_iter:
                        for idx, row in chunk.iterrows():
                            if len(memories) >= max_samples:
                                return memories
                            raw_msg = str(row.get("message", ""))
                            # Extract body (strip excessive MIME headers)
                            parts = raw_msg.split("\n\n", 1)
                            body = parts[1] if len(parts) > 1 else raw_msg
                            cleaned = clean_text(body[:600])
                            if not is_valid_text(cleaned, min_length=20):
                                continue

                            mem_id = f"MEM-ENRON-{len(memories):05d}"
                            mem = create_memory(
                                memory_id=mem_id,
                                content=cleaned,
                                source_id=str(row.get("file", f"enron-{len(memories)}")),
                                source_dataset="enron",
                                source_channel="internal_email_archive",
                                is_poisoned=False,  # Authentic benign corporate email
                                trust_tier=TrustTier.AUTHENTICATED_USER,  # Synthetic experimental metadata
                                attack_type="none",
                                domain="operations",
                                difficulty="easy",
                                content_type=ContentType.FACT
                            )
                            memories.append(mem)
                break

    return memories
