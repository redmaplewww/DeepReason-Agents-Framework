from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reasoning_agent_template.models import EvidenceItem, stable_hash, utc_now


class EvidenceLedger:
    """Append-only JSONL ledger for citable evidence."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def record(
        self,
        *,
        source_type: str,
        uri: str,
        locator: str,
        content: str,
        summary: str,
        confidence: float,
        used_for: list[str] | None = None,
    ) -> EvidenceItem:
        content_hash = stable_hash(content)
        identity = stable_hash("|".join([source_type, uri, locator, content_hash]))
        evidence_id = f"ev_{identity[:12]}"
        for existing in self.list():
            if existing.id == evidence_id:
                return existing
        item = EvidenceItem(
            id=evidence_id,
            source_type=source_type,
            uri=uri,
            locator=locator,
            content_hash=content_hash,
            summary=summary,
            confidence=confidence,
            collected_at=utc_now(),
            used_for=used_for or [],
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        return item

    def list(self) -> list[EvidenceItem]:
        if not self.path.exists():
            return []
        items: list[EvidenceItem] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    items.append(EvidenceItem.from_dict(json.loads(line)))
        return items


class EvidenceConsolidationEngine:
    """Create reviewable evidence consolidation proposals without mutating knowledge."""

    def __init__(self, proposals_dir: Path):
        self.proposals_dir = Path(proposals_dir)

    def propose(
        self,
        *,
        query: str,
        category: str,
        risk_level: str,
        evidence: list[EvidenceItem],
    ) -> dict[str, Any]:
        if not evidence:
            raise ValueError("evidence consolidation proposals require evidence")
        seed = "|".join([query, category, risk_level, *[item.id for item in evidence]])
        proposal_id = f"evcon_{stable_hash(seed)[:12]}"
        target = f"knowledge/pending-evidence/{proposal_id}.md"
        payload = {
            "proposal_id": proposal_id,
            "status": "proposed",
            "created_at": utc_now(),
            "query": query,
            "category": category,
            "risk_level": risk_level,
            "target": target,
            "evidence_ids": [item.id for item in evidence],
            "source_uris": [item.uri for item in evidence],
            "requires_human_approval": True,
            "direct_mutation_performed": False,
            "suggested_summary": "Review these evidence items before adding them to the project knowledge base.",
        }
        self.proposals_dir.mkdir(parents=True, exist_ok=True)
        path = self.proposals_dir / f"{proposal_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload["path"] = str(path)
        return payload
