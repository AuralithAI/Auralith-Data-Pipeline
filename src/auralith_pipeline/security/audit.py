"""Privacy-specific audit trail for regulatory compliance.

Logs every PII redaction, data sanitization decision, and
data-subject-related action for GDPR Article 30, CCPA § 1798.100,
and equivalent requirements worldwide.

Output: JSONL file (append-only) + optional structured summary.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PrivacyAuditEntry:
    """A single privacy audit event."""

    timestamp: float
    event_type: str           # 'pii_redaction', 'secret_redaction', 'sample_blocked', etc.
    sample_id: str
    source: str
    categories: list[str]     # PII categories found
    redaction_count: int
    action: str               # 'redacted', 'blocked', 'passed'
    details: dict[str, Any] = field(default_factory=dict)


class PrivacyAuditLogger:
    """Append-only audit logger for privacy compliance.

    Every PII detection, secret redaction, or data-blocking decision
    is recorded here. This log can be presented to auditors, DPOs,
    and regulators to demonstrate compliance.

    Usage:
        audit = PrivacyAuditLogger(log_path="./audit/privacy.jsonl")
        audit.log_redaction(
            sample_id="wiki_12345",
            source="wikipedia",
            categories=["email", "phone"],
            redaction_count=3,
        )
        audit.log_blocked(sample_id="pile_999", source="the_pile", reason="pii_fail_rescan")
        print(audit.summary())
    """

    def __init__(
        self,
        log_path: str | None = None,
        *,
        enabled: bool = True,
        buffer_size: int = 100,
    ):
        """Initialize privacy audit logger.

        Args:
            log_path: Path to JSONL audit log. If None, uses Python logger only.
            enabled: Master switch.
            buffer_size: Flush to disk every N entries.
        """
        self.enabled = enabled
        self.log_path = Path(log_path) if log_path else None
        self.buffer_size = buffer_size

        self._entries: list[PrivacyAuditEntry] = []
        self._buffer: list[str] = []
        self._counts: dict[str, int] = {
            "total_events": 0,
            "redactions": 0,
            "blocked": 0,
            "passed": 0,
        }

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_entry(self, entry: PrivacyAuditEntry) -> None:
        """Serialize and buffer an entry."""
        line = json.dumps({
            "timestamp": entry.timestamp,
            "event_type": entry.event_type,
            "sample_id": entry.sample_id,
            "source": entry.source,
            "categories": entry.categories,
            "redaction_count": entry.redaction_count,
            "action": entry.action,
            "details": entry.details,
        })

        self._buffer.append(line)

        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Flush buffered entries to disk."""
        if not self._buffer or not self.log_path:
            return
        with open(self.log_path, "a", encoding="utf-8") as f:
            for line in self._buffer:
                f.write(line + "\n")
        self._buffer.clear()

    def log_redaction(
        self,
        sample_id: str,
        source: str,
        categories: list[str],
        redaction_count: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log a PII/secret redaction event."""
        if not self.enabled:
            return

        entry = PrivacyAuditEntry(
            timestamp=time.time(),
            event_type="pii_redaction",
            sample_id=sample_id,
            source=source,
            categories=categories,
            redaction_count=redaction_count,
            action="redacted",
            details=details or {},
        )
        self._entries.append(entry)
        self._counts["total_events"] += 1
        self._counts["redactions"] += 1
        self._write_entry(entry)

    def log_blocked(
        self,
        sample_id: str,
        source: str,
        reason: str,
        categories: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log a sample that was completely blocked (PII could not be safely removed)."""
        if not self.enabled:
            return

        entry = PrivacyAuditEntry(
            timestamp=time.time(),
            event_type="sample_blocked",
            sample_id=sample_id,
            source=source,
            categories=categories or [],
            redaction_count=0,
            action="blocked",
            details={"reason": reason, **(details or {})},
        )
        self._entries.append(entry)
        self._counts["total_events"] += 1
        self._counts["blocked"] += 1
        self._write_entry(entry)

    def log_passed(
        self,
        sample_id: str,
        source: str,
    ) -> None:
        """Log a sample that passed privacy checks cleanly (no PII found)."""
        if not self.enabled:
            return

        entry = PrivacyAuditEntry(
            timestamp=time.time(),
            event_type="privacy_check_passed",
            sample_id=sample_id,
            source=source,
            categories=[],
            redaction_count=0,
            action="passed",
        )
        self._entries.append(entry)
        self._counts["total_events"] += 1
        self._counts["passed"] += 1
        # Only write to disk if we have a log path (avoid noise for clean samples)
        if self.log_path:
            self._write_entry(entry)

    def summary(self) -> dict[str, Any]:
        """Return aggregate statistics."""
        # Category frequency
        cat_freq: dict[str, int] = {}
        for entry in self._entries:
            for cat in entry.categories:
                cat_freq[cat] = cat_freq.get(cat, 0) + 1

        return {
            **self._counts,
            "category_frequency": cat_freq,
        }

    def close(self) -> None:
        """Flush remaining buffer and close."""
        self.flush()

    def __del__(self) -> None:
        try:
            self.flush()
        except Exception:
            pass
