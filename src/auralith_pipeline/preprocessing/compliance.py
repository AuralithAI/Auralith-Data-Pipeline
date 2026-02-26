"""License detection for code data sources.

Ensures code data used for training
has compatible licenses (Apache-2.0, MIT, BSD, etc.).
"""

import logging
import re
from typing import Any

from auralith_pipeline.sources.data_sources import DataSample

logger = logging.getLogger(__name__)

# Licenses compatible with Apache-2.0 training data
PERMISSIVE_LICENSES = {
    "apache-2.0",
    "apache 2.0",
    "apache license 2.0",
    "mit",
    "mit license",
    "bsd-2-clause",
    "bsd-3-clause",
    "bsd",
    "isc",
    "isc license",
    "unlicense",
    "the unlicense",
    "cc0-1.0",
    "cc0",
    "public domain",
    "wtfpl",
    "0bsd",
    "postgresql",
    "zlib",
    "boost",
    "mpl-2.0",  # Weak copyleft, generally safe for training
}

# Licenses that require caution (copyleft)
COPYLEFT_LICENSES = {
    "gpl-2.0",
    "gpl-3.0",
    "gpl",
    "agpl-3.0",
    "agpl",
    "lgpl-2.1",
    "lgpl-3.0",
    "lgpl",
    "cc-by-sa-4.0",
    "cc-by-sa",
    "cc-by-nc-4.0",
    "cc-by-nc",
    "sspl",
    "eupl",
}

# Regex patterns for license detection in source code headers
LICENSE_PATTERNS = [
    (r"Apache\s+License,?\s+Version\s+2\.0", "apache-2.0"),
    (r"MIT\s+License", "mit"),
    (r"BSD\s+[23]-Clause", "bsd"),
    (r"GNU\s+General\s+Public\s+License\s+v([23])", "gpl-{match}"),
    (r"GNU\s+Affero\s+General\s+Public\s+License", "agpl-3.0"),
    (r"Mozilla\s+Public\s+License\s+2\.0", "mpl-2.0"),
    (r"ISC\s+License", "isc"),
    (r"The\s+Unlicense", "unlicense"),
    (r"Creative\s+Commons.*BY-SA", "cc-by-sa"),
    (r"Creative\s+Commons.*BY-NC", "cc-by-nc"),
    (r"SPDX-License-Identifier:\s*(\S+)", "spdx:{match}"),
]


class LicenseDetector:
    """Detect and filter licenses for code training data."""

    def __init__(
        self,
        allow_permissive: bool = True,
        allow_copyleft: bool = False,
        custom_allowed: set[str] | None = None,
    ):
        """Initialize license detector.

        Args:
            allow_permissive: Allow permissive licenses (MIT, Apache, BSD, etc.)
            allow_copyleft: Allow copyleft licenses (GPL, AGPL, etc.)
            custom_allowed: Additional license identifiers to allow
        """
        self.allowed: set[str] = set()
        if allow_permissive:
            self.allowed |= PERMISSIVE_LICENSES
        if allow_copyleft:
            self.allowed |= COPYLEFT_LICENSES
        if custom_allowed:
            self.allowed |= {lic.lower() for lic in custom_allowed}

        self.stats = {
            "total_checked": 0,
            "permissive": 0,
            "copyleft": 0,
            "unknown": 0,
            "blocked": 0,
        }

    def detect_license(self, text: str) -> str | None:
        """Detect license from text content (source code header, LICENSE file, etc.).

        Returns:
            Detected license identifier or None if unknown.
        """
        # Check first 2000 chars for license headers
        for pattern, license_id in LICENSE_PATTERNS:
            match = re.search(pattern, text[:2000], re.IGNORECASE)
            if match:
                if "{match}" in license_id:
                    if license_id.startswith("spdx:"):
                        return match.group(1).lower()
                    return license_id.replace("{match}", match.group(1))
                return license_id

        return None

    def detect_from_metadata(self, metadata: dict[str, Any]) -> str | None:
        """Extract license from sample metadata (e.g. GitHub API data)."""
        for key in ("license", "license_name", "repo_license", "spdx_id"):
            if key in metadata and metadata[key]:
                return str(metadata[key]).lower().strip()
        return None

    def is_allowed(self, sample: DataSample) -> bool:
        """Check if a code sample's license allows training use.

        Args:
            sample: DataSample (modality='code')

        Returns:
            True if the license is allowed for training.
        """
        self.stats["total_checked"] += 1

        # Try metadata first
        license_id = self.detect_from_metadata(sample.metadata)

        # Fall back to content detection
        if not license_id:
            license_id = self.detect_license(sample.content)

        if license_id is None:
            self.stats["unknown"] += 1
            # Conservative: block unknown licenses for code
            return False

        license_id = license_id.lower().strip()

        if license_id in PERMISSIVE_LICENSES:
            self.stats["permissive"] += 1
        elif license_id in COPYLEFT_LICENSES:
            self.stats["copyleft"] += 1
        else:
            self.stats["unknown"] += 1

        allowed = license_id in self.allowed
        if not allowed:
            self.stats["blocked"] += 1

        # Attach license info to metadata
        sample.metadata["detected_license"] = license_id
        sample.metadata["license_allowed"] = allowed

        return allowed


class AuditLogger:
    """Full audit logging for compliance and reproducibility.

    Logs every decision (accept/reject) with reason, for regulatory audits.
    """

    def __init__(self, log_path: str | None = None):
        """Initialize audit logger.

        Args:
            log_path: Path to audit log file (JSONL). If None, uses Python logger.
        """
        from pathlib import Path

        self.log_path = Path(log_path) if log_path else None
        self._entries: list[dict[str, Any]] = []

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        sample_id: str,
        action: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log an audit event.

        Args:
            sample_id: Unique sample identifier
            action: 'accept', 'reject', 'redact', 'transform'
            reason: Human-readable reason
            details: Additional context
        """
        import json
        import time

        entry = {
            "timestamp": time.time(),
            "sample_id": sample_id,
            "action": action,
            "reason": reason,
            "details": details or {},
        }
        self._entries.append(entry)

        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        else:
            logger.info(f"AUDIT: [{action}] {sample_id} — {reason}")

    def summary(self) -> dict[str, int]:
        """Summarize audit log."""
        actions: dict[str, int] = {}
        for e in self._entries:
            actions[e["action"]] = actions.get(e["action"], 0) + 1
        return actions
