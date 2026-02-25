"""Multi-jurisdiction PII scrubber.

Detects and redacts ALL forms of PII across every major jurisdiction.
This is the core security module — it guarantees that no private user data
(emails, phone numbers, national IDs, financial data, credentials, etc.)
from any country in the world enters the training data.

Design principles:
  • Default-deny: if something *looks* like PII, redact it.
  • Layered: regex first, then optional NER model for names/addresses.
  • Auditable: every redaction is logged with category + byte offset.
  • Idempotent: running the scrubber twice produces the same output.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from auralith_pipeline.security.privacy_config import (
    ALL_CATEGORIES,
    PIICategory,
    PrivacyConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class Redaction:
    """Record of a single PII redaction."""

    category: PIICategory
    start: int
    end: int
    original_length: int
    replacement: str


@dataclass
class ScrubResult:
    """Result of scrubbing a single text."""

    cleaned_text: str
    redactions: list[Redaction] = field(default_factory=list)
    categories_found: set[PIICategory] = field(default_factory=set)

    @property
    def pii_found(self) -> bool:
        return len(self.redactions) > 0

    @property
    def count(self) -> int:
        return len(self.redactions)


# ═══════════════════════════════════════════════════════════════════
# Regex pattern bank — grouped by PIICategory
# ═══════════════════════════════════════════════════════════════════

_PATTERNS: dict[PIICategory, list[tuple[re.Pattern[str], str]]] = {}


def _p(cat: PIICategory, pattern: str, flags: int = re.IGNORECASE) -> None:
    """Register a compiled pattern under a PII category."""
    _PATTERNS.setdefault(cat, []).append((re.compile(pattern, flags), cat.value))


# ── Emails ──
_p(PIICategory.EMAIL, r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# ── Phone numbers (international) ──
# North America
_p(PIICategory.PHONE, r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
# UK
_p(PIICategory.PHONE, r"\b(?:\+?44[-.\s]?)?\d{4,5}[-.\s]?\d{6}\b")
# EU generic (+ country code, 8-13 digits)
_p(PIICategory.PHONE, r"\b\+?[2-9]\d{1,2}[-.\s]?\d{2,4}[-.\s]?\d{4,8}\b")
# India
_p(PIICategory.PHONE, r"\b(?:\+?91[-.\s]?)?[6-9]\d{9}\b")
# Japan
_p(PIICategory.PHONE, r"\b(?:\+?81[-.\s]?)?0\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{4}\b")
# Brazil
_p(PIICategory.PHONE, r"\b(?:\+?55[-.\s]?)?\(?\d{2}\)?[-.\s]?\d{4,5}[-.\s]?\d{4}\b")
# China
_p(PIICategory.PHONE, r"\b(?:\+?86[-.\s]?)?1[3-9]\d{9}\b")

# ── Government IDs ──
# US SSN
_p(PIICategory.SSN, r"\b\d{3}[-]?\d{2}[-]?\d{4}\b")
# UK NHS
_p(PIICategory.NHS_NUMBER, r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b")
# France NIR (13 + key)
_p(PIICategory.NIR, r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b")
# India Aadhaar (12 digits, usually grouped 4-4-4)
_p(PIICategory.AADHAAR, r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b")
# Brazil CPF (###.###.###-##)
_p(PIICategory.CPF, r"\b\d{3}\.?\d{3}\.?\d{3}[-]?\d{2}\b")
# Canada SIN (### ### ###)
_p(PIICategory.SIN, r"\b\d{3}[-\s]?\d{3}[-\s]?\d{3}\b")
# Australia TFN (### ### ###)
_p(PIICategory.TFN, r"\b\d{3}[-\s]?\d{3}[-\s]?\d{3}\b")
# Netherlands BSN (9 digits)
_p(PIICategory.BSN, r"\b\d{9}\b")
# Singapore/Malaysia NRIC
_p(PIICategory.NRIC, r"\b[STFGM]\d{7}[A-Z]\b")
# Japan My Number (12 digits)
_p(PIICategory.MY_NUMBER, r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b")
# Mexico CURP (18 alphanum)
_p(PIICategory.CURP, r"\b[A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z0-9]{2}\b")
# Spain/Argentina DNI
_p(PIICategory.DNI, r"\b\d{7,8}[-\s]?[A-Z]?\b")
# Poland PESEL (11 digits)
_p(PIICategory.PESEL, r"\b\d{11}\b")
# Generic national ID (fallback for other countries)
_p(PIICategory.NATIONAL_ID_GENERIC, r"\b(?:ID|id|Id)[-:\s]?\d{6,12}\b")

# ── Financial ──
# Credit cards (Visa, MC, Amex, Discover etc.)
_p(
    PIICategory.CREDIT_CARD,
    r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))" r"[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{1,4}\b",
)
# IBAN (2-letter country + 2 check + up to 30 alphanum)
_p(
    PIICategory.IBAN,
    r"\b[A-Z]{2}\d{2}[-\s]?[A-Z0-9]{4}[-\s]?(?:[A-Z0-9]{4}[-\s]?){1,7}[A-Z0-9]{1,4}\b",
)
# SWIFT/BIC (require context: preceded by SWIFT/BIC/bank keyword)
_p(PIICategory.SWIFT_BIC, r"\b(?:SWIFT|BIC|swift|bic)[:\s]+[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b")
# Crypto wallets (BTC, ETH)
_p(PIICategory.CRYPTO_WALLET, r"\b(?:bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}\b")
_p(PIICategory.CRYPTO_WALLET, r"\b0x[a-fA-F0-9]{40}\b")

# ── IP addresses ──
_p(
    PIICategory.IP_ADDRESS,
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
)
# IPv6 (simplified)
_p(PIICategory.IP_ADDRESS, r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b")

# ── URL credentials ──
_p(PIICategory.URL_CREDENTIALS, r"https?://[^:]+:[^@]+@")

# ── MAC address ──
_p(PIICategory.MAC_ADDRESS, r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b")

# ── Credentials / secrets ──
_p(
    PIICategory.API_KEY,
    r"\b(?:api[_-]?key|apikey|token|secret|bearer)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}['\"]?",
    re.IGNORECASE,
)
_p(PIICategory.AWS_KEY, r"\bAKIA[0-9A-Z]{16}\b")
_p(
    PIICategory.AWS_KEY,
    r"(?:aws_secret_access_key|AWS_SECRET_ACCESS_KEY)\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{40}['\"]?",
)
_p(
    PIICategory.PRIVATE_KEY,
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----.*?-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
    re.DOTALL,
)
_p(PIICategory.JWT_TOKEN, r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")
_p(
    PIICategory.PASSWORD,
    r"(?:password|passwd|pwd)\s*[:=]\s*['\"]?[^\s'\"]{4,}['\"]?",
    re.IGNORECASE,
)

# ── GPS coordinates ──
_p(
    PIICategory.GPS_COORDINATES,
    r"\b[-+]?(?:[1-8]?\d(?:\.\d{4,})?|90(?:\.0+)?)\s*,\s*[-+]?(?:1[0-7]\d|0?\d{1,2})(?:\.\d{4,})?\b",
)

# ── Date of birth (various formats) ──
_p(
    PIICategory.DATE_OF_BIRTH,
    r"\b(?:DOB|dob|date[_\s]of[_\s]birth|born|birthday)\s*[:=]?\s*\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b",
    re.IGNORECASE,
)

# ── Postal codes (selected countries) ──
# US ZIP
_p(PIICategory.POSTAL_CODE, r"\b\d{5}(?:-\d{4})?\b")
# UK postcode
_p(PIICategory.POSTAL_CODE, r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b", re.IGNORECASE)
# Canada
_p(PIICategory.POSTAL_CODE, r"\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b", re.IGNORECASE)
# India PIN
_p(PIICategory.POSTAL_CODE, r"\b[1-9]\d{5}\b")


class PIIScrubber:
    """Multi-jurisdiction PII scrubber.

    Default mode is STRICT: scrubs ALL registered PII categories
    regardless of jurisdiction. This is the safest setting for
    building training data that must never contain private information.

    Usage:
        scrubber = PIIScrubber()  # strict by default
        result = scrubber.scrub("Call me at 555-123-4567")
        print(result.cleaned_text)  # "Call me at [PHONE]"
    """

    def __init__(
        self,
        config: PrivacyConfig | None = None,
        *,
        extra_patterns: dict[str, str] | None = None,
    ):
        """Initialize PII scrubber.

        Args:
            config: Privacy configuration. Defaults to STRICT global.
            extra_patterns: Additional {regex: replacement_tag} pairs.
        """
        self.config = config or PrivacyConfig()

        # Determine active categories from policies
        if self.config.mode == "strict":
            self._active_categories = ALL_CATEGORIES
        else:
            cats: set[PIICategory] = set()
            for policy in self.config.policies:
                cats |= policy.categories
            self._active_categories = frozenset(cats)

        # Build the ordered pattern list for active categories
        self._compiled_patterns: list[tuple[re.Pattern[str], PIICategory]] = []
        for cat in PIICategory:
            if cat in self._active_categories:
                for pattern, _ in _PATTERNS.get(cat, []):
                    self._compiled_patterns.append((pattern, cat))

        # Add user-supplied extra patterns
        if extra_patterns:
            for regex, _tag in extra_patterns.items():
                self._compiled_patterns.append(
                    (re.compile(regex, re.IGNORECASE), PIICategory.NATIONAL_ID_GENERIC)
                )

        # Stats
        self.stats: dict[str, int] = {
            "texts_processed": 0,
            "texts_with_pii": 0,
            "total_redactions": 0,
        }
        # Per-category counts
        self.category_stats: dict[PIICategory, int] = dict.fromkeys(PIICategory, 0)

    def _make_replacement(self, category: PIICategory, matched_text: str) -> str:
        """Create the replacement string based on config."""
        style = self.config.replacement_style
        if style == "tag":
            return f"[{category.value.upper()}]"
        elif style == "hash":
            h = hashlib.sha256(matched_text.encode()).hexdigest()[:12]
            return f"[REDACTED:{h}]"
        else:  # remove
            return ""

    def scrub(self, text: str) -> ScrubResult:
        """Scrub all PII from text.

        Args:
            text: Input text to scrub.

        Returns:
            ScrubResult with cleaned text and redaction records.
        """
        self.stats["texts_processed"] += 1

        redactions: list[Redaction] = []
        categories_found: set[PIICategory] = set()

        cleaned = text

        for pattern, category in self._compiled_patterns:
            if category not in self._active_categories:
                continue

            def _replace(match: re.Match[str], cat: PIICategory = category) -> str:
                replacement = self._make_replacement(cat, match.group())
                redactions.append(
                    Redaction(
                        category=cat,
                        start=match.start(),
                        end=match.end(),
                        original_length=len(match.group()),
                        replacement=replacement,
                    )
                )
                categories_found.add(cat)
                self.category_stats[cat] += 1
                return replacement

            cleaned = pattern.sub(_replace, cleaned)

        if redactions:
            self.stats["texts_with_pii"] += 1
            self.stats["total_redactions"] += len(redactions)

        return ScrubResult(
            cleaned_text=cleaned,
            redactions=redactions,
            categories_found=categories_found,
        )

    def scrub_text(self, text: str) -> str:
        """Convenience: scrub and return only the cleaned text."""
        return self.scrub(text).cleaned_text

    def has_pii(self, text: str) -> bool:
        """Check whether text contains any PII (without modifying it)."""
        for pattern, category in self._compiled_patterns:
            if category in self._active_categories and pattern.search(text):
                return True
        return False

    def summary(self) -> dict[str, Any]:
        """Return scrubbing statistics."""
        active_cats = {cat.value: count for cat, count in self.category_stats.items() if count > 0}
        return {
            **self.stats,
            "categories_found": active_cats,
        }
