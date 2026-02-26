"""Data sanitizer — removes credentials, secrets, and dangerous content.

Complements PIIScrubber by focusing on:
  • API keys, tokens, passwords embedded in text
  • Private keys, certificates
  • Database connection strings
  • AWS / GCP / Azure credentials
  • Internal URLs, hostnames, IP ranges
  • Potentially dangerous content (shell commands with creds, etc.)

This is the second layer of defense after PIIScrubber.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SanitizationResult:
    """Result of sanitizing a text."""

    cleaned_text: str
    issues_found: list[dict[str, str]] = field(default_factory=list)

    @property
    def had_issues(self) -> bool:
        return len(self.issues_found) > 0


# ── Pattern definitions ──

_SANITIZE_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    # AWS access keys
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[AWS_KEY]"),
    # AWS secret keys
    (
        "aws_secret_key",
        re.compile(
            r"(?:aws_secret_access_key|AWS_SECRET_ACCESS_KEY|aws_secret)\s*[:=]\s*"
            r"['\"]?[A-Za-z0-9/+=]{40}['\"]?",
            re.IGNORECASE,
        ),
        "[AWS_SECRET]",
    ),
    # GCP service account keys (JSON key file patterns)
    (
        "gcp_service_account",
        re.compile(
            r'"private_key"\s*:\s*"-----BEGIN (?:RSA )?PRIVATE KEY-----[^"]*-----END (?:RSA )?PRIVATE KEY-----"',
            re.DOTALL,
        ),
        '"private_key": "[GCP_KEY_REDACTED]"',
    ),
    # Azure connection strings
    (
        "azure_connection_string",
        re.compile(
            r"(?:DefaultEndpointsProtocol|AccountKey|SharedAccessSignature)"
            r"\s*=\s*[A-Za-z0-9+/=]{20,}",
            re.IGNORECASE,
        ),
        "[AZURE_CREDENTIAL]",
    ),
    # Generic API keys / tokens
    (
        "generic_api_key",
        re.compile(
            r"(?:api[_-]?key|api[_-]?token|auth[_-]?token|bearer|secret[_-]?key|access[_-]?token)"
            r"\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{20,}['\"]?",
            re.IGNORECASE,
        ),
        "[API_KEY]",
    ),
    # Bearer tokens in headers
    (
        "bearer_token",
        re.compile(r"Authorization\s*:\s*Bearer\s+[A-Za-z0-9_\-.]+", re.IGNORECASE),
        "Authorization: Bearer [TOKEN]",
    ),
    # Private keys (PEM format)
    (
        "private_key",
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"
            r".*?"
            r"-----END (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[PRIVATE_KEY_REDACTED]",
    ),
    # Certificates (just detect, don't remove — they're public)
    # But PEM private keys embedded alongside should be caught above
    # JWT tokens
    (
        "jwt_token",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
        "[JWT_TOKEN]",
    ),
    # Database connection strings
    (
        "db_connection_string",
        re.compile(
            r"(?:mongodb|mysql|postgres|postgresql|redis|amqp|mssql)" r"(?:\+\w+)?://[^\s'\"]+",
            re.IGNORECASE,
        ),
        "[DB_CONNECTION_STRING]",
    ),
    # URL with embedded credentials
    ("url_credentials", re.compile(r"https?://[^:]+:[^@]+@[^\s]+"), "[URL_WITH_CREDENTIALS]"),
    # Slack webhooks
    (
        "slack_webhook",
        re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/]+"),
        "[SLACK_WEBHOOK]",
    ),
    # GitHub tokens
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b"), "[GITHUB_TOKEN]"),
    # Passwords in common config formats
    (
        "config_password",
        re.compile(
            r"(?:password|passwd|pwd|pass)\s*[:=]\s*['\"]?[^\s'\"]{4,}['\"]?",
            re.IGNORECASE,
        ),
        "[PASSWORD]",
    ),
    # .env file lines with secrets
    (
        "env_secret",
        re.compile(
            r"^(?:SECRET|TOKEN|KEY|PASSWORD|CREDENTIAL|API_KEY|AUTH)\w*\s*=\s*.+$",
            re.MULTILINE | re.IGNORECASE,
        ),
        "[ENV_SECRET]",
    ),
    # SSH keys (public — less critical, but good practice)
    (
        "ssh_public_key",
        re.compile(r"ssh-(?:rsa|dss|ed25519|ecdsa)\s+[A-Za-z0-9+/=]{100,}"),
        "[SSH_PUBLIC_KEY]",
    ),
]


class DataSanitizer:
    """Sanitize text by removing embedded credentials, secrets, and
    potentially dangerous content.

    This is a defense-in-depth layer on top of PIIScrubber.
    PIIScrubber handles personal data; DataSanitizer handles
    infrastructure secrets and credentials.

    Usage:
        sanitizer = DataSanitizer()
        result = sanitizer.sanitize("Connect with AKIA1234567890ABCDEF")
        print(result.cleaned_text)
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        extra_patterns: list[tuple[str, str, str]] | None = None,
        block_internal_urls: bool = True,
        internal_domains: list[str] | None = None,
    ):
        """Initialize data sanitizer.

        Args:
            enabled: Master switch.
            extra_patterns: Additional [(name, regex, replacement), ...].
            block_internal_urls: Redact URLs pointing to internal/private domains.
            internal_domains: List of domain suffixes to treat as internal
                              (e.g. ['.corp.example.com', '.internal']).
        """
        self.enabled = enabled
        self.block_internal_urls = block_internal_urls
        self.internal_domains = internal_domains or [
            ".internal",
            ".local",
            ".corp",
            ".intranet",
            ".private",
        ]

        # Build full pattern list
        self._patterns = list(_SANITIZE_PATTERNS)
        if extra_patterns:
            for name, regex, replacement in extra_patterns:
                self._patterns.append((name, re.compile(regex, re.IGNORECASE), replacement))

        # Stats
        self.stats: dict[str, int] = {
            "texts_processed": 0,
            "texts_with_secrets": 0,
            "total_redactions": 0,
        }
        self.pattern_stats: dict[str, int] = {}

    def sanitize(self, text: str) -> SanitizationResult:
        """Sanitize text by redacting secrets and credentials.

        Args:
            text: Input text.

        Returns:
            SanitizationResult with cleaned text and issue records.
        """
        if not self.enabled:
            return SanitizationResult(cleaned_text=text)

        self.stats["texts_processed"] += 1
        issues: list[dict[str, str]] = []
        cleaned = text

        # Apply all patterns
        for name, pattern, replacement in self._patterns:
            matches = list(pattern.finditer(cleaned))
            if matches:
                cleaned = pattern.sub(replacement, cleaned)
                for _ in matches:
                    issues.append(
                        {
                            "type": name,
                            "replacement": replacement,
                        }
                    )
                    self.pattern_stats[name] = self.pattern_stats.get(name, 0) + 1

        # Block internal URLs
        if self.block_internal_urls:
            for domain in self.internal_domains:
                url_pattern = re.compile(
                    rf"https?://[^\s]*{re.escape(domain)}[^\s]*",
                    re.IGNORECASE,
                )
                if url_pattern.search(cleaned):
                    cleaned = url_pattern.sub("[INTERNAL_URL]", cleaned)
                    issues.append(
                        {
                            "type": "internal_url",
                            "domain": domain,
                            "replacement": "[INTERNAL_URL]",
                        }
                    )

        if issues:
            self.stats["texts_with_secrets"] += 1
            self.stats["total_redactions"] += len(issues)

        return SanitizationResult(cleaned_text=cleaned, issues_found=issues)

    def sanitize_text(self, text: str) -> str:
        """Convenience: sanitize and return only cleaned text."""
        return self.sanitize(text).cleaned_text

    def has_secrets(self, text: str) -> bool:
        """Check if text contains any secrets (without modifying it)."""
        for _, pattern, _ in self._patterns:
            if pattern.search(text):
                return True
        return False

    def summary(self) -> dict[str, Any]:
        """Return sanitization statistics."""
        return {
            **self.stats,
            "patterns_triggered": {k: v for k, v in self.pattern_stats.items() if v > 0},
        }
