"""Security module — PII scrubbing, data sanitization, and privacy compliance.

Handles removal of ALL personally identifiable information (PII) across
every jurisdiction worldwide. No user private data (emails, phone numbers,
national IDs, financial data, biometrics, etc.) should ever reach training.

Modules:
    pii_scrubber:       Multi-jurisdiction PII detection and redaction
    data_sanitizer:     Content-level sanitization (credentials, secrets, keys)
    privacy_config:     Privacy policy configuration (GDPR, CCPA, LGPD, etc.)
    audit:              Privacy-specific audit trail for compliance reporting
"""

from auralith_pipeline.security.audit import PrivacyAuditLogger
from auralith_pipeline.security.data_sanitizer import DataSanitizer
from auralith_pipeline.security.pii_scrubber import PIIScrubber
from auralith_pipeline.security.privacy_config import PrivacyConfig, PrivacyPolicy

__all__ = [
    "PIIScrubber",
    "DataSanitizer",
    "PrivacyConfig",
    "PrivacyPolicy",
    "PrivacyAuditLogger",
]
