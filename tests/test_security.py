"""Tests for the security module — PII scrubbing, data sanitization, and audit."""

import json
import tempfile
from pathlib import Path

import pytest

from auralith_pipeline.security.audit import PrivacyAuditLogger
from auralith_pipeline.security.data_sanitizer import DataSanitizer
from auralith_pipeline.security.pii_scrubber import PIIScrubber, ScrubResult
from auralith_pipeline.security.privacy_config import (
    GLOBAL_STRICT_POLICY,
    PIICategory,
    PrivacyConfig,
    PrivacyPolicy,
)

# ════════════════════════════════════════════════════════════════
#  PIIScrubber tests
# ════════════════════════════════════════════════════════════════


class TestPIIScrubber:
    """Test multi-jurisdiction PII scrubbing."""

    def setup_method(self) -> None:
        self.scrubber = PIIScrubber()

    # ── Email ──

    def test_scrub_email(self) -> None:
        result = self.scrubber.scrub("Contact me at john.doe@example.com for info.")
        assert "[EMAIL]" in result.cleaned_text
        assert "john.doe@example.com" not in result.cleaned_text
        assert PIICategory.EMAIL in result.categories_found

    def test_scrub_multiple_emails(self) -> None:
        text = "Send to alice@test.org or bob@company.co.uk"
        result = self.scrubber.scrub(text)
        assert result.cleaned_text.count("[EMAIL]") == 2
        assert "alice@test.org" not in result.cleaned_text
        assert "bob@company.co.uk" not in result.cleaned_text

    # ── Phone numbers (international) ──

    def test_scrub_us_phone(self) -> None:
        result = self.scrubber.scrub("Call 555-123-4567 or (555) 987-6543")
        assert "555-123-4567" not in result.cleaned_text
        assert result.pii_found

    def test_scrub_uk_phone(self) -> None:
        result = self.scrubber.scrub("Ring +44 20 7946 0958 please")
        assert "+44 20 7946 0958" not in result.cleaned_text

    def test_scrub_india_phone(self) -> None:
        result = self.scrubber.scrub("WhatsApp me at +91 9876543210")
        assert "9876543210" not in result.cleaned_text

    # ── Government IDs ──

    def test_scrub_us_ssn(self) -> None:
        result = self.scrubber.scrub("SSN: 123-45-6789")
        assert "123-45-6789" not in result.cleaned_text

    def test_scrub_india_aadhaar(self) -> None:
        result = self.scrubber.scrub("Aadhaar: 1234 5678 9012")
        assert "1234 5678 9012" not in result.cleaned_text

    def test_scrub_brazil_cpf(self) -> None:
        result = self.scrubber.scrub("CPF: 123.456.789-00")
        assert "123.456.789-00" not in result.cleaned_text

    def test_scrub_singapore_nric(self) -> None:
        result = self.scrubber.scrub("NRIC: S1234567D")
        assert "S1234567D" not in result.cleaned_text

    # ── Financial ──

    def test_scrub_credit_card(self) -> None:
        result = self.scrubber.scrub("Card: 4111-1111-1111-1111")
        assert "4111-1111-1111-1111" not in result.cleaned_text
        assert result.pii_found

    def test_scrub_iban(self) -> None:
        # IBAN without spaces (avoids phone regex overlap)
        result = self.scrubber.scrub("IBAN: DE89370400440532013000")
        assert "DE89370400440532013000" not in result.cleaned_text
        assert result.pii_found

    def test_scrub_crypto_btc(self) -> None:
        result = self.scrubber.scrub("Send BTC to 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")
        assert "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa" not in result.cleaned_text

    def test_scrub_crypto_eth(self) -> None:
        result = self.scrubber.scrub("ETH: 0x742d35Cc6634C0532925a3b844Bc9e7595f2bD18")
        assert "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD18" not in result.cleaned_text

    # ── IP addresses ──

    def test_scrub_ipv4(self) -> None:
        result = self.scrubber.scrub("Server at 192.168.1.100 is down")
        assert "192.168.1.100" not in result.cleaned_text
        assert PIICategory.IP_ADDRESS in result.categories_found

    # ── Credentials ──

    def test_scrub_aws_key(self) -> None:
        result = self.scrubber.scrub("key: AKIAIOSFODNN7EXAMPLE")
        assert "AKIAIOSFODNN7EXAMPLE" not in result.cleaned_text

    def test_scrub_jwt(self) -> None:
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        result = self.scrubber.scrub(f"Token: {jwt}")
        assert jwt not in result.cleaned_text

    def test_scrub_password(self) -> None:
        result = self.scrubber.scrub("password=SuperSecret123!")
        assert "SuperSecret123!" not in result.cleaned_text

    # ── GPS coordinates ──

    def test_scrub_gps(self) -> None:
        result = self.scrubber.scrub("Location: 40.7128, -74.0060")
        assert "40.7128" not in result.cleaned_text

    # ── Date of birth ──

    def test_scrub_dob(self) -> None:
        result = self.scrubber.scrub("DOB: 01/15/1990")
        assert "01/15/1990" not in result.cleaned_text

    # ── Clean text passes through ──

    def test_clean_text_unchanged(self) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        result = self.scrubber.scrub(text)
        assert result.cleaned_text == text
        assert not result.pii_found
        assert result.count == 0

    # ── Replacement styles ──

    def test_hash_replacement(self) -> None:
        config = PrivacyConfig(replacement_style="hash")
        scrubber = PIIScrubber(config=config)
        result = scrubber.scrub("Email: test@example.com")
        assert "[REDACTED:" in result.cleaned_text
        assert "test@example.com" not in result.cleaned_text

    def test_remove_replacement(self) -> None:
        config = PrivacyConfig(replacement_style="remove")
        scrubber = PIIScrubber(config=config)
        result = scrubber.scrub("Email: test@example.com ok")
        assert "test@example.com" not in result.cleaned_text

    # ── has_pii check ──

    def test_has_pii_true(self) -> None:
        assert self.scrubber.has_pii("Contact: alice@test.com")

    def test_has_pii_false(self) -> None:
        assert not self.scrubber.has_pii("The sky is blue.")

    # ── Stats ──

    def test_stats_accumulate(self) -> None:
        self.scrubber.scrub("Email: a@b.com")
        self.scrubber.scrub("Phone: 555-123-4567")
        self.scrubber.scrub("Nothing here.")
        summary = self.scrubber.summary()
        assert summary["texts_processed"] == 3
        assert summary["texts_with_pii"] == 2


# ════════════════════════════════════════════════════════════════
#  DataSanitizer tests
# ════════════════════════════════════════════════════════════════


class TestDataSanitizer:
    """Test credential and secret sanitization."""

    def setup_method(self) -> None:
        self.sanitizer = DataSanitizer()

    def test_sanitize_aws_access_key(self) -> None:
        result = self.sanitizer.sanitize("Key: AKIAIOSFODNN7EXAMPLE")
        assert "AKIAIOSFODNN7EXAMPLE" not in result.cleaned_text
        assert "[AWS_KEY]" in result.cleaned_text

    def test_sanitize_aws_secret_key(self) -> None:
        text = "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        result = self.sanitizer.sanitize(text)
        assert "wJalrXUtnFEMI" not in result.cleaned_text

    def test_sanitize_github_token(self) -> None:
        result = self.sanitizer.sanitize("Token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij1234")
        assert "ghp_" not in result.cleaned_text
        assert "[GITHUB_TOKEN]" in result.cleaned_text

    def test_sanitize_db_connection_string(self) -> None:
        result = self.sanitizer.sanitize("DB: postgresql://user:pass@host:5432/db")
        assert "user:pass" not in result.cleaned_text
        assert "[DB_CONNECTION_STRING]" in result.cleaned_text

    def test_sanitize_jwt(self) -> None:
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        result = self.sanitizer.sanitize(f"Bearer {jwt}")
        assert jwt not in result.cleaned_text

    def test_sanitize_private_key(self) -> None:
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIBog...\n-----END RSA PRIVATE KEY-----"
        result = self.sanitizer.sanitize(text)
        assert "MIIBog" not in result.cleaned_text
        assert "[PRIVATE_KEY_REDACTED]" in result.cleaned_text

    def test_sanitize_password(self) -> None:
        result = self.sanitizer.sanitize("password=MyS3cretP@ss!")
        assert "MyS3cretP@ss!" not in result.cleaned_text

    def test_sanitize_url_credentials(self) -> None:
        result = self.sanitizer.sanitize("https://admin:password123@api.example.com/v1")
        assert "admin:password123" not in result.cleaned_text

    def test_sanitize_internal_url(self) -> None:
        result = self.sanitizer.sanitize("Check https://dashboard.internal/status")
        assert "[INTERNAL_URL]" in result.cleaned_text

    def test_sanitize_slack_webhook(self) -> None:
        result = self.sanitizer.sanitize("Hook: https://hooks.slack.com/services/T00/B00/xxxx")
        assert "hooks.slack.com" not in result.cleaned_text

    def test_clean_text_passes(self) -> None:
        text = "Machine learning is great."
        result = self.sanitizer.sanitize(text)
        assert result.cleaned_text == text
        assert not result.had_issues

    def test_disabled_sanitizer(self) -> None:
        sanitizer = DataSanitizer(enabled=False)
        text = "AKIAIOSFODNN7EXAMPLE should stay"
        result = sanitizer.sanitize(text)
        assert result.cleaned_text == text

    def test_has_secrets(self) -> None:
        assert self.sanitizer.has_secrets("key: AKIAIOSFODNN7EXAMPLE")
        assert not self.sanitizer.has_secrets("The weather is nice.")


# ════════════════════════════════════════════════════════════════
#  PrivacyAuditLogger tests
# ════════════════════════════════════════════════════════════════


class TestPrivacyAuditLogger:
    """Test privacy audit logging."""

    def test_log_redaction(self) -> None:
        audit = PrivacyAuditLogger()
        audit.log_redaction(
            sample_id="test_1",
            source="wikipedia",
            categories=["email", "phone"],
            redaction_count=3,
        )
        summary = audit.summary()
        assert summary["redactions"] == 1
        assert summary["total_events"] == 1

    def test_log_blocked(self) -> None:
        audit = PrivacyAuditLogger()
        audit.log_blocked(
            sample_id="test_2",
            source="the_pile",
            reason="pii_rescan_failed",
        )
        summary = audit.summary()
        assert summary["blocked"] == 1

    def test_log_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            audit = PrivacyAuditLogger(log_path=str(log_path))

            audit.log_redaction(
                sample_id="s1",
                source="wiki",
                categories=["email"],
                redaction_count=1,
            )
            audit.flush()

            assert log_path.exists()
            lines = log_path.read_text().strip().split("\n")
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["event_type"] == "pii_redaction"
            assert entry["sample_id"] == "s1"

    def test_summary_category_frequency(self) -> None:
        audit = PrivacyAuditLogger()
        audit.log_redaction("s1", "wiki", ["email", "phone"], 2)
        audit.log_redaction("s2", "wiki", ["email", "ssn"], 2)
        summary = audit.summary()
        assert summary["category_frequency"]["email"] == 2
        assert summary["category_frequency"]["phone"] == 1
        assert summary["category_frequency"]["ssn"] == 1

    def test_disabled_logger(self) -> None:
        audit = PrivacyAuditLogger(enabled=False)
        audit.log_redaction("s1", "wiki", ["email"], 1)
        summary = audit.summary()
        assert summary["total_events"] == 0


# ════════════════════════════════════════════════════════════════
#  PrivacyConfig tests
# ════════════════════════════════════════════════════════════════


class TestPrivacyConfig:
    """Test privacy configuration."""

    def test_default_is_strict(self) -> None:
        config = PrivacyConfig()
        assert config.enabled is True
        assert config.mode == "strict"
        assert config.rescan_after_processing is True

    def test_global_strict_policy_covers_all(self) -> None:
        policy = GLOBAL_STRICT_POLICY
        for cat in PIICategory:
            assert cat in policy.categories, f"Missing category: {cat}"

    def test_custom_policy(self) -> None:
        policy = PrivacyPolicy(
            name="Custom",
            jurisdiction="TEST",
            categories=frozenset({PIICategory.EMAIL, PIICategory.PHONE}),
        )
        assert PIICategory.EMAIL in policy.categories
        assert PIICategory.SSN not in policy.categories


# ════════════════════════════════════════════════════════════════
#  Integration tests
# ════════════════════════════════════════════════════════════════


class TestSecurityIntegration:
    """End-to-end integration tests combining scrubber + sanitizer."""

    def test_combined_scrub_and_sanitize(self) -> None:
        """Ensure both layers work together."""
        scrubber = PIIScrubber()
        sanitizer = DataSanitizer()

        text = (
            "Contact John at john@example.com. "
            "AWS key: AKIAIOSFODNN7EXAMPLE. "
            "Server: 192.168.1.1. "
            "DB: postgresql://admin:secret@db.internal:5432/prod"
        )

        # Layer 1: PII scrub
        result = scrubber.scrub(text)
        cleaned = result.cleaned_text

        # Layer 2: Secret sanitize
        sanitized = sanitizer.sanitize(cleaned)
        final = sanitized.cleaned_text

        # Nothing private should remain
        assert "john@example.com" not in final
        assert "AKIAIOSFODNN7EXAMPLE" not in final
        assert "192.168.1.1" not in final
        assert "admin:secret" not in final
        assert "db.internal" not in final

    def test_idempotent_scrubbing(self) -> None:
        """Running scrubber twice should produce same output."""
        scrubber = PIIScrubber()
        text = "Email: test@example.com, Phone: 555-123-4567"
        once = scrubber.scrub(text).cleaned_text
        twice = scrubber.scrub(once).cleaned_text
        assert once == twice

    def test_clean_text_survives_both_layers(self) -> None:
        """Clean text should pass through unchanged."""
        scrubber = PIIScrubber()
        sanitizer = DataSanitizer()
        text = "The Transformer architecture uses multi-head attention mechanisms."
        assert scrubber.scrub(text).cleaned_text == text
        assert sanitizer.sanitize(text).cleaned_text == text
