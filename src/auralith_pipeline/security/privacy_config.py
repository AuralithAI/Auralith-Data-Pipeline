"""Privacy policy configuration for multi-jurisdiction compliance.

Supports GDPR (EU), CCPA (California), LGPD (Brazil), PIPEDA (Canada),
POPIA (South Africa), PDPA (Singapore/Thailand), APPI (Japan),
Privacy Act (Australia), DPDPA (India), and more.

Each policy defines which PII categories must be scrubbed for that jurisdiction.
The default mode is STRICT — scrub everything globally.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class PIICategory(str, Enum):
    """Categories of personally identifiable information."""

    # ── Universal identifiers ──
    EMAIL = "email"
    PHONE = "phone"
    IP_ADDRESS = "ip_address"
    URL_CREDENTIALS = "url_credentials"

    # ── Names & demographics ──
    PERSON_NAME = "person_name"
    DATE_OF_BIRTH = "date_of_birth"
    AGE = "age"
    GENDER = "gender"
    ETHNICITY = "ethnicity"

    # ── Physical / postal ──
    STREET_ADDRESS = "street_address"
    POSTAL_CODE = "postal_code"
    GPS_COORDINATES = "gps_coordinates"

    # ── Government IDs (per country) ──
    SSN = "ssn"                           # US Social Security Number
    NHS_NUMBER = "nhs_number"             # UK National Health Service
    NIR = "nir"                           # France — numéro INSEE
    AADHAAR = "aadhaar"                   # India
    CPF = "cpf"                           # Brazil — Cadastro de Pessoas Físicas
    SIN = "sin"                           # Canada — Social Insurance Number
    TFN = "tfn"                           # Australia — Tax File Number
    BSN = "bsn"                           # Netherlands — Burgerservicenummer
    NRIC = "nric"                         # Singapore / Malaysia
    MY_NUMBER = "my_number"               # Japan — マイナンバー
    CURP = "curp"                         # Mexico
    DNI = "dni"                           # Spain / Argentina
    PESEL = "pesel"                       # Poland
    NATIONAL_ID_GENERIC = "national_id"   # Catch-all

    # ── Financial ──
    CREDIT_CARD = "credit_card"
    BANK_ACCOUNT = "bank_account"
    IBAN = "iban"
    SWIFT_BIC = "swift_bic"
    CRYPTO_WALLET = "crypto_wallet"

    # ── Digital / credentials ──
    PASSWORD = "password"
    API_KEY = "api_key"
    AWS_KEY = "aws_key"
    PRIVATE_KEY = "private_key"
    JWT_TOKEN = "jwt_token"
    OAUTH_TOKEN = "oauth_token"
    SESSION_ID = "session_id"
    MAC_ADDRESS = "mac_address"

    # ── Health ──
    MEDICAL_RECORD = "medical_record"
    HEALTH_INSURANCE_ID = "health_insurance_id"

    # ── Biometrics ──
    BIOMETRIC_DATA = "biometric_data"


# All categories — used for STRICT mode
ALL_CATEGORIES: frozenset[PIICategory] = frozenset(PIICategory)


@dataclass(frozen=True)
class PrivacyPolicy:
    """A named privacy policy that defines which PII must be scrubbed.

    Attributes:
        name:           Human-readable policy name
        jurisdiction:   ISO 3166-1 alpha-2 or region name
        categories:     PII categories to scrub
        strict:         If True, also scrub unrecognised patterns that look like IDs
        retain_hashes:  If True, replace PII with deterministic hash (for dedup)
    """

    name: str
    jurisdiction: str
    categories: frozenset[PIICategory]
    strict: bool = True
    retain_hashes: bool = False


# ── Pre-defined jurisdiction policies ──

GDPR_POLICY = PrivacyPolicy(
    name="GDPR (EU/EEA)",
    jurisdiction="EU",
    categories=ALL_CATEGORIES,
    strict=True,
)

CCPA_POLICY = PrivacyPolicy(
    name="CCPA (California)",
    jurisdiction="US-CA",
    categories=ALL_CATEGORIES,
    strict=True,
)

LGPD_POLICY = PrivacyPolicy(
    name="LGPD (Brazil)",
    jurisdiction="BR",
    categories=ALL_CATEGORIES,
    strict=True,
)

PIPEDA_POLICY = PrivacyPolicy(
    name="PIPEDA (Canada)",
    jurisdiction="CA",
    categories=ALL_CATEGORIES,
    strict=True,
)

POPIA_POLICY = PrivacyPolicy(
    name="POPIA (South Africa)",
    jurisdiction="ZA",
    categories=ALL_CATEGORIES,
    strict=True,
)

DPDPA_POLICY = PrivacyPolicy(
    name="DPDPA (India)",
    jurisdiction="IN",
    categories=ALL_CATEGORIES,
    strict=True,
)

APPI_POLICY = PrivacyPolicy(
    name="APPI (Japan)",
    jurisdiction="JP",
    categories=ALL_CATEGORIES,
    strict=True,
)

PDPA_POLICY = PrivacyPolicy(
    name="PDPA (Singapore)",
    jurisdiction="SG",
    categories=ALL_CATEGORIES,
    strict=True,
)

GLOBAL_STRICT_POLICY = PrivacyPolicy(
    name="Global Strict — All Jurisdictions",
    jurisdiction="GLOBAL",
    categories=ALL_CATEGORIES,
    strict=True,
)


@dataclass
class PrivacyConfig:
    """Top-level privacy configuration for the pipeline.

    Attributes:
        enabled:            Master switch for PII scrubbing
        mode:               'strict' = scrub ALL PII globally (recommended)
                            'jurisdiction' = only scrub categories required
                              by the specified policies
        policies:           Active policies (default: GLOBAL_STRICT)
        replacement_style:  How to replace PII:
                            'tag'   → [EMAIL], [PHONE], etc.
                            'hash'  → sha256 truncated (for dedup-safe redaction)
                            'remove' → delete the PII span entirely
        rescan_after_processing: Run a second PII pass after all other
                                 preprocessing (catches edge cases)
        log_redactions:     Write every redaction to the privacy audit log
        fail_on_pii:        If True, raise an error if PII is still found
                            in the final output (paranoid mode)
    """

    enabled: bool = True
    mode: Literal["strict", "jurisdiction"] = "strict"
    policies: list[PrivacyPolicy] = field(
        default_factory=lambda: [GLOBAL_STRICT_POLICY]
    )
    replacement_style: Literal["tag", "hash", "remove"] = "tag"
    rescan_after_processing: bool = True
    log_redactions: bool = True
    fail_on_pii: bool = False
