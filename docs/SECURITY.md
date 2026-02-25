# 🔒 Security & Privacy — Auralith Data Pipeline

## Principles

1. **No private user data in training** — Zero tolerance for PII from any jurisdiction worldwide.
2. **No static credentials in infrastructure** — IRSA / Workload Identity only.
3. **No external LLM calls for data generation** — Avoids copyright risk and data leakage.
4. **Defense in depth** — Multiple overlapping layers, any one of which is sufficient.
5. **Full audit trail** — Every redaction and decision is logged for regulatory review.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 Incoming Data Sample                    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 1: PIIScrubber (security/pii_scrubber.py)        │
│  • Emails, phones (all intl formats)                    │
│  • SSN, Aadhaar, CPF, SIN, NRIC, NIR, NHS, BSN, etc.    │
│  • Credit cards, IBAN, SWIFT, crypto wallets            │
│  • IP addresses (v4+v6), MAC addresses                  │
│  • GPS coordinates, postal codes                        │
│  • Dates of birth, passwords, API keys, JWTs            │
│  • Private keys (PEM), AWS keys                         │
│  → Replaces with [EMAIL], [PHONE], [SSN], etc.          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 2: DataSanitizer (security/data_sanitizer.py)    │
│  • AWS access keys (AKIA...) + secret keys              │
│  • GCP service account keys                             │
│  • Azure connection strings                             │
│  • Database connection strings                          │
│  • GitHub tokens (ghp_...), Slack webhooks              │
│  • Bearer tokens, .env secrets                          │
│  • Internal/corporate URLs                              │
│  → Replaces with [AWS_KEY], [DB_CONNECTION_STRING], etc.│
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Legacy PIIRemover (preprocessor.py)           │
│  • Backup regex layer (original v1 patterns)            │
│  • Runs during standard preprocessing                   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 4: Rescan (optional, config: fail_on_pii=true)   │
│  • Re-runs PIIScrubber after ALL processing             │
│  • If PII still found → sample is BLOCKED entirely      │
│  → Paranoid mode for maximum safety                     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
                   Clean training data
```

---

## Jurisdictions Covered

| Jurisdiction | Law | PII Categories Scrubbed |
|---|---|---|
| 🇪🇺 EU/EEA | GDPR | All |
| 🇺🇸 California | CCPA | All |
| 🇧🇷 Brazil | LGPD | All (incl. CPF) |
| 🇨🇦 Canada | PIPEDA | All (incl. SIN) |
| 🇿🇦 South Africa | POPIA | All |
| 🇮🇳 India | DPDPA | All (incl. Aadhaar) |
| 🇯🇵 Japan | APPI | All (incl. My Number) |
| 🇸🇬 Singapore | PDPA | All (incl. NRIC) |
| 🇬🇧 UK | UK GDPR | All (incl. NHS #) |
| 🇫🇷 France | GDPR + CNIL | All (incl. NIR/INSEE) |
| 🇳🇱 Netherlands | GDPR | All (incl. BSN) |
| 🇲🇽 Mexico | LFPDPPP | All (incl. CURP) |
| 🇪🇸 Spain | LOPDGDD | All (incl. DNI) |
| 🇵🇱 Poland | GDPR | All (incl. PESEL) |
| 🇦🇺 Australia | Privacy Act | All (incl. TFN) |
| 🇲🇾 Malaysia | PDPA | All (incl. NRIC) |
| 🌍 Global | — | Catch-all generic ID patterns |

---

## Configuration

### Strict mode (default — recommended)

```yaml
# configs/production.yaml
security:
  enabled: true
  mode: strict                   # scrub ALL PII regardless of jurisdiction
  replacement_style: tag         # [EMAIL], [PHONE], etc.
  rescan_after_processing: true  # double-check after preprocessing
  log_redactions: true           # full audit trail
  fail_on_pii: false             # set true to BLOCK residual PII
  audit_log_path: ./data/audit/privacy.jsonl
  sanitize_secrets: true         # AWS keys, passwords, tokens
  block_internal_urls: true      # corporate intranet URLs
```

### Paranoid mode (blocks any sample with residual PII)

```yaml
security:
  enabled: true
  mode: strict
  fail_on_pii: true    # ← sample is DROPPED if PII detected after scrubbing
```

---

## Infrastructure Security

### ❌ NEVER: Static AWS credentials

```yaml
# BAD — DO NOT DO THIS
env:
  - name: AWS_ACCESS_KEY_ID
    valueFrom:
      secretKeyRef:
        name: aws-credentials
        key: access-key-id
```

### ✅ ALWAYS: IRSA (IAM Roles for Service Accounts)

```yaml
# ServiceAccount with IRSA annotation
apiVersion: v1
kind: ServiceAccount
metadata:
  name: auralith-pipeline
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/auralith-pipeline
```

The EKS pod identity webhook automatically injects:
- `AWS_ROLE_ARN`
- `AWS_WEB_IDENTITY_TOKEN_FILE`

The AWS SDK picks these up with zero code changes.

### IAM Policy (minimum required)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::auralith-training-data",
        "arn:aws:s3:::auralith-training-data/*"
      ]
    }
  ]
}
```

### Trust Policy (restricts to this SA only)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::123456789012:oidc-provider/oidc.eks.us-east-1.amazonaws.com/id/EXAMPLED539D4633E53DE1B71EXAMPLE"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "oidc.eks.us-east-1.amazonaws.com/id/EXAMPLED539D4633E53DE1B71EXAMPLE:sub": "system:serviceaccount:default:auralith-pipeline"
        }
      }
    }
  ]
}
```

---

## Synthetic Data Policy

**We do NOT call external LLMs (OpenAI, Anthropic, etc.) for data generation.**

Reasons:
1. **Copyright**: LLM outputs may be derivative works under their ToS.
2. **Data leakage**: Sending training data to third-party APIs exposes it.
3. **Unverifiable quality**: Can't audit what the model actually generated.

Instead, we use `LocalDataAugmenter` with safe, deterministic operations:
- Sentence shuffling (preserves meaning)
- Paragraph extraction (chunking)
- Token-level noise (typo simulation for robustness)
- Back-translation via local MarianMT models (no API)

---

## Audit Log Format

Privacy audit log (`privacy.jsonl`):

```json
{
  "timestamp": 1740000000.0,
  "event_type": "pii_redaction",
  "sample_id": "wiki_12345",
  "source": "wikipedia",
  "categories": ["email", "phone", "ip_address"],
  "redaction_count": 3,
  "action": "redacted",
  "details": {}
}
```

---

## Running Security Tests

```bash
python -m pytest tests/test_security.py -v
```
