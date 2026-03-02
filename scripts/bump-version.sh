#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
#  Auralith Data Pipeline 
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
fail()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Args ──────────────────────────────────────────────────────────────
NEW_VERSION="${1:-}"

if [ -z "$NEW_VERSION" ]; then
    LATEST=$(git tag -l 'v[0-9]*.[0-9]*.[0-9]*' --sort=-v:refname | head -1)
    echo ""
    echo -e "  Latest tag: ${BOLD}${LATEST:-none}${NC}"
    echo ""
    echo "  Usage: $0 <new-version>"
    echo ""
    echo "  Examples:"
    echo "    $0 0.2.0          # minor bump"
    echo "    $0 1.0.0          # major bump"
    echo "    $0 1.0.0-rc.1     # pre-release"
    echo ""
    echo "  Note: Patch versions (e.g. 0.1.1 → 0.1.2) are created"
    echo "        automatically on every PR merge to main."
    echo ""
    exit 1
fi

BASE_VERSION="${NEW_VERSION%%-*}"

# ── Validate format ──────────────────────────────────────────────────
if ! echo "$BASE_VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    fail "Invalid version format: '$BASE_VERSION' (expected X.Y.Z)"
fi

TAG="v${NEW_VERSION}"

if git rev-parse "refs/tags/${TAG}" >/dev/null 2>&1; then
    fail "Tag ${TAG} already exists!"
fi

LATEST=$(git tag -l 'v[0-9]*.[0-9]*.[0-9]*' --sort=-v:refname | head -1)
info "Latest tag:  ${LATEST:-none}"
info "New tag:     ${TAG}"
echo ""

# ── Confirm ──────────────────────────────────────────────────────────
read -r -p "  Create and push ${TAG}? [y/N]: " CONFIRM
if [[ ! "$CONFIRM" =~ ^[yY]$ ]]; then
    echo "  Aborted."
    exit 0
fi

# ── Create and push tag ──────────────────────────────────────────────
info "Creating tag ${TAG} ..."
git tag -a "${TAG}" -m "Release ${TAG}"
ok "Tag created"

info "Pushing to origin ..."
git push origin "${TAG}"
ok "Tag pushed — release pipeline will start automatically"

echo ""
echo -e "${GREEN}${BOLD}✓ Created ${TAG} — release pipeline triggered${NC}"
echo ""
