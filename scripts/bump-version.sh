#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
#  Auralith Data Pipeline — Version Bump & Tag
#
#  Usage:
#    ./scripts/bump-version.sh 2.1.0          # bump to 2.1.0
#    ./scripts/bump-version.sh 2.1.0 --push   # bump, commit, tag, push
#    ./scripts/bump-version.sh 2.1.0-rc.1     # pre-release tag
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
PUSH=false

if [ -z "$NEW_VERSION" ]; then
    CURRENT=$(python3 -c "
import re
with open('pyproject.toml') as f:
    m = re.search(r'version\s*=\s*\"([^\"]+)\"', f.read())
    print(m.group(1) if m else 'unknown')
")
    echo ""
    echo -e "  Current version: ${BOLD}${CURRENT}${NC}"
    echo ""
    echo "  Usage: $0 <new-version> [--push]"
    echo ""
    echo "  Examples:"
    echo "    $0 2.1.0          # bump to 2.1.0"
    echo "    $0 2.1.0 --push   # bump + commit + tag + push"
    echo "    $0 3.0.0-rc.1     # pre-release"
    echo ""
    exit 1
fi

for arg in "${@:2}"; do
    case "$arg" in
        --push) PUSH=true ;;
        *) fail "Unknown option: $arg" ;;
    esac
done

# Strip pre-release suffix for file version (files use base version)
BASE_VERSION="${NEW_VERSION%%-*}"

# ── Validate format ──────────────────────────────────────────────────
if ! echo "$BASE_VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    fail "Invalid version format: '$BASE_VERSION' (expected X.Y.Z)"
fi

# ── Current version ──────────────────────────────────────────────────
CURRENT=$(python3 -c "
import re
with open('pyproject.toml') as f:
    m = re.search(r'version\s*=\s*\"([^\"]+)\"', f.read())
    print(m.group(1) if m else 'unknown')
")
info "Current version: ${CURRENT}"
info "New version:     ${BASE_VERSION} (tag: v${NEW_VERSION})"
echo ""

# ── Confirm ──────────────────────────────────────────────────────────
read -r -p "  Proceed? [y/N]: " CONFIRM
if [[ ! "$CONFIRM" =~ ^[yY]$ ]]; then
    echo "  Aborted."
    exit 0
fi

# ── Update pyproject.toml ────────────────────────────────────────────
info "Updating pyproject.toml ..."
sed -i.bak "s/^version = \".*\"/version = \"${BASE_VERSION}\"/" pyproject.toml
rm -f pyproject.toml.bak
ok "pyproject.toml → ${BASE_VERSION}"

# ── Update __init__.py ───────────────────────────────────────────────
INIT_FILE="src/auralith_pipeline/__init__.py"
info "Updating ${INIT_FILE} ..."
sed -i.bak "s/^__version__ = \".*\"/__version__ = \"${BASE_VERSION}\"/" "$INIT_FILE"
rm -f "${INIT_FILE}.bak"
ok "__init__.py → ${BASE_VERSION}"

# ── Git operations ───────────────────────────────────────────────────
if [ "$PUSH" = true ]; then
    info "Committing version bump ..."
    git add pyproject.toml "$INIT_FILE"
    git commit -m "release: v${NEW_VERSION}" -m "Bump version to ${BASE_VERSION}"
    ok "Committed"

    info "Creating tag v${NEW_VERSION} ..."
    git tag -a "v${NEW_VERSION}" -m "Release v${NEW_VERSION}"
    ok "Tagged"

    info "Pushing to origin ..."
    git push origin HEAD
    git push origin "v${NEW_VERSION}"
    ok "Pushed — release pipeline will start automatically"
else
    echo ""
    echo -e "  ${BOLD}Files updated. To complete the release:${NC}"
    echo ""
    echo -e "    ${CYAN}git add pyproject.toml ${INIT_FILE}${NC}"
    echo -e "    ${CYAN}git commit -m \"release: v${NEW_VERSION}\"${NC}"
    echo -e "    ${CYAN}git tag -a v${NEW_VERSION} -m \"Release v${NEW_VERSION}\"${NC}"
    echo -e "    ${CYAN}git push origin HEAD && git push origin v${NEW_VERSION}${NC}"
    echo ""
fi

echo ""
echo -e "${GREEN}${BOLD}✓ Version bumped to ${BASE_VERSION} (tag: v${NEW_VERSION})${NC}"
echo ""
