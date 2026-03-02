#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
#  Auralith Data Pipeline — One-shot setup
#
#  Detects OS (macOS / Linux / Windows-Git-Bash / WSL), installs Python
#  if missing, creates a venv, installs pip, and installs all deps.
#
#  Usage:
#    chmod +x setup.sh
#    ./setup.sh              # Full install (all extras)
#    ./setup.sh --core       # Core only (text pipeline, ~500 MB)
#    ./setup.sh --dev        # Core + dev tools (tests, linting)
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No colour

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Detect OS ─────────────────────────────────────────────────────────
detect_os() {
    case "$(uname -s)" in
        Linux*)
            if grep -qi microsoft /proc/version 2>/dev/null; then
                OS="wsl"
            else
                OS="linux"
            fi
            ;;
        Darwin*)  OS="macos"   ;;
        CYGWIN*|MINGW*|MSYS*)  OS="windows" ;;
        *)        OS="unknown" ;;
    esac

    # Detect Linux distro family (for package manager selection)
    DISTRO=""
    if [ "$OS" = "linux" ] || [ "$OS" = "wsl" ]; then
        if command -v apt-get &>/dev/null; then
            DISTRO="debian"    # Ubuntu / Debian / Pop!_OS / Mint
        elif command -v dnf &>/dev/null; then
            DISTRO="fedora"    # Fedora / RHEL 8+ / CentOS Stream
        elif command -v yum &>/dev/null; then
            DISTRO="rhel"      # CentOS 7 / older RHEL
        elif command -v pacman &>/dev/null; then
            DISTRO="arch"      # Arch / Manjaro
        elif command -v apk &>/dev/null; then
            DISTRO="alpine"    # Alpine (Docker)
        fi
    fi

    info "Detected OS: ${BOLD}${OS}${NC}$([ -n "$DISTRO" ] && echo " ($DISTRO)")"
}

# ── Parse args ────────────────────────────────────────────────────────
MODE="all"
for arg in "$@"; do
    case "$arg" in
        --core)  MODE="core" ;;
        --dev)   MODE="dev"  ;;
        --help|-h)
            echo "Usage: ./setup.sh [--core | --dev]"
            echo ""
            echo "  (default)  Install with ALL extras (~3 GB, includes PyTorch)"
            echo "  --core     Core only — text pipeline, no heavy ML deps (~500 MB)"
            echo "  --dev      Core + development tools (pytest, black, ruff, mypy)"
            echo ""
            exit 0
            ;;
        *) fail "Unknown option: $arg (try --help)" ;;
    esac
done

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║       Auralith Data Pipeline — Setup                ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

detect_os

# ── Install system build dependencies (per OS) ───────────────────────
install_system_deps() {
    info "Installing system-level build dependencies ..."

    case "$OS" in
        macos)
            # Xcode command-line tools provide gcc/clang; brew for extras
            if ! xcode-select -p &>/dev/null; then
                info "Installing Xcode command-line tools ..."
                xcode-select --install 2>/dev/null || true
                warn "A dialog may have appeared — accept it, then re-run this script."
                exit 0
            fi
            if command -v brew &>/dev/null; then
                brew install libffi openssl readline xz 2>/dev/null || true
            fi
            ok "macOS build dependencies ready"
            ;;
        linux|wsl)
            case "$DISTRO" in
                debian)
                    sudo apt-get update -qq
                    sudo apt-get install -y -qq \
                        build-essential libffi-dev libssl-dev zlib1g-dev \
                        libbz2-dev libreadline-dev libsqlite3-dev \
                        liblzma-dev libncurses5-dev libgdbm-dev \
                        tk-dev curl git >/dev/null
                    ;;
                fedora)
                    sudo dnf install -y --quiet \
                        gcc gcc-c++ make libffi-devel openssl-devel \
                        zlib-devel bzip2-devel readline-devel sqlite-devel \
                        xz-devel ncurses-devel gdbm-devel tk-devel \
                        curl git >/dev/null
                    ;;
                rhel)
                    sudo yum install -y -q \
                        gcc gcc-c++ make libffi-devel openssl-devel \
                        zlib-devel bzip2-devel readline-devel sqlite-devel \
                        xz-devel ncurses-devel gdbm-devel tk-devel \
                        curl git >/dev/null
                    ;;
                arch)
                    sudo pacman -Sy --noconfirm --quiet \
                        base-devel libffi openssl zlib bzip2 readline \
                        sqlite xz ncurses gdbm tk curl git >/dev/null
                    ;;
                alpine)
                    sudo apk add --quiet \
                        build-base libffi-dev openssl-dev zlib-dev \
                        bzip2-dev readline-dev sqlite-dev xz-dev \
                        ncurses-dev gdbm-dev tk-dev curl git >/dev/null
                    ;;
                *)
                    warn "Unknown Linux distro — please install build-essential, libffi-dev, libssl-dev manually"
                    ;;
            esac
            ok "System build dependencies installed"
            ;;
        windows)
            # Git Bash / MSYS2 — system deps are handled by the Python installer
            ok "Windows detected — system deps managed by Python installer"
            ;;
        *)
            warn "Unknown OS — skipping system dependency installation"
            ;;
    esac
}

# ── Install Python (per OS) ──────────────────────────────────────────
install_python() {
    info "Attempting to install Python 3.12 ..."

    case "$OS" in
        macos)
            if command -v brew &>/dev/null; then
                info "Installing Python via Homebrew ..."
                brew install python@3.12
            else
                fail "Homebrew not found. Install it first:  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            fi
            ;;
        linux|wsl)
            case "$DISTRO" in
                debian)
                    # Try deadsnakes PPA for latest Python
                    if ! command -v add-apt-repository &>/dev/null; then
                        sudo apt-get install -y -qq software-properties-common >/dev/null
                    fi
                    sudo add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1 || true
                    sudo apt-get update -qq
                    sudo apt-get install -y -qq python3.12 python3.12-venv python3.12-dev >/dev/null
                    ;;
                fedora)
                    sudo dnf install -y --quiet python3.12 python3.12-devel >/dev/null
                    ;;
                rhel)
                    sudo yum install -y -q python3.12 python3.12-devel >/dev/null 2>&1 \
                        || sudo yum install -y -q python3 python3-devel >/dev/null
                    ;;
                arch)
                    sudo pacman -Sy --noconfirm --quiet python >/dev/null
                    ;;
                alpine)
                    sudo apk add --quiet python3 python3-dev py3-pip >/dev/null
                    ;;
                *)
                    fail "Cannot auto-install Python on this distro. Please install Python 3.10+ manually."
                    ;;
            esac
            ;;
        windows)
            # Check for winget (Windows 10/11 built-in package manager)
            if command -v winget &>/dev/null; then
                info "Installing Python via winget ..."
                winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
            else
                fail "Cannot auto-install Python on Windows without winget.\n       Download from: https://www.python.org/downloads/"
            fi
            ;;
        *)
            fail "Cannot auto-install Python on this OS. Please install Python 3.10+ manually."
            ;;
    esac
}

# ── Check / Install Python ───────────────────────────────────────────
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    warn "Python not found on this system."
    echo ""
    echo -e "  ${BOLD}Options:${NC}"
    echo -e "    ${CYAN}1${NC}) Install Python automatically (requires sudo / admin)"
    echo -e "    ${CYAN}2${NC}) Exit — I'll install Python myself"
    echo ""
    read -r -p "  Choose [1/2]: " CHOICE

    case "$CHOICE" in
        1)
            install_system_deps
            install_python
            # Re-detect Python after install
            for cmd in python3.12 python3.11 python3.10 python3 python; do
                if command -v "$cmd" &>/dev/null; then
                    PYTHON="$cmd"
                    break
                fi
            done
            [ -z "$PYTHON" ] && fail "Python installation succeeded but binary not found. Try opening a new terminal."
            ;;
        *)
            echo ""
            echo -e "  Install Python 3.10+ and re-run this script."
            echo -e "  Download: ${CYAN}https://www.python.org/downloads/${NC}"
            exit 1
            ;;
    esac
fi

# Validate version
PY_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$("$PYTHON" -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$("$PYTHON" -c 'import sys; print(sys.version_info.minor)')

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    warn "Python $PY_VERSION found — but 3.10+ is required."
    echo ""
    echo -e "  ${BOLD}Options:${NC}"
    echo -e "    ${CYAN}1${NC}) Install Python 3.12 automatically (requires sudo / admin)"
    echo -e "    ${CYAN}2${NC}) Exit — I'll upgrade Python myself"
    echo ""
    read -r -p "  Choose [1/2]: " CHOICE

    case "$CHOICE" in
        1)
            install_system_deps
            install_python
            for cmd in python3.12 python3.11 python3.10 python3 python; do
                if command -v "$cmd" &>/dev/null; then
                    ver=$("$cmd" -c 'import sys; print(sys.version_info.minor)')
                    if [ "$ver" -ge 10 ]; then
                        PYTHON="$cmd"
                        break
                    fi
                fi
            done
            PY_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            ;;
        *)
            echo ""
            echo -e "  Upgrade to Python 3.10+ and re-run this script."
            echo -e "  Download: ${CYAN}https://www.python.org/downloads/${NC}"
            exit 1
            ;;
    esac
fi

ok "Python $PY_VERSION detected ($PYTHON)"

# ── Install system build deps (if not done already) ──────────────────
# Only runs if we didn't already install them during Python setup
if [ "${CHOICE:-}" != "1" ]; then
    install_system_deps
fi

# ── Ensure venv module is available ──────────────────────────────────
if ! "$PYTHON" -m venv --help &>/dev/null; then
    warn "Python venv module not found — installing it ..."
    case "$OS" in
        linux|wsl)
            case "$DISTRO" in
                debian)
                    sudo apt-get install -y -qq "python${PY_VERSION}-venv" >/dev/null 2>&1 \
                        || sudo apt-get install -y -qq python3-venv >/dev/null
                    ;;
                *)
                    # Most distros include venv with Python
                    warn "Please install the python3-venv package for your distro"
                    ;;
            esac
            ;;
        *)
            # macOS/Windows ship venv with Python
            ;;
    esac
fi

# ── Create virtual environment ────────────────────────────────────────
VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    info "Virtual environment '$VENV_DIR' already exists — reusing it"
else
    info "Creating virtual environment in ./$VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Virtual environment created"
fi

# ── Activate virtual environment (OS-aware) ──────────────────────────
ACTIVATE_CMD=""
if [ -f "$VENV_DIR/bin/activate" ]; then
    ACTIVATE_CMD="$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    ACTIVATE_CMD="$VENV_DIR/Scripts/activate"
else
    fail "Could not find activation script in $VENV_DIR"
fi

# shellcheck disable=SC1090
source "$ACTIVATE_CMD"
ok "Virtual environment activated ($(python --version))"

# Verify we're inside the venv
VENV_PYTHON="$(which python)"
case "$VENV_PYTHON" in
    *"$VENV_DIR"*)
        ok "Confirmed: using venv Python at $VENV_PYTHON"
        ;;
    *)
        fail "Python is NOT running from venv ($VENV_PYTHON). Activation may have failed."
        ;;
esac

# ── Ensure pip is available ──────────────────────────────────────────
if ! python -m pip --version &>/dev/null; then
    info "pip not found — bootstrapping via ensurepip ..."
    python -m ensurepip --upgrade --default-pip 2>/dev/null \
        || fail "Could not bootstrap pip. Install it manually: https://pip.pypa.io/en/stable/installation/"
    ok "pip bootstrapped"
fi

info "Upgrading pip + setuptools + wheel ..."
python -m pip install --upgrade pip setuptools wheel --quiet
ok "pip $(python -m pip --version | awk '{print $2}') ready"

# ── Install project dependencies ─────────────────────────────────────
case "$MODE" in
    core)
        info "Installing core dependencies ..."
        python -m pip install -e . --quiet
        ok "Core installation complete"
        ;;
    dev)
        info "Installing core + dev dependencies ..."
        python -m pip install -e ".[dev]" --quiet
        ok "Dev installation complete"
        # Set up pre-commit hooks if available
        if command -v pre-commit &>/dev/null; then
            info "Installing pre-commit hooks ..."
            pre-commit install --quiet
            ok "Pre-commit hooks installed"
        fi
        ;;
    all)
        info "Installing ALL dependencies (this may take a while — ~3 GB with PyTorch) ..."
        python -m pip install -e ".[all]" --quiet
        ok "Full installation complete"
        # Set up pre-commit hooks
        if command -v pre-commit &>/dev/null; then
            info "Installing pre-commit hooks ..."
            pre-commit install --quiet
            ok "Pre-commit hooks installed"
        fi
        ;;
esac

# ── Verify installation ──────────────────────────────────────────────
info "Verifying installation ..."

if python -c "import auralith_pipeline" 2>/dev/null; then
    VERSION=$(python -c "from auralith_pipeline import __version__; print(__version__)")
    ok "auralith-pipeline v${VERSION} installed successfully"
else
    fail "Installation verification failed — 'import auralith_pipeline' raised an error"
fi

# Verify CLI entry point
if command -v auralith-pipeline &>/dev/null; then
    ok "CLI entry point 'auralith-pipeline' is available"
else
    warn "CLI entry point not on PATH — try: pip install -e ."
fi

# ── Done ──────────────────────────────────────────────────────────────
# Show the correct activation command for the user's OS
if [ "$OS" = "windows" ]; then
    ACTIVATE_HINT=".\\\\${VENV_DIR}\\\\Scripts\\\\activate"
else
    ACTIVATE_HINT="source ${VENV_DIR}/bin/activate"
fi

echo ""
echo -e "${GREEN}${BOLD}✓ Setup complete!${NC}"
echo ""
echo -e "  ${BOLD}Environment:${NC}"
echo -e "    Python:  ${CYAN}$(python --version)${NC}"
echo -e "    Venv:    ${CYAN}$(pwd)/${VENV_DIR}${NC}"
echo -e "    pip:     ${CYAN}$(python -m pip --version | awk '{print $2}')${NC}"
echo -e "    Mode:    ${CYAN}${MODE}${NC}"
echo ""
echo -e "  ${BOLD}Next steps:${NC}"
echo -e "    Activate the environment:  ${CYAN}${ACTIVATE_HINT}${NC}"
echo -e "    Show CLI help:             ${CYAN}auralith-pipeline --help${NC}"
echo -e "    Run tests:                 ${CYAN}pytest tests/ -v${NC}"
echo ""
echo -e "  ${BOLD}Quick start:${NC}"
echo -e "    1. Prepare data:       ${CYAN}mkdir -p data/raw/corpus && echo 'Hello world' > data/raw/corpus/sample.txt${NC}"
echo -e "    2. Train tokenizers:   ${CYAN}auralith-pipeline train-tokenizer text --corpus data/raw/corpus/ --output tokenizers/text${NC}"
echo -e "    3. Process shards:     ${CYAN}auralith-pipeline process --input data/raw/ --output shards/ --tokenizers tokenizers/${NC}"
echo ""
