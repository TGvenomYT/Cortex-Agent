#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Cortex — Full Installer
# Sets up: venv, deps, DB schema, launchd daemon, collector schedules, CLI
# ─────────────────────────────────────────────────────────────

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python3"
PLIST_DIR="$HOME/Library/LaunchAgents"

echo ""
echo -e "${BOLD}Cortex — The Control Layer${RESET}"
echo -e "${DIM}Full install${RESET}"
echo ""

# ─── Python check ───
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}✗ python3 not found. Install via: brew install python3${RESET}"
    exit 1
fi
echo -e "  ${GREEN}✓${RESET} python3 $(python3 --version 2>&1 | cut -d' ' -f2)"

# ─── Venv ───
if [[ ! -d "$VENV_DIR" ]]; then
    echo -e "${CYAN}Creating virtual environment...${RESET}"
    python3 -m venv "$VENV_DIR"
fi
echo -e "  ${GREEN}✓${RESET} venv at .venv/"

# ─── Dependencies ───
echo -e "${CYAN}Installing Python packages...${RESET}"
"$VENV_DIR/bin/pip" install -q -r "$PROJECT_DIR/requirements.txt"
echo -e "  ${GREEN}✓${RESET} Dependencies installed"

# ─── Logs directory ───
mkdir -p "$PROJECT_DIR/logs"

# ─── Environment check ───
if [[ -z "${CORTEX_DB_PASSWORD:-}" ]]; then
    echo -e "\n  ${YELLOW}△${RESET} Set CORTEX_DB_PASSWORD in your ~/.zshrc:"
    echo -e "    ${DIM}export CORTEX_DB_PASSWORD=\"your-password\"${RESET}"
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo -e "  ${YELLOW}△${RESET} Set OPENAI_API_KEY in your ~/.zshrc:"
    echo -e "    ${DIM}export OPENAI_API_KEY=\"sk-...\"${RESET}"
fi

# ─── DB Schema ───
echo ""
echo -e "${CYAN}Database schema:${RESET}"
if [[ -n "${CORTEX_DB_PASSWORD:-}" ]]; then
    echo -e "  Attempting to apply schema..."
    PGPASSWORD="$CORTEX_DB_PASSWORD" "$VENV_PYTHON" test_connection.py 2>/dev/null && \
        echo -e "  ${GREEN}✓${RESET} Schema applied" || \
        echo -e "  ${YELLOW}△${RESET} Run manually: CORTEX_DB_PASSWORD=... python3 test_connection.py"
else
    echo -e "  ${DIM}Run: CORTEX_DB_PASSWORD=... python3 test_connection.py${RESET}"
fi

# ─── Daemon launchd ───
echo ""
echo -e "${CYAN}Installing Cortex daemon...${RESET}"
mkdir -p "$PLIST_DIR"
PLIST_FILE="$PLIST_DIR/com.cortex.daemon.plist"
cp "$PROJECT_DIR/com.cortex.daemon.plist" "$PLIST_FILE"
launchctl unload "$PLIST_FILE" 2>/dev/null || true
launchctl load "$PLIST_FILE"
echo -e "  ${GREEN}✓${RESET} Daemon installed (auto-starts on login)"
echo -e "  ${DIM}Control: launchctl unload/load $PLIST_FILE${RESET}"

# ─── Collector schedules ───
echo ""
echo -e "${CYAN}Installing collector schedules...${RESET}"
bash "$PROJECT_DIR/collectors/install_schedules.sh"

# ─── CLI tool ───
echo ""
echo -e "${CYAN}Installing 'cx' CLI...${RESET}"
chmod +x "$PROJECT_DIR/cx"
if [[ -d /usr/local/bin ]]; then
    sudo ln -sf "$PROJECT_DIR/cx" /usr/local/bin/cx 2>/dev/null && \
        echo -e "  ${GREEN}✓${RESET} 'cx' available system-wide" || \
        echo -e "  ${YELLOW}△${RESET} Run: sudo ln -sf \"$PROJECT_DIR/cx\" /usr/local/bin/cx"
else
    echo -e "  ${YELLOW}△${RESET} Add to PATH or symlink: ln -sf \"$PROJECT_DIR/cx\" /usr/local/bin/cx"
fi

# ─── Done ───
echo ""
echo -e "${GREEN}${BOLD}✓ Cortex fully installed${RESET}"
echo ""
echo -e "  Quick start:"
echo -e "    ${BOLD}cx status${RESET}         — check if daemon is running"
echo -e "    ${BOLD}cx hello${RESET}          — talk to Cortex"
echo -e "    ${BOLD}cx memory${RESET}         — see stored memories"
echo -e "    ${BOLD}cx remind ...${RESET}     — set a reminder"
echo -e "    ${BOLD}cx devops ...${RESET}     — plan a multi-step task"
echo ""
