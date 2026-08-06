#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Cortex Collectors — Install launchd schedules
# Creates LaunchAgents for each collector with proper intervals.
# ─────────────────────────────────────────────────────────────

set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
DIM='\033[2m'
RESET='\033[0m'

PLIST_DIR="$HOME/Library/LaunchAgents"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python3"
COLLECTORS_DIR="$PROJECT_DIR/collectors"

mkdir -p "$PLIST_DIR"

echo ""
echo -e "${BOLD}Cortex — Installing collector schedules${RESET}"
echo ""

# Check venv exists
if [[ ! -f "$VENV_PYTHON" ]]; then
    echo "ERROR: venv not found at $VENV_PYTHON"
    exit 1
fi

# Define collectors: name, script, interval_seconds
declare -a COLLECTORS=(
    "shell:shell_collector.py:60"
    "process:process_collector.py:60"
    "git:git_collector.py:120"
    "system:system_collector.py:300"
    "project:project_collector.py:1800"
    "metrics:metrics_collector.py:60"
)

for entry in "${COLLECTORS[@]}"; do
    IFS=':' read -r name script interval <<< "$entry"
    label="com.cortex.collector.${name}"
    plist="$PLIST_DIR/${label}.plist"

    cat > "$plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${VENV_PYTHON}</string>
        <string>${COLLECTORS_DIR}/${script}</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>CORTEX_DB_PASSWORD</key>
        <string>${CORTEX_DB_PASSWORD:-}</string>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
    <key>WorkingDirectory</key>
    <string>${PROJECT_DIR}</string>
    <key>StartInterval</key>
    <integer>${interval}</integer>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${PROJECT_DIR}/logs/collector-${name}.log</string>
    <key>StandardErrorPath</key>
    <string>${PROJECT_DIR}/logs/collector-${name}-error.log</string>
</dict>
</plist>
EOF

    # Unload if already loaded, then load
    launchctl unload "$plist" 2>/dev/null || true
    launchctl load "$plist"

    echo -e "  ${GREEN}✓${RESET} ${name} — every ${interval}s → ${plist}"
done

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"

echo ""
echo -e "${BOLD}All collectors scheduled.${RESET}"
echo -e "${DIM}Logs: $PROJECT_DIR/logs/${RESET}"
echo ""
echo -e "  Manage:"
echo -e "    List:    launchctl list | grep cortex"
echo -e "    Stop:    launchctl unload ~/Library/LaunchAgents/com.cortex.collector.*.plist"
echo -e "    Restart: launchctl unload <plist> && launchctl load <plist>"
echo ""
