#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Cortex — install the SCHEDULED (wake-run-exit) run model.
#
# Replaces the always-on KeepAlive Flask daemon (com.cortex.daemon) with
# periodic, battery-gated launchd jobs:
#   autonomy   every 30 min   (heavy; skips on low battery / thermal)
#   reminders  every 60 s      (cheap)
#   automation every 120 s     (cheap)
#   reports    09:00 & 18:00   (heavy; quiet delivery)
#
# Secrets: the DB password lives ONLY in ~/.cortex/cortex.env (chmod 600),
# never in the plists.
# ─────────────────────────────────────────────────────────────
set -euo pipefail

BOLD='\033[1m'; GREEN='\033[0;32m'; DIM='\033[2m'; YELLOW='\033[0;33m'; RESET='\033[0m'

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST_DIR="$HOME/Library/LaunchAgents"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python3"
RUNNER="$PROJECT_DIR/run_once.py"
ENV_FILE="$HOME/.cortex/cortex.env"
OLD_DAEMON="com.cortex.daemon"

mkdir -p "$PLIST_DIR" "$HOME/.cortex/state" "$PROJECT_DIR/logs"
chmod 700 "$HOME/.cortex/state"

echo ""
echo -e "${BOLD}Cortex — installing scheduled run model${RESET}"
echo ""

# ── 1. Secret: migrate DB password into a 0600 env file ──────────────────
if [[ ! -f "$ENV_FILE" ]]; then
    PW=""
    OLD_PLIST="$PLIST_DIR/$OLD_DAEMON.plist"
    if [[ -f "$OLD_PLIST" ]]; then
        PW="$(/usr/libexec/PlistBuddy -c 'Print :EnvironmentVariables:CORTEX_DB_PASSWORD' "$OLD_PLIST" 2>/dev/null || true)"
    fi
    if [[ -z "$PW" && -n "${CORTEX_DB_PASSWORD:-}" ]]; then
        PW="$CORTEX_DB_PASSWORD"
    fi
    if [[ -z "$PW" ]]; then
        read -rsp "Enter CORTEX_DB_PASSWORD (hidden): " PW; echo ""
    fi
    umask 177
    printf 'CORTEX_DB_PASSWORD=%s\n' "$PW" > "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo -e "  ${GREEN}✓${RESET} wrote $ENV_FILE (chmod 600)"
else
    echo -e "  ${DIM}env file already exists: $ENV_FILE${RESET}"
fi

# ── 2. Retire the always-on daemon ───────────────────────────────────────
if launchctl list 2>/dev/null | grep -q "$OLD_DAEMON"; then
    launchctl bootout "gui/$(id -u)/$OLD_DAEMON" 2>/dev/null \
        || launchctl unload "$PLIST_DIR/$OLD_DAEMON.plist" 2>/dev/null || true
    echo -e "  ${YELLOW}✓${RESET} retired always-on daemon ($OLD_DAEMON)"
else
    echo -e "  ${DIM}always-on daemon not loaded (nothing to retire)${RESET}"
fi

# ── 3. Helpers to emit plists ────────────────────────────────────────────
emit_interval_plist() {  # name  interval_seconds  arg1 [arg2]
    local name="$1" interval="$2"; shift 2
    local label="com.cortex.$name"
    local plist="$PLIST_DIR/$label.plist"
    local args_xml=""
    for a in "$@"; do args_xml+="        <string>$a</string>\n"; done
    cat > "$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$label</string>
    <key>ProgramArguments</key>
    <array>
        <string>$VENV_PYTHON</string>
        <string>$RUNNER</string>
$(printf "$args_xml")    </array>
    <key>WorkingDirectory</key><string>$PROJECT_DIR</string>
    <key>StartInterval</key><integer>$interval</integer>
    <key>ProcessType</key><string>Background</string>
    <key>LowPriorityIO</key><true/>
    <key>Nice</key><integer>5</integer>
    <key>StandardOutPath</key><string>$PROJECT_DIR/logs/sched-$name.log</string>
    <key>StandardErrorPath</key><string>$PROJECT_DIR/logs/sched-$name-error.log</string>
</dict>
</plist>
EOF
    launchctl unload "$plist" 2>/dev/null || true
    launchctl load "$plist"
    echo -e "  ${GREEN}✓${RESET} $name — every ${interval}s"
}

emit_calendar_plist() {  # name  hour  minute  arg1 arg2
    local name="$1" hour="$2" minute="$3"; shift 3
    local label="com.cortex.$name"
    local plist="$PLIST_DIR/$label.plist"
    local args_xml=""
    for a in "$@"; do args_xml+="        <string>$a</string>\n"; done
    cat > "$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$label</string>
    <key>ProgramArguments</key>
    <array>
        <string>$VENV_PYTHON</string>
        <string>$RUNNER</string>
$(printf "$args_xml")    </array>
    <key>WorkingDirectory</key><string>$PROJECT_DIR</string>
    <key>StartCalendarInterval</key>
    <dict><key>Hour</key><integer>$hour</integer><key>Minute</key><integer>$minute</integer></dict>
    <key>ProcessType</key><string>Background</string>
    <key>LowPriorityIO</key><true/>
    <key>StandardOutPath</key><string>$PROJECT_DIR/logs/sched-$name.log</string>
    <key>StandardErrorPath</key><string>$PROJECT_DIR/logs/sched-$name-error.log</string>
</dict>
</plist>
EOF
    launchctl unload "$plist" 2>/dev/null || true
    launchctl load "$plist"
    echo -e "  ${GREEN}✓${RESET} $name — daily at $(printf '%02d:%02d' "$hour" "$minute")"
}

# ── 4. Install the jobs ──────────────────────────────────────────────────
emit_interval_plist autonomy   1800 autonomy
emit_interval_plist reminders    60 reminders
emit_interval_plist automation  120 automation
emit_interval_plist sync       1800 sync
emit_calendar_plist report-morning  9  0 report "Morning Briefing"
emit_calendar_plist report-evening 18  0 report "Evening Wrap"

echo ""
echo -e "${BOLD}Scheduled run model installed.${RESET}"
echo -e "${DIM}  List:   launchctl list | grep cortex${RESET}"
echo -e "${DIM}  Logs:   $PROJECT_DIR/logs/sched-*.log${RESET}"
echo -e "${DIM}  Remove: ~/Library/LaunchAgents/com.cortex.{autonomy,reminders,automation,report-*}.plist${RESET}"
echo ""
