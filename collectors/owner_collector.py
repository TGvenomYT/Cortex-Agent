"""
Owner Collector — Complete machine + owner profile.
Runs every 30 minutes (mostly slow-changing data).
Captures EVERYTHING about the machine and its owner so Cortex has full context:
hardware, installed software, dev environment, accounts, login items, network history,
connected devices, preferences, scheduled jobs.

Excludes (by design, needs explicit opt-in): browser history/bookmarks, Messages, Mail, Contacts.
"""

import os
import sys
import re
import json
import logging
import plistlib

sys.path.insert(0, os.path.dirname(__file__))
from base import load_config, get_db_conn, upsert_context, run_cmd

logger = logging.getLogger("collector.owner")


# ─── Hardware (complete system_profiler dumps) ───

def collect_hardware(conn):
    """Full hardware profile: model, chip, memory, serial, storage, graphics, displays, power/battery."""
    hw_out, _, _ = run_cmd("system_profiler SPHardwareDataType", timeout=20)
    hw = _parse_profiler_block(hw_out)

    storage_out, _, _ = run_cmd("system_profiler SPStorageDataType", timeout=20)
    volumes = _parse_profiler_volumes(storage_out)

    graphics_out, _, _ = run_cmd("system_profiler SPDisplaysDataType", timeout=20)
    graphics = _parse_profiler_block(graphics_out)

    power_out, _, _ = run_cmd("system_profiler SPPowerDataType", timeout=20)
    power = _parse_profiler_block(power_out)

    battery_out, _, _ = run_cmd("pmset -g batt", timeout=10)

    embed_text = f"Hardware: {hw.get('Model Name','')} {hw.get('Chip','')} {hw.get('Memory','')} " \
                 f"serial {hw.get('Serial Number (system)','')}, {len(volumes)} storage volumes"
    upsert_context(conn, "owner", "hardware", {
        "summary": hw,
        "volumes": volumes,
        "graphics": graphics,
        "power": power,
        "battery_raw": battery_out,
    }, expires_minutes=120, embedding_text=embed_text)
    logger.info(f"Hardware: {hw.get('Model Name', '?')}, {len(volumes)} volumes")


def collect_usb_bluetooth(conn):
    """All connected USB and Bluetooth devices (full list, no caps)."""
    usb_out, _, _ = run_cmd("system_profiler SPUSBDataType", timeout=20)
    usb_devices = _parse_profiler_devices(usb_out)

    bt_out, _, _ = run_cmd("system_profiler SPBluetoothDataType", timeout=20)
    bt_devices = _parse_profiler_devices(bt_out)

    thunderbolt_out, _, _ = run_cmd("system_profiler SPThunderboltDataType", timeout=15)
    tb_devices = _parse_profiler_devices(thunderbolt_out)

    embed_text = "Connected devices: USB " + ", ".join(d.get("name", "") for d in usb_devices) + \
                 "; Bluetooth " + ", ".join(d.get("name", "") for d in bt_devices) + \
                 "; Thunderbolt " + ", ".join(d.get("name", "") for d in tb_devices)
    upsert_context(conn, "owner", "connected_devices", {
        "usb": usb_devices,
        "bluetooth": bt_devices,
        "thunderbolt": tb_devices,
    }, expires_minutes=60, embedding_text=embed_text)
    logger.info(f"Devices: {len(usb_devices)} USB, {len(bt_devices)} Bluetooth, {len(tb_devices)} Thunderbolt")


def collect_displays(conn):
    """All connected displays with full specs."""
    out, _, _ = run_cmd("system_profiler SPDisplaysDataType", timeout=15)
    displays = _parse_profiler_devices(out)
    embed_text = "Displays connected: " + ", ".join(d.get("name", "") for d in displays)
    upsert_context(conn, "owner", "displays", {"displays": displays}, expires_minutes=120, embedding_text=embed_text)
    logger.info(f"Displays: {len(displays)} connected")


# ─── Complete software inventory ───

def collect_installed_applications(conn):
    """Every application in /Applications, /System/Applications, and ~/Applications — full list."""
    apps = []
    search_dirs = [
        "/Applications",
        "/System/Applications",
        os.path.expanduser("~/Applications"),
        "/System/Applications/Utilities",
    ]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        try:
            for entry in os.listdir(d):
                if entry.endswith(".app"):
                    app_path = os.path.join(d, entry)
                    version = _get_app_version(app_path)
                    apps.append({
                        "name": entry.replace(".app", ""),
                        "path": app_path,
                        "version": version,
                    })
        except Exception as e:
            logger.debug(f"Skipping {d}: {e}")

    app_names = sorted(a["name"] for a in apps)
    embed_text = f"Installed applications ({len(apps)} total): " + ", ".join(app_names[:150])
    upsert_context(conn, "owner", "installed_applications", {
        "count": len(apps),
        "applications": sorted(apps, key=lambda a: a["name"].lower()),
    }, expires_minutes=60, embedding_text=embed_text)
    logger.info(f"Applications: {len(apps)} installed")


def collect_homebrew_packages(conn):
    """Every Homebrew formula and cask installed, with versions."""
    formulae_out, _, _ = run_cmd("brew list --formula --versions 2>/dev/null", timeout=15)
    casks_out, _, _ = run_cmd("brew list --cask --versions 2>/dev/null", timeout=15)
    leaves_out, _, _ = run_cmd("brew leaves 2>/dev/null", timeout=15)

    formulae = []
    for line in formulae_out.splitlines():
        parts = line.split()
        if parts:
            formulae.append({"name": parts[0], "versions": parts[1:]})

    casks = []
    for line in casks_out.splitlines():
        parts = line.split()
        if parts:
            casks.append({"name": parts[0], "versions": parts[1:]})

    leaves = [l.strip() for l in leaves_out.splitlines() if l.strip()]

    embed_text = f"Homebrew packages installed: " + ", ".join(leaves[:100])
    upsert_context(conn, "owner", "homebrew", {
        "formula_count": len(formulae),
        "cask_count": len(casks),
        "formulae": formulae,
        "casks": casks,
        "leaves": leaves,  # explicitly-installed (not deps) — most relevant list
    }, expires_minutes=60, embedding_text=embed_text)
    logger.info(f"Homebrew: {len(formulae)} formulae, {len(casks)} casks")


def collect_dev_environment(conn):
    """Every language runtime, version manager, and dev tool — complete versions."""
    tools = {}
    checks = {
        "python3": "python3 --version",
        "python": "python --version",
        "node": "node --version",
        "npm": "npm --version",
        "yarn": "yarn --version",
        "pnpm": "pnpm --version",
        "ruby": "ruby --version",
        "go": "go version",
        "rustc": "rustc --version",
        "cargo": "cargo --version",
        "java": "java -version",
        "docker": "docker --version",
        "docker-compose": "docker compose version",
        "git": "git --version",
        "gh": "gh --version",
        "psql": "psql --version",
        "redis-cli": "redis-cli --version",
        "aws": "aws --version",
        "kubectl": "kubectl version --client",
        "terraform": "terraform --version",
        "brew": "brew --version",
        "code": "code --version",
        "zsh": "zsh --version",
        "bash": "bash --version",
        "ffmpeg": "ffmpeg -version",
    }
    for tool, cmd in checks.items():
        out, err, rc = run_cmd(cmd, timeout=8)
        if rc != 0 or "command not found" in err or "command not found" in out:
            tools[tool] = None
        else:
            result = (out or err).splitlines()
            tools[tool] = result[0].strip() if result else None

    # Version managers — list ALL installed versions, not just active
    pyenv_versions, _, _ = run_cmd("pyenv versions 2>/dev/null", timeout=8)
    nvm_versions, _, _ = run_cmd("bash -lc 'nvm ls 2>/dev/null'", timeout=8)
    rbenv_versions, _, _ = run_cmd("rbenv versions 2>/dev/null", timeout=8)

    # Global git identity + full config
    git_config_out, _, _ = run_cmd("git config --global --list", timeout=8)
    git_config = {}
    for line in git_config_out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            git_config[k] = v

    # Global pip packages (complete list)
    pip_list, _, _ = run_cmd("pip3 list --format=freeze 2>/dev/null", timeout=15)
    pip_packages = [l.strip() for l in pip_list.splitlines() if l.strip()]

    # Global npm packages (complete list)
    npm_list, _, _ = run_cmd("npm list -g --depth=0 --json 2>/dev/null", timeout=15)
    npm_packages = []
    try:
        npm_data = json.loads(npm_list)
        npm_packages = list(npm_data.get("dependencies", {}).keys())
    except Exception:
        pass

    installed_tools = [f"{k} ({v})" for k, v in tools.items() if v]
    embed_text = "Dev environment tools: " + ", ".join(installed_tools) + \
                 f"; git user {git_config.get('user.name','')} <{git_config.get('user.email','')}>"
    upsert_context(conn, "owner", "dev_environment", {
        "tool_versions": tools,
        "pyenv_versions": pyenv_versions.splitlines() if pyenv_versions else [],
        "nvm_versions": nvm_versions.splitlines() if nvm_versions else [],
        "rbenv_versions": rbenv_versions.splitlines() if rbenv_versions else [],
        "git_global_config": git_config,
        "pip_global_packages": pip_packages,
        "npm_global_packages": npm_packages,
    }, expires_minutes=60, embedding_text=embed_text)
    logger.info(f"Dev environment: {len(tools)} tools checked, {len(pip_packages)} pip pkgs, {len(npm_packages)} npm pkgs")


def collect_ssh_keys(conn):
    """SSH key inventory — fingerprints and comments only, NEVER private key material."""
    ssh_dir = os.path.expanduser("~/.ssh")
    keys = []
    if os.path.isdir(ssh_dir):
        for entry in os.listdir(ssh_dir):
            if entry.endswith(".pub"):
                pub_path = os.path.join(ssh_dir, entry)
                fingerprint, _, _ = run_cmd(f"ssh-keygen -lf '{pub_path}' 2>/dev/null", timeout=5)
                try:
                    with open(pub_path) as f:
                        content = f.read().strip()
                    comment = content.split()[-1] if content else ""
                except Exception:
                    comment = ""
                keys.append({
                    "public_key_file": entry,
                    "fingerprint": fingerprint.strip(),
                    "comment": comment,
                })

    # SSH config hosts (host aliases only, no credentials)
    ssh_config_path = os.path.join(ssh_dir, "config")
    hosts = []
    if os.path.exists(ssh_config_path):
        try:
            with open(ssh_config_path) as f:
                for line in f:
                    m = re.match(r"^\s*Host\s+(.+)$", line, re.IGNORECASE)
                    if m:
                        hosts.append(m.group(1).strip())
        except Exception:
            pass

    embed_text = f"SSH keys: {len(keys)} public keys; configured hosts: " + ", ".join(hosts)
    upsert_context(conn, "owner", "ssh_keys", {
        "keys": keys,
        "configured_hosts": hosts,
    }, expires_minutes=120, embedding_text=embed_text)
    logger.info(f"SSH: {len(keys)} public keys, {len(hosts)} configured hosts")


# ─── Account, locale, preferences ───

def collect_user_account(conn):
    """Full user account info: name, UID, groups, shell, home, locale, timezone, computer name."""
    full_name, _, _ = run_cmd("id -F 2>/dev/null || dscl . -read /Users/$(whoami) RealName 2>/dev/null | sed 's/RealName: //'")
    username, _, _ = run_cmd("whoami")
    uid, _, _ = run_cmd("id -u")
    groups, _, _ = run_cmd("id -Gn")
    shell, _, _ = run_cmd("echo $SHELL")
    home, _, _ = run_cmd("echo $HOME")
    computer_name, _, _ = run_cmd("scutil --get ComputerName")
    local_hostname, _, _ = run_cmd("scutil --get LocalHostName")
    timezone, _, _ = run_cmd("systemsetup -gettimezone 2>/dev/null | sed 's/Time Zone: //'")
    locale, _, _ = run_cmd("defaults read -g AppleLocale 2>/dev/null")
    languages, _, _ = run_cmd("defaults read -g AppleLanguages 2>/dev/null")

    embed_text = f"Owner account: {full_name.strip()} ({username.strip()}) on computer {computer_name.strip()}, " \
                 f"timezone {timezone.strip()}, shell {shell.strip()}"
    upsert_context(conn, "owner", "account", {
        "full_name": full_name.strip(),
        "username": username.strip(),
        "uid": uid.strip(),
        "groups": groups.strip().split(),
        "shell": shell.strip(),
        "home": home.strip(),
        "computer_name": computer_name.strip(),
        "local_hostname": local_hostname.strip(),
        "timezone": timezone.strip(),
        "locale": locale.strip(),
        "languages_raw": languages.strip(),
    }, expires_minutes=240, embedding_text=embed_text)
    logger.info(f"Account: {full_name.strip()} ({username.strip()}) on {computer_name.strip()}")


def collect_preferences(conn):
    """System preferences: default browser, dock apps, finder prefs, dark mode, keyboard/trackpad settings."""
    default_browser, _, _ = run_cmd(
        "defaults read com.apple.LaunchServices/com.apple.launchservices.secure "
        "LSHandlers 2>/dev/null | grep -B2 'http;' | grep LSHandlerRoleAll | head -1"
    )
    dark_mode, _, _ = run_cmd("defaults read -g AppleInterfaceStyle 2>/dev/null")

    # Dock persistent apps (full list)
    dock_out, _, _ = run_cmd(
        "defaults read com.apple.dock persistent-apps 2>/dev/null | grep 'file-label' | sed 's/.*= //;s/;//'"
    )
    dock_apps = [l.strip().strip('"') for l in dock_out.splitlines() if l.strip()]

    finder_show_extensions, _, _ = run_cmd("defaults read NSGlobalDomain AppleShowAllExtensions 2>/dev/null")
    finder_show_hidden, _, _ = run_cmd("defaults read com.apple.finder AppleShowAllFiles 2>/dev/null")
    key_repeat, _, _ = run_cmd("defaults read -g KeyRepeat 2>/dev/null")
    natural_scroll, _, _ = run_cmd("defaults read -g com.apple.swipescrolldirection 2>/dev/null")
    screensaver_time, _, _ = run_cmd("defaults -currentHost read com.apple.screensaver idleTime 2>/dev/null")

    embed_text = f"System preferences: dark mode {bool(dark_mode.strip())}, dock apps: " + ", ".join(dock_apps)
    upsert_context(conn, "owner", "preferences", {
        "dark_mode": bool(dark_mode.strip()),
        "dock_apps": dock_apps,
        "finder_show_all_extensions": finder_show_extensions.strip(),
        "finder_show_hidden_files": finder_show_hidden.strip(),
        "key_repeat_rate": key_repeat.strip(),
        "natural_scroll": natural_scroll.strip(),
        "screensaver_idle_seconds": screensaver_time.strip(),
    }, expires_minutes=120, embedding_text=embed_text)
    logger.info(f"Preferences: dark_mode={bool(dark_mode.strip())}, {len(dock_apps)} dock apps")


# ─── Login items, launch agents/daemons, scheduled jobs ───

def collect_startup_items(conn):
    """Every login item, LaunchAgent, and LaunchDaemon (user + system) — complete list."""
    # Modern login items (macOS 13+)
    login_items_out, _, _ = run_cmd(
        "osascript -e 'tell application \"System Events\" to get the name of every login item' 2>/dev/null"
    )
    login_items = [i.strip() for i in login_items_out.split(",")] if login_items_out else []

    # LaunchAgents — user and system, full list of labels
    user_agents = _list_plist_labels(os.path.expanduser("~/Library/LaunchAgents"))
    system_agents = _list_plist_labels("/Library/LaunchAgents")
    system_daemons = _list_plist_labels("/Library/LaunchDaemons")

    # Currently loaded launchd jobs (full list via launchctl)
    launchctl_out, _, _ = run_cmd("launchctl list", timeout=10)
    loaded_jobs = []
    for line in launchctl_out.splitlines()[1:]:  # skip header
        parts = line.split(None, 2)
        if len(parts) >= 3:
            loaded_jobs.append({"pid": parts[0], "status": parts[1], "label": parts[2]})

    embed_text = "Startup items and login items: " + ", ".join(login_items) + \
                 "; LaunchAgents: " + ", ".join(a["label"] for a in (user_agents + system_agents)[:60])
    upsert_context(conn, "owner", "startup_items", {
        "login_items": login_items,
        "user_launch_agents": user_agents,
        "system_launch_agents": system_agents,
        "system_launch_daemons": system_daemons,
        "currently_loaded": loaded_jobs,
        "loaded_count": len(loaded_jobs),
    }, expires_minutes=60, embedding_text=embed_text)
    logger.info(f"Startup: {len(login_items)} login items, {len(loaded_jobs)} loaded launchd jobs")


def collect_scheduled_jobs(conn):
    """Full crontab (user) — complete, unmodified."""
    crontab_out, _, rc = run_cmd("crontab -l 2>/dev/null")
    cron_jobs = [l for l in crontab_out.splitlines() if l.strip() and not l.strip().startswith("#")] if rc == 0 else []

    embed_text = f"Scheduled cron jobs ({len(cron_jobs)}): " + "; ".join(cron_jobs)
    upsert_context(conn, "owner", "scheduled_jobs", {
        "crontab": cron_jobs,
        "count": len(cron_jobs),
    }, expires_minutes=120, embedding_text=embed_text)
    logger.info(f"Scheduled jobs: {len(cron_jobs)} crontab entries")


# ─── Network history (WiFi + known networks, no traffic content) ───

def collect_network_profile(conn):
    """All known/remembered WiFi networks and current network details — complete list."""
    known_networks_out, _, _ = run_cmd(
        "networksetup -listpreferredwirelessnetworks en0 2>/dev/null"
    )
    known_networks = [l.strip() for l in known_networks_out.splitlines()[1:] if l.strip()]

    current_wifi, _, _ = run_cmd("networksetup -getairportnetwork en0 2>/dev/null")

    # All network services configured
    services_out, _, _ = run_cmd("networksetup -listallnetworkservices 2>/dev/null")
    services = [l.strip() for l in services_out.splitlines()[1:] if l.strip() and not l.startswith("*")]

    # DNS servers configured
    dns_out, _, _ = run_cmd("networksetup -getdnsservers en0 2>/dev/null")

    # VPN / network interfaces hardware
    hw_ports_out, _, _ = run_cmd("networksetup -listallhardwareports 2>/dev/null")

    embed_text = f"Network: current WiFi {current_wifi.strip()}; known networks: " + ", ".join(known_networks)
    upsert_context(conn, "owner", "network_profile", {
        "known_wifi_networks": known_networks,
        "known_wifi_count": len(known_networks),
        "current_wifi": current_wifi.strip(),
        "network_services": services,
        "dns_servers": dns_out.strip(),
        "hardware_ports_raw": hw_ports_out.strip(),
    }, expires_minutes=60, embedding_text=embed_text)
    logger.info(f"Network profile: {len(known_networks)} known WiFi networks, current={current_wifi.strip()}")


# ─── Fonts, time machine, security ───

def collect_security_settings(conn):
    """FileVault, firewall, Gatekeeper, SIP status, Touch ID/Face ID enrollment count."""
    filevault, _, _ = run_cmd("fdesetup status 2>/dev/null")
    firewall, _, _ = run_cmd("defaults read /Library/Preferences/com.apple.alf globalstate 2>/dev/null")
    gatekeeper, _, _ = run_cmd("spctl --status 2>/dev/null")
    sip_status, _, _ = run_cmd("csrutil status 2>/dev/null")
    time_machine, _, _ = run_cmd("tmutil status 2>/dev/null")

    embed_text = f"Security settings: FileVault {filevault.strip()}, Gatekeeper {gatekeeper.strip()}, SIP {sip_status.strip()}"
    upsert_context(conn, "owner", "security", {
        "filevault": filevault.strip(),
        "firewall_state": firewall.strip(),
        "gatekeeper": gatekeeper.strip(),
        "sip_status": sip_status.strip(),
        "time_machine_status": time_machine.strip(),
    }, expires_minutes=120, embedding_text=embed_text)
    logger.info(f"Security: filevault={filevault.strip()}, gatekeeper={gatekeeper.strip()}")


def collect_time_machine_backups(conn):
    """Time Machine backup destinations and snapshot history — full list."""
    destinations_out, _, _ = run_cmd("tmutil destinationinfo 2>/dev/null")
    snapshots_out, _, _ = run_cmd("tmutil listlocalsnapshots / 2>/dev/null")
    snapshots = [l.strip() for l in snapshots_out.splitlines() if l.strip()]

    embed_text = f"Time Machine backups: {len(snapshots)} local snapshots, destinations: {destinations_out.strip()[:200]}"
    upsert_context(conn, "owner", "backups", {
        "destinations_raw": destinations_out.strip(),
        "local_snapshots": snapshots,
        "snapshot_count": len(snapshots),
    }, expires_minutes=120, embedding_text=embed_text)
    logger.info(f"Backups: {len(snapshots)} local snapshots")


# ─── Parsing helpers ───

def _parse_profiler_block(output: str) -> dict:
    """Parse a flat system_profiler text block into a dict (first-level keys only)."""
    result = {}
    for line in output.splitlines():
        line = line.strip()
        if ":" in line and not line.endswith(":"):
            k, v = line.split(":", 1)
            result[k.strip()] = v.strip()
    return result


def _parse_profiler_devices(output: str) -> list[dict]:
    """
    Parse system_profiler output with nested device entries into a list of dicts.
    Each top-level indented entry (device name line ending in ':') becomes a device
    with its child key:value lines as properties.
    """
    devices = []
    current = None
    for raw_line in output.splitlines():
        if not raw_line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        line = raw_line.strip()

        if line.endswith(":") and ":" not in line[:-1]:
            # This is a device name header
            if current:
                devices.append(current)
            current = {"name": line[:-1].strip(), "indent": indent}
        elif ":" in line and current is not None:
            k, v = line.split(":", 1)
            if v.strip():
                current[k.strip()] = v.strip()

    if current:
        devices.append(current)

    # Drop the pure category headers (no properties beyond name/indent)
    return [d for d in devices if len(d) > 2]


def _parse_profiler_volumes(output: str) -> list[dict]:
    """Parse SPStorageDataType into a list of volume dicts."""
    volumes = []
    current = None
    for raw_line in output.splitlines():
        if not raw_line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        line = raw_line.strip()

        if line.endswith(":") and indent <= 4 and ":" not in line[:-1]:
            if current:
                volumes.append(current)
            current = {"name": line[:-1].strip()}
        elif ":" in line and current is not None:
            k, v = line.split(":", 1)
            if v.strip():
                current[k.strip()] = v.strip()

    if current:
        volumes.append(current)
    return volumes


def _get_app_version(app_path: str) -> str:
    """Read CFBundleShortVersionString from an app's Info.plist."""
    plist_path = os.path.join(app_path, "Contents", "Info.plist")
    try:
        with open(plist_path, "rb") as f:
            plist = plistlib.load(f)
        return plist.get("CFBundleShortVersionString", "")
    except Exception:
        return ""


def _list_plist_labels(directory: str) -> list[dict]:
    """List every .plist in a LaunchAgents/LaunchDaemons dir with its Label."""
    items = []
    if not os.path.isdir(directory):
        return items
    try:
        for entry in os.listdir(directory):
            if not entry.endswith(".plist"):
                continue
            path = os.path.join(directory, entry)
            label = entry.replace(".plist", "")
            try:
                with open(path, "rb") as f:
                    plist = plistlib.load(f)
                label = plist.get("Label", label)
            except Exception:
                pass
            items.append({"file": entry, "label": label})
    except Exception as e:
        logger.debug(f"Cannot list {directory}: {e}")
    return items


def main():
    config = load_config()
    conn = get_db_conn(config)

    collect_hardware(conn)
    collect_usb_bluetooth(conn)
    collect_displays(conn)
    collect_installed_applications(conn)
    collect_homebrew_packages(conn)
    collect_dev_environment(conn)
    collect_ssh_keys(conn)
    collect_user_account(conn)
    collect_preferences(conn)
    collect_startup_items(conn)
    collect_scheduled_jobs(conn)
    collect_network_profile(conn)
    collect_security_settings(conn)
    collect_time_machine_backups(conn)

    conn.close()
    logger.info("Owner collection done — complete machine profile stored")


if __name__ == "__main__":
    main()
