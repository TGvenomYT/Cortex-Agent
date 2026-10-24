"""
Calendar Integration — Reads macOS Calendar events via AppleScript.
Gives Cortex awareness of your schedule for time-based context.
"""

import time
import hashlib
import subprocess
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("cortex.calendar")

# AppleScript over all calendars is slow (seconds) and was blocking EVERY LLM
# turn via build_messages. Cache results briefly and keep the timeout tight so
# calendar awareness never dominates response latency.
_CAL_CACHE: dict[str, tuple[float, list]] = {}
_CAL_TTL = 300.0       # seconds (successful reads)
_CAL_MISS_TTL = 1800.0 # seconds — a denied/unavailable calendar won't fix itself soon,
                       # so cache the empty result long so we don't re-block every turn
_CAL_TIMEOUT = 2.5     # seconds


def get_todays_events() -> list[dict]:
    """Get all events for today from macOS Calendar."""
    script = '''
    set today to current date
    set hours of today to 0
    set minutes of today to 0
    set seconds of today to 0
    set tomorrow to today + 1 * days
    set output to ""
    tell application "Calendar"
        repeat with cal in calendars
            set calName to name of cal
            set evts to (every event of cal whose start date >= today and start date < tomorrow)
            repeat with evt in evts
                set evtName to summary of evt
                set evtStart to start date of evt
                set evtEnd to end date of evt
                set evtLoc to location of evt
                if evtLoc is missing value then set evtLoc to ""
                set output to output & evtName & "|" & (evtStart as string) & "|" & (evtEnd as string) & "|" & calName & "|" & evtLoc & linefeed
            end repeat
        end repeat
    end tell
    return output
    '''
    return _run_calendar_script(script)


def get_upcoming_events(hours: int = 4) -> list[dict]:
    """Get events in the next N hours."""
    script = f'''
    set now to current date
    set later to now + {hours * 3600}
    set output to ""
    tell application "Calendar"
        repeat with cal in calendars
            set calName to name of cal
            set evts to (every event of cal whose start date >= now and start date <= later)
            repeat with evt in evts
                set evtName to summary of evt
                set evtStart to start date of evt
                set evtEnd to end date of evt
                set evtLoc to location of evt
                if evtLoc is missing value then set evtLoc to ""
                set output to output & evtName & "|" & (evtStart as string) & "|" & (evtEnd as string) & "|" & calName & "|" & evtLoc & linefeed
            end repeat
        end repeat
    end tell
    return output
    '''
    return _run_calendar_script(script)


def get_tomorrows_events() -> list[dict]:
    """Get events for tomorrow."""
    script = '''
    set today to current date
    set hours of today to 0
    set minutes of today to 0
    set seconds of today to 0
    set tomorrow to today + 1 * days
    set dayAfter to today + 2 * days
    set output to ""
    tell application "Calendar"
        repeat with cal in calendars
            set calName to name of cal
            set evts to (every event of cal whose start date >= tomorrow and start date < dayAfter)
            repeat with evt in evts
                set evtName to summary of evt
                set evtStart to start date of evt
                set evtEnd to end date of evt
                set evtLoc to location of evt
                if evtLoc is missing value then set evtLoc to ""
                set output to output & evtName & "|" & (evtStart as string) & "|" & (evtEnd as string) & "|" & calName & "|" & evtLoc & linefeed
            end repeat
        end repeat
    end tell
    return output
    '''
    return _run_calendar_script(script)


def _run_calendar_script(script: str) -> list[dict]:
    """Run AppleScript and parse the output (cached, tight timeout).

    macOS Calendar only — on other OSes there's no osascript, so skip without a
    pointless subprocess. A timeout/failure is cached for a long time so a denied
    or unavailable calendar never blocks more than one turn per 30 minutes."""
    import cortex.util.osutil as osutil
    if not osutil.IS_MAC:
        return []
    key = hashlib.md5(script.encode()).hexdigest()
    hit = _CAL_CACHE.get(key)
    if hit:
        ttl = hit[2] if len(hit) > 2 else _CAL_TTL
        if (time.time() - hit[0]) < ttl:
            return hit[1]
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=_CAL_TIMEOUT
        )
        if result.returncode != 0:
            logger.warning("Calendar unavailable (permission?) — skipping calendar context")
            _CAL_CACHE[key] = (time.time(), [], _CAL_MISS_TTL)
            return []

        events = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("|")
            if len(parts) >= 4:
                events.append({
                    "title": parts[0].strip(),
                    "start": parts[1].strip(),
                    "end": parts[2].strip(),
                    "calendar": parts[3].strip(),
                    "location": parts[4].strip() if len(parts) > 4 else "",
                })
        _CAL_CACHE[key] = (time.time(), events, _CAL_TTL)
        return events

    except subprocess.TimeoutExpired:
        logger.warning("Calendar script timed out — skipping calendar context")
        _CAL_CACHE[key] = (time.time(), [], _CAL_MISS_TTL)
        return []
    except Exception as e:
        logger.error(f"Calendar error: {e}")
        _CAL_CACHE[key] = (time.time(), [], _CAL_MISS_TTL)
        return []


def format_events_for_context(events: list[dict]) -> str:
    """Format events as context for the LLM."""
    if not events:
        return "No upcoming events."
    lines = []
    for e in events:
        loc = f" at {e['location']}" if e.get("location") else ""
        lines.append(f"- {e['title']} ({e['start']} - {e['end']}){loc}")
    return "\n".join(lines)


def format_events_for_voice(events: list[dict]) -> str:
    """Format events as natural speech."""
    if not events:
        return "Your calendar is clear."
    if len(events) == 1:
        e = events[0]
        return f"You have {e['title']} at {e['start']}."
    return f"You have {len(events)} events. First up is {events[0]['title']} at {events[0]['start']}."
