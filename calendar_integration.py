"""
Calendar Integration — Reads macOS Calendar events via AppleScript.
Gives Cortex awareness of your schedule for time-based context.
"""

import subprocess
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("cortex.calendar")


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
    """Run AppleScript and parse the output."""
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            logger.error(f"Calendar script failed: {result.stderr}")
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
        return events

    except subprocess.TimeoutExpired:
        logger.error("Calendar script timed out")
        return []
    except Exception as e:
        logger.error(f"Calendar error: {e}")
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
