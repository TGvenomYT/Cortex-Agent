"""
mail.py — Gmail over IMAP/SMTP (stdlib only), for one or many accounts.

Uses Gmail app passwords (needs 2-Step Verification on the Google account), so
there's no OAuth / Google Cloud project and nothing leaves the machine except
the IMAP/SMTP connection to Gmail itself. Reading uses BODY.PEEK so messages
are NOT marked read.
"""
from __future__ import annotations

import imaplib
import smtplib
import email
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.utils import parseaddr

IMAP_HOST = "imap.gmail.com"
SMTP_HOST = "smtp.gmail.com"
_TIMEOUT = 20


def _dec(raw) -> str:
    if not raw:
        return ""
    try:
        return str(make_header(decode_header(raw)))
    except Exception:
        return str(raw)


def test_login(addr: str, pw: str) -> tuple[bool, str]:
    try:
        m = imaplib.IMAP4_SSL(IMAP_HOST, timeout=_TIMEOUT)
        m.login(addr, pw)
        m.logout()
        return True, ""
    except imaplib.IMAP4.error:
        return False, "Login rejected — check the address and the 16-char app password (not your normal password)."
    except Exception as e:
        return False, str(e)


def _open(addr: str, pw: str):
    m = imaplib.IMAP4_SSL(IMAP_HOST, timeout=_TIMEOUT)
    m.login(addr, pw)
    return m


def unread_count(addr: str, pw: str) -> int:
    try:
        m = _open(addr, pw)
        m.select("INBOX", readonly=True)
        typ, data = m.search(None, "UNSEEN")
        m.logout()
        return len(data[0].split()) if typ == "OK" and data and data[0] else 0
    except Exception:
        return -1


def fetch_unread(addr: str, pw: str, limit: int = 10) -> list[dict]:
    """Recent unread messages (newest first). Does NOT mark them read."""
    out = []
    try:
        m = _open(addr, pw)
        m.select("INBOX", readonly=True)
        typ, data = m.search(None, "UNSEEN")
        if typ != "OK" or not data or not data[0]:
            m.logout(); return []
        ids = data[0].split()[-limit:][::-1]
        for i in ids:
            typ, d = m.fetch(i, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])")
            if typ != "OK" or not d or not d[0]:
                continue
            hdr = email.message_from_bytes(d[0][1])
            frm = parseaddr(_dec(hdr.get("From")))
            out.append({"account": addr, "from": frm[0] or frm[1], "from_email": frm[1],
                        "subject": _dec(hdr.get("Subject")) or "(no subject)",
                        "date": _dec(hdr.get("Date"))})
        m.logout()
    except Exception as e:
        return [{"account": addr, "error": str(e)}]
    return out


def search(addr: str, pw: str, query: str, limit: int = 12) -> list[dict]:
    out = []
    try:
        m = _open(addr, pw)
        m.select("INBOX", readonly=True)
        # Gmail supports X-GM-RAW for full search-operator queries
        try:
            typ, data = m.search(None, "X-GM-RAW", f'"{query}"')
        except Exception:
            typ, data = m.search(None, "TEXT", query)
        if typ != "OK" or not data or not data[0]:
            m.logout(); return []
        for i in data[0].split()[-limit:][::-1]:
            typ, d = m.fetch(i, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])")
            if typ != "OK" or not d or not d[0]:
                continue
            hdr = email.message_from_bytes(d[0][1])
            frm = parseaddr(_dec(hdr.get("From")))
            out.append({"account": addr, "from": frm[0] or frm[1],
                        "subject": _dec(hdr.get("Subject")) or "(no subject)",
                        "date": _dec(hdr.get("Date"))})
        m.logout()
    except Exception as e:
        return [{"account": addr, "error": str(e)}]
    return out


def send(addr: str, pw: str, to: str, subject: str, body: str) -> tuple[bool, str]:
    try:
        msg = EmailMessage()
        msg["From"] = addr
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)
        s = smtplib.SMTP_SSL(SMTP_HOST, 465, timeout=_TIMEOUT)
        s.login(addr, pw)
        s.send_message(msg)
        s.quit()
        return True, ""
    except Exception as e:
        return False, str(e)
