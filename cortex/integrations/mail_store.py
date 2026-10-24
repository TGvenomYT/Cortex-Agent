"""
mail_store.py — manage connected Gmail accounts.

The account list (email + label) lives in ~/.cortex/settings.json; each app
password is stored in the macOS Keychain (service "cortex", account
"mail:<email>"), never in a file.
"""
from __future__ import annotations

import cortex.storage.settings_store as settings_store
from cortex.util.cortex_bootstrap import keychain_get, keychain_set


def list_accounts() -> list[dict]:
    return settings_store.load().get("emails", []) or []


def password(addr: str) -> str | None:
    return keychain_get("mail:" + addr)


def add(addr: str, pw: str, label: str = "") -> None:
    keychain_set("mail:" + addr, pw)
    accts = list_accounts()
    accts = [a for a in accts if a.get("email") != addr]  # de-dupe
    accts.append({"email": addr, "label": label or addr.split("@")[0]})
    settings_store.save({"emails": accts})


def remove(addr: str) -> None:
    import cortex.util.osutil as osutil
    accts = [a for a in list_accounts() if a.get("email") != addr]
    settings_store.save({"emails": accts})
    osutil.secret_delete("mail:" + addr)  # best-effort, cross-platform
