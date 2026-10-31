"""Every test runs against a throwaway ~/.cortex and never touches launchd or the Keychain."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    import cortex.storage.settings_store as ss
    monkeypatch.setattr(ss, "_DIR", str(tmp_path))
    monkeypatch.setattr(ss, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    monkeypatch.setattr(ss, "OWNER_FILE", str(tmp_path / "owner.json"))
    monkeypatch.setattr(ss, "ONBOARD_FLAG", str(tmp_path / ".onboarded"))
    monkeypatch.setattr(ss, "AVATAR_FILE", str(tmp_path / "avatar.png"))
    return tmp_path
