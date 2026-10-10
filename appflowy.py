"""
AppFlowy Client — Connects to AppFlowy Cloud API for todo/task management.
Handles auth, workspace discovery, database CRUD for tasks.

AppFlowy API: https://beta.appflowy.cloud (or self-hosted)
Auth: GoTrue (email/password → JWT)
Data model: Workspace → Database (table/board) → Rows (tasks)
"""

import json
import logging
import urllib.request
import urllib.error
from datetime import datetime, timedelta

logger = logging.getLogger("cortex.appflowy")


class AppFlowyClient:
    """REST client for AppFlowy Cloud API."""

    def __init__(self, base_url: str, email: str = None, password: str = None, token: str = None):
        self.base_url = base_url.rstrip("/")
        self.email = email
        self.password = password
        self._token = token
        self._token_expires = None
        self._workspace_id = None
        self._todo_db_id = None

    # ─── Auth ───

    def authenticate(self) -> bool:
        """Get a JWT via GoTrue email/password grant."""
        if self._token and self._token_expires and datetime.now() < self._token_expires:
            return True

        if not self.email or not self.password:
            if self._token:
                return True
            logger.error("No credentials or token configured")
            return False

        try:
            data = json.dumps({
                "email": self.email,
                "password": self.password,
                "grant_type": "password"
            }).encode()

            req = urllib.request.Request(
                f"{self.base_url}/gotrue/token?grant_type=password",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read())

            self._token = body.get("access_token", "")
            expires_in = body.get("expires_in", 3600)
            self._token_expires = datetime.now() + timedelta(seconds=expires_in - 60)
            logger.info("AppFlowy authenticated successfully")
            return True

        except Exception as e:
            logger.error(f"AppFlowy auth failed: {e}")
            return False

    def _request(self, method: str, path: str, body: dict = None) -> dict:
        """Make an authenticated API request."""
        if not self.authenticate():
            return {"error": "Not authenticated"}

        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body else None

        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self._token}")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            logger.error(f"AppFlowy API error {e.code}: {error_body[:200]}")
            return {"error": f"HTTP {e.code}", "detail": error_body[:200]}
        except Exception as e:
            logger.error(f"AppFlowy request failed: {e}")
            return {"error": str(e)}

    # ─── Workspace ───

    def get_workspaces(self) -> list:
        """List all workspaces."""
        result = self._request("GET", "/api/workspace")
        return result if isinstance(result, list) else result.get("data", [])

    def set_workspace(self, workspace_id: str):
        """Set the active workspace."""
        self._workspace_id = workspace_id

    # ─── Database Discovery ───

    def get_databases(self) -> list:
        """List all databases in the active workspace."""
        if not self._workspace_id:
            return []
        result = self._request("GET", f"/api/workspace/{self._workspace_id}/database")
        return result if isinstance(result, list) else result.get("data", [])

    def get_folder(self) -> dict:
        """Get workspace folder structure to find databases."""
        if not self._workspace_id:
            return {}
        return self._request("GET", f"/api/workspace/{self._workspace_id}/folder")

    def set_todo_database(self, database_id: str):
        """Set which database is the todo list."""
        self._todo_db_id = database_id

    # ─── Todo Operations ───

    def get_fields(self) -> list:
        """Get fields (columns) of the todo database."""
        if not self._workspace_id or not self._todo_db_id:
            return []
        result = self._request(
            "GET",
            f"/api/workspace/{self._workspace_id}/database/{self._todo_db_id}/fields"
        )
        return result if isinstance(result, list) else result.get("data", [])

    def get_todos(self) -> list:
        """Get all rows (tasks) from the todo database."""
        if not self._workspace_id or not self._todo_db_id:
            return []
        result = self._request(
            "GET",
            f"/api/workspace/{self._workspace_id}/database/{self._todo_db_id}/row/detail"
        )
        return result if isinstance(result, list) else result.get("data", [])

    def add_todo(self, title: str, extra_fields: dict = None) -> dict:
        """Create a new task in the todo database."""
        if not self._workspace_id or not self._todo_db_id:
            return {"error": "No workspace or todo database configured"}

        # Build the row data — AppFlowy uses cell_by_field format
        cells = {}
        if extra_fields:
            cells.update(extra_fields)

        # The title field is typically the first field
        fields = self.get_fields()
        title_field = None
        for f in fields:
            if isinstance(f, dict):
                # First RichText or Title field
                field_type = f.get("field_type") or f.get("type", "")
                if field_type in ("RichText", "Title", 0):
                    title_field = f.get("id") or f.get("field_id")
                    break

        if title_field:
            cells[title_field] = title

        body = {"cells": cells} if cells else {}

        result = self._request(
            "POST",
            f"/api/workspace/{self._workspace_id}/database/{self._todo_db_id}/row",
            body=body
        )
        return result

    def complete_todo(self, row_id: str, checkbox_field_id: str = None) -> dict:
        """Mark a todo as complete (check the checkbox field)."""
        if not self._workspace_id or not self._todo_db_id:
            return {"error": "No workspace or todo database configured"}

        # Find checkbox field if not provided
        if not checkbox_field_id:
            fields = self.get_fields()
            for f in fields:
                if isinstance(f, dict):
                    field_type = f.get("field_type") or f.get("type", "")
                    if field_type in ("Checkbox", 4):
                        checkbox_field_id = f.get("id") or f.get("field_id")
                        break

        if not checkbox_field_id:
            return {"error": "No checkbox field found in todo database"}

        body = {
            "row_id": row_id,
            "cells": {checkbox_field_id: "Yes"}
        }

        return self._request(
            "PUT",
            f"/api/workspace/{self._workspace_id}/database/{self._todo_db_id}/row",
            body=body
        )

    # ─── Formatted output ───

    def format_todos(self, todos: list = None) -> str:
        """Format todos for display."""
        if todos is None:
            todos = self.get_todos()

        if not todos:
            return "No tasks found."

        lines = []
        for i, row in enumerate(todos, 1):
            if isinstance(row, dict):
                cells = row.get("cells", {})
                # Try to extract title and status
                title = ""
                done = False
                for field_id, cell in cells.items():
                    if isinstance(cell, dict):
                        if cell.get("field_type") in ("RichText", "Title", 0):
                            title = cell.get("data", "") or cell.get("value", "")
                        elif cell.get("field_type") in ("Checkbox", 4):
                            done = cell.get("data") in ("Yes", True, "true")
                    elif isinstance(cell, str):
                        if not title:
                            title = cell

                if not title:
                    title = str(row.get("id", f"Task {i}"))[:50]

                status = "✓" if done else "○"
                lines.append(f"{status} {title}")

        return "\n".join(lines) if lines else "No tasks found."


def create_client(config: dict) -> AppFlowyClient:
    """Factory: create AppFlowy client from config."""
    af_config = config.get("appflowy", {})
    client = AppFlowyClient(
        base_url=af_config.get("base_url", "https://beta.appflowy.cloud"),
        email=af_config.get("email"),
        password=af_config.get("password"),
        token=af_config.get("token"),
    )

    workspace_id = af_config.get("workspace_id")
    if workspace_id:
        client.set_workspace(workspace_id)

    todo_db_id = af_config.get("todo_database_id")
    if todo_db_id:
        client.set_todo_database(todo_db_id)

    return client
