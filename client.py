"""Thin HTTP client for the Endless Elevator (无尽电梯) web API."""
from __future__ import annotations

import os
from typing import Any

import requests

BASE_URL = "https://elevator.slime.icu"
API_ROOT = f"{BASE_URL}/elevator/api"


def _load_dotenv(path: str = os.path.join(os.path.dirname(__file__), ".env")) -> None:
    """Populate os.environ from a simple KEY=VALUE file without overriding existing vars."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


class GameError(RuntimeError):
    def __init__(self, code: str | None, message: str, payload: dict | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.payload = payload or {}


class ElevatorClient:
    """Session-based client. Keeps the latest full state and applies server deltas."""

    def __init__(self, timeout: float = 20.0):
        self.http = requests.Session()
        self.http.headers.update({
            "Origin": BASE_URL,
            "Referer": f"{BASE_URL}/elevator/",
            "Accept": "application/json",
        })
        self.timeout = timeout
        self.state: dict[str, Any] | None = None
        self.sync_token: str = ""

    # -- transport -------------------------------------------------------
    def _call(self, path: str, body: dict | None = None) -> dict:
        headers = {}
        if path.startswith("/game/"):
            headers["X-Elevator-Sync"] = "1"
            if body is not None and self.sync_token:
                headers["X-Elevator-Base"] = self.sync_token
        if body is None:
            resp = self.http.get(API_ROOT + path, headers=headers, timeout=self.timeout)
        else:
            resp = self.http.post(API_ROOT + path, json=body, headers=headers, timeout=self.timeout)
        try:
            data = resp.json()
        except ValueError as exc:  # HTML error page / truncated body (gateway hiccup, restart)
            raise requests.ConnectionError(f"non-JSON response (HTTP {resp.status_code})") from exc
        if not resp.ok or data.get("error"):
            err = data.get("error")
            code = err.get("code") if isinstance(err, dict) else data.get("code")
            msg = err.get("message") if isinstance(err, dict) else str(err)
            if isinstance(data.get("state"), dict):
                self._set_state(data["state"], data.get("sync_token"))
            raise GameError(code, msg or "request failed", data)
        return data

    def _set_state(self, state: dict, token: str | None) -> None:
        self.state = state
        if token:
            self.sync_token = token

    # -- auth ------------------------------------------------------------
    def login(self, username: str | None = None, password: str | None = None) -> dict:
        _load_dotenv()
        username = username or os.environ["ELEVATOR_USER"]
        password = password or os.environ.get("ELEVATOR_PASS") or os.environ["ELEVATOR_PASSWORD"]
        return self._call("/auth/login", {"username": username, "password": password, "remember": True})

    # -- game ------------------------------------------------------------
    def refresh(self) -> dict:
        data = self._call("/game/state")
        self._set_state(data.get("state", data), data.get("sync_token"))
        return self.state

    @property
    def revision(self) -> str:
        return self.state.get("revision") or self.state.get("version")

    def command(self, command: str, params: dict | None = None) -> dict:
        """Structured in-game command (start, down, fight, flee, extract, use, event...)."""
        data = self._call("/game/command", {"command": command, "params": params or {}, "revision": self.revision})
        self._absorb(data)
        return data

    def action(self, action: str, argument: str = "") -> dict:
        """Text-style action (help, lookup, quests...). Returns server reply."""
        data = self._call("/game/action", {"action": action, "argument": argument, "revision": self.revision})
        self._absorb(data)
        return data

    def _absorb(self, data: dict) -> None:
        if data.get("delta"):
            patch = data["delta"]
            if patch.get("base") == self.sync_token and patch.get("base_revision") == self.revision:
                new = {**self.state, **patch.get("fields", {})}
                for key, changed in (patch.get("sections") or {}).items():
                    new[key] = {**(new.get(key) or {}), **changed}
                for name, upd in (patch.get("lists") or {}).items():
                    original = self.state.get(name) or []
                    by_id = {row[upd["key"]]: row for row in original}
                    for row in upd.get("upsert", []):
                        by_id[row[upd["key"]]] = row
                    order = upd.get("order") or [row[upd["key"]] for row in original]
                    new[name] = [by_id[i] for i in order]
                self._set_state(new, data.get("sync_token"))
            else:
                self.refresh()
        elif data.get("state"):
            self._set_state(data["state"], data.get("sync_token"))
