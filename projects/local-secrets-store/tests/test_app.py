from __future__ import annotations

import http.client
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Any

from local_secrets_store.app import (
    HOST,
    ApplicationState,
    RequestHandler,
    VaultHTTPServer,
)
from local_secrets_store.vault import VaultStore


@contextmanager
def running_server(tmp_path: Path) -> Iterator[VaultHTTPServer]:
    vault = VaultStore(tmp_path / ".personal-creds", kdf_log_n=15)
    server = VaultHTTPServer((HOST, 0), RequestHandler, ApplicationState(vault))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.state.end_session()
        server.shutdown()
        server.server_close()
        thread.join()


def send(
    connection: http.client.HTTPConnection,
    server: VaultHTTPServer,
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    cookie: str | None = None,
    csrf: str | None = None,
    origin: str | None = None,
) -> tuple[int, dict[str, Any], list[tuple[str, str]]]:
    headers = {"Origin": origin or server.origin}
    body = None
    if payload is not None:
        body = json.dumps(payload)
        headers["Content-Type"] = "application/json"
    if cookie:
        headers["Cookie"] = cookie
    if csrf:
        headers["X-Vault-CSRF"] = csrf
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    result = json.loads(response.read())
    return response.status, result, response.getheaders()


def test_http_lifecycle_and_security_headers(tmp_path: Path) -> None:
    with running_server(tmp_path) as server:
        connection = http.client.HTTPConnection(HOST, server.server_port)

        status, body, _ = send(connection, server, "GET", "/api/status")
        assert status == HTTPStatus.OK
        assert body == {
            "initialized": False,
            "unlocked": False,
            "csrfToken": None,
            "idleTimeoutSeconds": 900,
        }

        status, body, headers = send(
            connection,
            server,
            "POST",
            "/api/initialize",
            payload={"pin": "2468"},
        )
        assert status == HTTPStatus.CREATED
        csrf = body["csrfToken"]
        cookie = next(value.split(";", 1)[0] for name, value in headers if name == "Set-Cookie")
        assert any(name == "Content-Security-Policy" for name, _ in headers)

        status, _, _ = send(
            connection,
            server,
            "POST",
            "/api/secrets",
            payload={"name": "Email", "username": "me", "secret": "value", "notes": ""},
            cookie=cookie,
            csrf=csrf,
            origin="http://example.test",
        )
        assert status == HTTPStatus.FORBIDDEN

        status, body, _ = send(
            connection,
            server,
            "POST",
            "/api/secrets",
            payload={"name": "Email", "username": "me", "secret": "value", "notes": ""},
            cookie=cookie,
            csrf=csrf,
        )
        assert status == HTTPStatus.CREATED
        secret_id = body["secret"]["id"]

        status, body, _ = send(
            connection, server, "GET", "/api/secrets", cookie=cookie
        )
        assert status == HTTPStatus.OK
        assert body["secrets"][0]["secret"] == "value"

        status, _, _ = send(
            connection,
            server,
            "DELETE",
            f"/api/secrets/{secret_id}",
            cookie=cookie,
            csrf=csrf,
        )
        assert status == HTTPStatus.OK

        status, body, _ = send(
            connection, server, "GET", "/api/secrets", cookie=cookie
        )
        assert status == HTTPStatus.OK
        assert body["secrets"] == []

        status, _, _ = send(
            connection,
            server,
            "POST",
            "/api/lock",
            cookie=cookie,
            csrf=csrf,
        )
        assert status == HTTPStatus.OK

        status, body, headers = send(
            connection,
            server,
            "POST",
            "/api/unlock",
            payload={"pin": "2468"},
        )
        assert status == HTTPStatus.OK
        assert body["csrfToken"]
        assert any(name == "Set-Cookie" for name, _ in headers)
        connection.close()


def test_server_session_expiry_locks_in_memory_vault(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import local_secrets_store.app as app_module

    monkeypatch.setattr(app_module, "SESSION_IDLE_SECONDS", 0.05)
    vault = VaultStore(tmp_path / ".personal-creds", kdf_log_n=15)
    vault.initialize("1234")
    state = ApplicationState(vault)
    state.open_session()

    deadline = time.monotonic() + 1
    while vault.unlocked and time.monotonic() < deadline:
        time.sleep(0.01)

    assert not vault.unlocked
