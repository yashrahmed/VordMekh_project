"""Loopback-only web application for the encrypted secrets store."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import secrets
import threading
import time
import webbrowser
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .vault import (
    InvalidPinError,
    ValidationError,
    VaultCorruptError,
    VaultError,
    VaultLockedError,
    VaultStore,
)


HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_BODY_SIZE = 64 * 1024
SESSION_IDLE_SECONDS = 15 * 60
COOKIE_NAME = "local_vault_session"
STATIC_DIRECTORY = files("local_secrets_store").joinpath("static")


class ApplicationState:
    def __init__(self, vault: VaultStore):
        self.vault = vault
        self._session_digest: bytes | None = None
        self._csrf_token: str | None = None
        self._last_activity = 0.0
        self._failed_unlocks: list[float] = []
        self._lock = threading.RLock()
        self._idle_timer: threading.Timer | None = None
        self._session_generation = 0

    def open_session(self) -> tuple[str, str]:
        with self._lock:
            token = secrets.token_urlsafe(32)
            self._session_digest = hashlib.sha256(token.encode()).digest()
            self._csrf_token = secrets.token_urlsafe(24)
            self._last_activity = time.monotonic()
            self._failed_unlocks.clear()
            self._session_generation += 1
            self._schedule_expiry_locked()
            return token, self._csrf_token

    def authenticate(self, token: str | None) -> bool:
        with self._lock:
            if self._session_digest is None or token is None:
                return False
            if time.monotonic() - self._last_activity > SESSION_IDLE_SECONDS:
                self._end_session_locked()
                return False
            digest = hashlib.sha256(token.encode()).digest()
            if not hmac.compare_digest(digest, self._session_digest):
                return False
            self._last_activity = time.monotonic()
            self._schedule_expiry_locked()
            return True

    def validate_csrf(self, token: str | None) -> bool:
        with self._lock:
            return bool(
                self._csrf_token
                and token
                and hmac.compare_digest(token, self._csrf_token)
            )

    @property
    def csrf_token(self) -> str | None:
        with self._lock:
            return self._csrf_token

    def end_session(self) -> None:
        with self._lock:
            self._end_session_locked()

    def _end_session_locked(self) -> None:
        self._session_generation += 1
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None
        self.vault.lock()
        self._session_digest = None
        self._csrf_token = None
        self._last_activity = 0.0

    def _schedule_expiry_locked(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        generation = self._session_generation
        self._idle_timer = threading.Timer(
            SESSION_IDLE_SECONDS, self._expire_session, args=(generation,)
        )
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _expire_session(self, generation: int) -> None:
        with self._lock:
            if generation != self._session_generation or self._session_digest is None:
                return
            remaining = SESSION_IDLE_SECONDS - (time.monotonic() - self._last_activity)
            if remaining > 0:
                self._idle_timer = threading.Timer(
                    remaining, self._expire_session, args=(generation,)
                )
                self._idle_timer.daemon = True
                self._idle_timer.start()
                return
            self._end_session_locked()

    def unlock_delay(self) -> int:
        with self._lock:
            now = time.monotonic()
            self._failed_unlocks = [
                timestamp for timestamp in self._failed_unlocks if now - timestamp < 60
            ]
            if len(self._failed_unlocks) < 5:
                return 0
            elapsed = now - self._failed_unlocks[-1]
            return max(0, math.ceil(30 - elapsed))

    def record_failed_unlock(self) -> None:
        with self._lock:
            self._failed_unlocks.append(time.monotonic())


class VaultHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], state: ApplicationState
    ):
        super().__init__(address, handler)
        self.state = state

    @property
    def origin(self) -> str:
        return f"http://{HOST}:{self.server_port}"


class RequestHandler(BaseHTTPRequestHandler):
    server: VaultHTTPServer
    protocol_version = "HTTP/1.1"
    server_version = "LocalSecretsStore"
    sys_version = ""

    def do_GET(self) -> None:
        if not self._valid_host():
            self._json_error(HTTPStatus.BAD_REQUEST, "Invalid request host.")
            return
        path = urlsplit(self.path).path
        if path == "/api/status":
            self._get_status()
        elif path == "/api/secrets":
            self._get_secrets()
        elif path in {"/", "/index.html"}:
            self._serve_static("index.html", "text/html; charset=utf-8")
        elif path == "/app.js":
            self._serve_static("app.js", "text/javascript; charset=utf-8")
        elif path == "/styles.css":
            self._serve_static("styles.css", "text/css; charset=utf-8")
        else:
            self._json_error(HTTPStatus.NOT_FOUND, "Not found.")

    def do_POST(self) -> None:
        if not self._valid_host():
            self._json_error(HTTPStatus.BAD_REQUEST, "Invalid request host.")
            return
        path = urlsplit(self.path).path
        if not self._valid_mutation_origin():
            self._json_error(HTTPStatus.FORBIDDEN, "Invalid request origin.")
            return
        if path == "/api/initialize":
            self._initialize()
        elif path == "/api/unlock":
            self._unlock()
        elif path == "/api/lock":
            self._lock_vault()
        elif path == "/api/secrets":
            self._create_secret()
        else:
            self._json_error(HTTPStatus.NOT_FOUND, "Not found.")

    def do_PUT(self) -> None:
        if not self._valid_host():
            self._json_error(HTTPStatus.BAD_REQUEST, "Invalid request host.")
            return
        if not self._valid_mutation_origin():
            self._json_error(HTTPStatus.FORBIDDEN, "Invalid request origin.")
            return
        secret_id = self._secret_id_from_path()
        if secret_id is None:
            self._json_error(HTTPStatus.NOT_FOUND, "Not found.")
            return
        if not self._require_authenticated_mutation():
            return
        try:
            body = self._read_json()
            updated = self.server.state.vault.update_secret(secret_id, **self._secret_fields(body))
            self._send_json(HTTPStatus.OK, {"secret": updated})
        except (ValidationError, TypeError) as exc:
            self._json_error(HTTPStatus.BAD_REQUEST, str(exc))
        except VaultError as exc:
            self._json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_DELETE(self) -> None:
        if not self._valid_host():
            self._json_error(HTTPStatus.BAD_REQUEST, "Invalid request host.")
            return
        if not self._valid_mutation_origin():
            self._json_error(HTTPStatus.FORBIDDEN, "Invalid request origin.")
            return
        secret_id = self._secret_id_from_path()
        if secret_id is None:
            self._json_error(HTTPStatus.NOT_FOUND, "Not found.")
            return
        if not self._require_authenticated_mutation():
            return
        try:
            self.server.state.vault.delete_secret(secret_id)
            self._send_json(HTTPStatus.OK, {"ok": True})
        except ValidationError as exc:
            self._json_error(HTTPStatus.NOT_FOUND, str(exc))
        except VaultError as exc:
            self._json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def log_message(self, format: str, *args: Any) -> None:
        # Avoid logging URLs or user-driven values to a terminal.
        return

    def _get_status(self) -> None:
        authenticated = self.server.state.authenticate(self._session_cookie())
        self._send_json(
            HTTPStatus.OK,
            {
                "initialized": self.server.state.vault.initialized,
                "unlocked": authenticated,
                "csrfToken": self.server.state.csrf_token if authenticated else None,
                "idleTimeoutSeconds": SESSION_IDLE_SECONDS,
            },
        )

    def _get_secrets(self) -> None:
        if not self._require_authentication():
            return
        try:
            self._send_json(
                HTTPStatus.OK, {"secrets": self.server.state.vault.list_secrets()}
            )
        except VaultLockedError:
            self._json_error(HTTPStatus.UNAUTHORIZED, "Vault is locked.")

    def _initialize(self) -> None:
        if self.server.state.vault.initialized:
            self._json_error(HTTPStatus.CONFLICT, "A vault already exists.")
            return
        try:
            body = self._read_json()
            pin = body.get("pin", "")
            self.server.state.vault.initialize(pin)
            token, csrf = self.server.state.open_session()
            self._send_json(
                HTTPStatus.CREATED,
                {"ok": True, "csrfToken": csrf},
                cookie=token,
            )
        except ValidationError as exc:
            self._json_error(HTTPStatus.BAD_REQUEST, str(exc))
        except VaultError as exc:
            self._json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _unlock(self) -> None:
        delay = self.server.state.unlock_delay()
        if delay:
            self.close_connection = True
            self._send_json(
                HTTPStatus.TOO_MANY_REQUESTS,
                {"error": f"Too many attempts. Try again in {delay} seconds."},
                extra_headers={"Retry-After": str(delay)},
            )
            return
        try:
            body = self._read_json()
            self.server.state.vault.unlock(body.get("pin", ""))
            token, csrf = self.server.state.open_session()
            self._send_json(
                HTTPStatus.OK, {"ok": True, "csrfToken": csrf}, cookie=token
            )
        except (InvalidPinError, ValidationError):
            self.server.state.record_failed_unlock()
            self._json_error(HTTPStatus.UNAUTHORIZED, "Incorrect PIN.")
        except VaultCorruptError:
            self._json_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "The encrypted vault is damaged or unsupported.",
            )
        except VaultError as exc:
            self._json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _lock_vault(self) -> None:
        if not self._require_authenticated_mutation():
            return
        self.server.state.end_session()
        self._send_json(HTTPStatus.OK, {"ok": True}, clear_cookie=True)

    def _create_secret(self) -> None:
        if not self._require_authenticated_mutation():
            return
        try:
            body = self._read_json()
            created = self.server.state.vault.create_secret(**self._secret_fields(body))
            self._send_json(HTTPStatus.CREATED, {"secret": created})
        except (ValidationError, TypeError) as exc:
            self._json_error(HTTPStatus.BAD_REQUEST, str(exc))
        except VaultError as exc:
            self._json_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    @staticmethod
    def _secret_fields(body: dict[str, Any]) -> dict[str, str]:
        return {
            "name": body.get("name", ""),
            "username": body.get("username", ""),
            "secret": body.get("secret", ""),
            "notes": body.get("notes", ""),
        }

    def _secret_id_from_path(self) -> int | None:
        path = urlsplit(self.path).path
        prefix = "/api/secrets/"
        if not path.startswith(prefix):
            return None
        try:
            value = int(path[len(prefix) :])
            return value if value > 0 else None
        except ValueError:
            return None

    def _require_authentication(self) -> bool:
        if self.server.state.authenticate(self._session_cookie()):
            return True
        self._json_error(HTTPStatus.UNAUTHORIZED, "Vault is locked.")
        return False

    def _require_authenticated_mutation(self) -> bool:
        if not self._require_authentication():
            return False
        if not self.server.state.validate_csrf(self.headers.get("X-Vault-CSRF")):
            self._json_error(HTTPStatus.FORBIDDEN, "Invalid request token.")
            return False
        return True

    def _valid_mutation_origin(self) -> bool:
        return self.headers.get("Origin") == self.server.origin

    def _valid_host(self) -> bool:
        return self.headers.get("Host") == f"{HOST}:{self.server.server_port}"

    def _session_cookie(self) -> str | None:
        raw_cookie = self.headers.get("Cookie")
        if not raw_cookie:
            return None
        cookie = SimpleCookie()
        try:
            cookie.load(raw_cookie)
        except Exception:
            return None
        morsel = cookie.get(COOKIE_NAME)
        return morsel.value if morsel else None

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValidationError("Invalid request length.") from exc
        if not 0 < length <= MAX_BODY_SIZE:
            raise ValidationError("Request body is empty or too large.")
        if self.headers.get_content_type() != "application/json":
            raise ValidationError("Request must contain JSON.")
        try:
            value = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValidationError("Request contains invalid JSON.") from exc
        if not isinstance(value, dict):
            raise ValidationError("Request must contain a JSON object.")
        return value

    def _serve_static(self, name: str, content_type: str) -> None:
        try:
            content = STATIC_DIRECTORY.joinpath(name).read_bytes()
        except (FileNotFoundError, OSError):
            self._json_error(HTTPStatus.NOT_FOUND, "Not found.")
            return
        self.send_response(HTTPStatus.OK)
        self._security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _json_error(self, status: HTTPStatus, message: str) -> None:
        # An error may be returned before a request body is consumed. Closing the
        # connection prevents those unread bytes from being parsed as a new request.
        self.close_connection = True
        self._send_json(status, {"error": message})

    def _send_json(
        self,
        status: HTTPStatus,
        value: dict[str, Any],
        *,
        cookie: str | None = None,
        clear_cookie: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        content = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        if self.close_connection:
            self.send_header("Connection", "close")
        if cookie:
            self.send_header(
                "Set-Cookie",
                f"{COOKIE_NAME}={cookie}; HttpOnly; SameSite=Strict; Path=/",
            )
        elif clear_cookie:
            self.send_header(
                "Set-Cookie",
                f"{COOKIE_NAME}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0",
            )
        for name, header_value in (extra_headers or {}).items():
            self.send_header(name, header_value)
        self.end_headers()
        self.wfile.write(content)

    def _security_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'none'; connect-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local encrypted secrets store.")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"loopback port to use (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open the browser automatically"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    state = ApplicationState(VaultStore())
    try:
        server = VaultHTTPServer((HOST, args.port), RequestHandler, state)
    except OSError as exc:
        raise SystemExit(
            f"Could not start on {HOST}:{args.port}. Try --port with another local port."
        ) from exc

    url = server.origin
    print(f"Local Secrets Store is ready at {url}")
    print("Press Ctrl+C to stop it.")
    if not args.no_browser:
        threading.Timer(0.25, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nLocking the vault and stopping.")
    finally:
        state.end_session()
        server.server_close()


if __name__ == "__main__":
    main()
