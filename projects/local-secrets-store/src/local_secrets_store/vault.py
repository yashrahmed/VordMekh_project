"""Encrypted, in-memory SQLite storage for the local secrets store."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import struct
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


MAGIC = b"LOCALVAULT"
FORMAT_VERSION = 1
KDF_LOG_N = 16
SALT_SIZE = 16
NONCE_SIZE = 12
PIN_HASH_ITERATIONS = 600_000
PIN_PATTERN = re.compile(r"^[0-9]{4}$")


class VaultError(Exception):
    """Base class for vault errors."""


class InvalidPinError(VaultError):
    """Raised when a PIN cannot unlock the vault."""


class VaultCorruptError(VaultError):
    """Raised when the encrypted vault is malformed."""


class VaultLockedError(VaultError):
    """Raised when an operation requires an unlocked vault."""


class ValidationError(VaultError):
    """Raised when user-provided data is invalid."""


def validate_pin(pin: str) -> None:
    """Require exactly four ASCII digits."""
    if not isinstance(pin, str) or not PIN_PATTERN.fullmatch(pin):
        raise ValidationError("PIN must contain exactly four digits.")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _derive_encryption_key(pin: str, salt: bytes, log_n: int) -> bytes:
    if not 15 <= log_n <= 20:
        raise VaultCorruptError("The vault uses an unsupported key-derivation cost.")
    try:
        return hashlib.scrypt(
            pin.encode("ascii"),
            salt=salt,
            n=1 << log_n,
            r=8,
            p=1,
            maxmem=1024 * 1024 * 1024,
            dklen=32,
        )
    except (ValueError, MemoryError) as exc:
        raise VaultCorruptError("Unable to derive the vault encryption key.") from exc


def _pin_hash(pin: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256",
        pin.encode("ascii"),
        salt,
        PIN_HASH_ITERATIONS,
        dklen=32,
    )


class VaultStore:
    """Keep SQLite in memory and persist only authenticated ciphertext."""

    def __init__(self, data_dir: Path | None = None, *, kdf_log_n: int = KDF_LOG_N):
        self.data_dir = data_dir or (Path.home() / ".personal-creds")
        self.vault_path = self.data_dir / "vault.enc"
        self._kdf_log_n = kdf_log_n
        self._connection: sqlite3.Connection | None = None
        self._key: bytes | None = None
        self._salt: bytes | None = None
        self._lock = threading.RLock()

    @property
    def initialized(self) -> bool:
        return self.vault_path.is_file()

    @property
    def unlocked(self) -> bool:
        with self._lock:
            return self._connection is not None

    def initialize(self, pin: str) -> None:
        validate_pin(pin)
        with self._lock:
            if self.initialized:
                raise ValidationError("A vault already exists.")

            self._prepare_data_directory()
            connection = sqlite3.connect(":memory:", check_same_thread=False)
            connection.row_factory = sqlite3.Row
            self._create_schema(connection)

            pin_salt = secrets.token_bytes(SALT_SIZE)
            with connection:
                connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    [
                        ("schema_version", b"1"),
                        ("pin_salt", pin_salt),
                        ("pin_hash", _pin_hash(pin, pin_salt)),
                    ],
                )

            salt = secrets.token_bytes(SALT_SIZE)
            key = _derive_encryption_key(pin, salt, self._kdf_log_n)
            self._connection = connection
            self._salt = salt
            self._key = key
            try:
                self._persist()
            except Exception:
                self.lock()
                raise

    def unlock(self, pin: str) -> None:
        validate_pin(pin)
        with self._lock:
            if not self.initialized:
                raise ValidationError("The vault has not been initialized.")

            try:
                encrypted = self.vault_path.read_bytes()
                minimum_size = len(MAGIC) + 2 + SALT_SIZE + NONCE_SIZE + 16
                if len(encrypted) < minimum_size:
                    raise VaultCorruptError("The vault file is incomplete.")

                offset = len(MAGIC)
                if encrypted[:offset] != MAGIC:
                    raise VaultCorruptError("The vault file has an invalid header.")

                version, log_n = struct.unpack("!BB", encrypted[offset : offset + 2])
                offset += 2
                if version != FORMAT_VERSION:
                    raise VaultCorruptError("The vault format is not supported.")

                salt = encrypted[offset : offset + SALT_SIZE]
                offset += SALT_SIZE
                nonce = encrypted[offset : offset + NONCE_SIZE]
                offset += NONCE_SIZE
                prefix = encrypted[:offset]
                ciphertext = encrypted[offset:]
                key = _derive_encryption_key(pin, salt, log_n)
                database_bytes = AESGCM(key).decrypt(nonce, ciphertext, prefix)
            except InvalidTag as exc:
                raise InvalidPinError("Incorrect PIN.") from exc
            except OSError as exc:
                raise VaultError("Unable to read the vault file.") from exc

            connection = sqlite3.connect(":memory:", check_same_thread=False)
            connection.row_factory = sqlite3.Row
            try:
                connection.deserialize(database_bytes)
                self._validate_database(connection, pin)
            except InvalidPinError:
                connection.close()
                raise
            except sqlite3.DatabaseError as exc:
                connection.close()
                raise VaultCorruptError("The decrypted vault is not a valid database.") from exc

            if self._connection is not None:
                self._connection.close()
            self._connection = connection
            self._key = key
            self._salt = salt
            self._kdf_log_n = log_n

    def lock(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
            self._connection = None
            self._key = None
            self._salt = None

    def list_secrets(self) -> list[dict[str, Any]]:
        with self._lock:
            connection = self._require_connection()
            rows = connection.execute(
                """
                SELECT id, name, username, secret, notes, created_at, updated_at
                FROM secrets
                ORDER BY lower(name), id
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def create_secret(
        self, *, name: str, username: str = "", secret: str = "", notes: str = ""
    ) -> dict[str, Any]:
        values = self._validate_secret_fields(name, username, secret, notes)
        with self._lock:
            connection = self._require_connection()
            now = _utc_now()
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO secrets(name, username, secret, notes, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (*values, now, now),
                )
            self._persist()
            return self.get_secret(cursor.lastrowid)

    def get_secret(self, secret_id: int) -> dict[str, Any]:
        with self._lock:
            connection = self._require_connection()
            row = connection.execute(
                """
                SELECT id, name, username, secret, notes, created_at, updated_at
                FROM secrets WHERE id = ?
                """,
                (secret_id,),
            ).fetchone()
            if row is None:
                raise ValidationError("Secret not found.")
            return dict(row)

    def update_secret(
        self,
        secret_id: int,
        *,
        name: str,
        username: str = "",
        secret: str = "",
        notes: str = "",
    ) -> dict[str, Any]:
        values = self._validate_secret_fields(name, username, secret, notes)
        with self._lock:
            connection = self._require_connection()
            with connection:
                cursor = connection.execute(
                    """
                    UPDATE secrets
                    SET name = ?, username = ?, secret = ?, notes = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (*values, _utc_now(), secret_id),
                )
            if cursor.rowcount == 0:
                raise ValidationError("Secret not found.")
            self._persist()
            return self.get_secret(secret_id)

    def delete_secret(self, secret_id: int) -> None:
        with self._lock:
            connection = self._require_connection()
            with connection:
                cursor = connection.execute(
                    "DELETE FROM secrets WHERE id = ?", (secret_id,)
                )
            if cursor.rowcount == 0:
                raise ValidationError("Secret not found.")
            self._persist()

    def _prepare_data_directory(self) -> None:
        try:
            self.data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.data_dir, 0o700)
        except OSError as exc:
            raise VaultError("Unable to create the private data directory.") from exc

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value BLOB NOT NULL
            );

            CREATE TABLE secrets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                username TEXT NOT NULL DEFAULT '',
                secret TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )

    @staticmethod
    def _validate_database(connection: sqlite3.Connection, pin: str) -> None:
        rows = connection.execute(
            "SELECT key, value FROM metadata WHERE key IN ('schema_version', 'pin_salt', 'pin_hash')"
        ).fetchall()
        metadata = {row["key"]: row["value"] for row in rows}
        if metadata.get("schema_version") != b"1":
            raise VaultCorruptError("The vault schema is not supported.")
        if "pin_salt" not in metadata or "pin_hash" not in metadata:
            raise VaultCorruptError("The vault PIN verifier is missing.")
        actual = _pin_hash(pin, metadata["pin_salt"])
        if not hmac.compare_digest(actual, metadata["pin_hash"]):
            raise InvalidPinError("Incorrect PIN.")

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise VaultLockedError("The vault is locked.")
        return self._connection

    @staticmethod
    def _validate_secret_fields(
        name: str, username: str, secret: str, notes: str
    ) -> tuple[str, str, str, str]:
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("Name is required.")
        fields = (name.strip(), username, secret, notes)
        if not all(isinstance(value, str) for value in fields):
            raise ValidationError("Secret fields must be text.")
        limits = (200, 500, 20_000, 20_000)
        if any(len(value) > limit for value, limit in zip(fields, limits, strict=True)):
            raise ValidationError("One or more fields are too long.")
        return fields

    def _persist(self) -> None:
        connection = self._require_connection()
        if self._key is None or self._salt is None:
            raise VaultLockedError("The vault is locked.")

        nonce = secrets.token_bytes(NONCE_SIZE)
        prefix = (
            MAGIC
            + struct.pack("!BB", FORMAT_VERSION, self._kdf_log_n)
            + self._salt
            + nonce
        )
        ciphertext = AESGCM(self._key).encrypt(nonce, connection.serialize(), prefix)
        payload = prefix + ciphertext

        self._prepare_data_directory()
        temporary_path = self.data_dir / f".vault-{secrets.token_hex(8)}.tmp"
        try:
            descriptor = os.open(
                temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.vault_path)
            os.chmod(self.vault_path, 0o600)
            directory_descriptor = os.open(self.data_dir, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError as exc:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise VaultError("Unable to save the encrypted vault.") from exc
