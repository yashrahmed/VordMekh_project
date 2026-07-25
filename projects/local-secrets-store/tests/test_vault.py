from __future__ import annotations

import stat
from pathlib import Path

import pytest

from local_secrets_store.vault import (
    InvalidPinError,
    ValidationError,
    VaultLockedError,
    VaultStore,
)


@pytest.fixture
def vault(tmp_path: Path) -> VaultStore:
    # A lower test-only work factor keeps the test suite fast. Production uses 2**16.
    return VaultStore(tmp_path / ".personal-creds", kdf_log_n=15)


def test_initialize_crud_lock_and_unlock(vault: VaultStore) -> None:
    vault.initialize("1234")
    assert vault.initialized
    assert vault.unlocked

    created = vault.create_secret(
        name="Email",
        username="person@example.com",
        secret="correct horse battery staple",
        notes="Recovery codes are elsewhere.",
    )
    assert created["id"] == 1
    assert vault.list_secrets()[0]["secret"] == "correct horse battery staple"

    updated = vault.update_secret(
        created["id"],
        name="Personal email",
        username="person@example.com",
        secret="new secret",
        notes="Updated",
    )
    assert updated["name"] == "Personal email"

    encrypted = vault.vault_path.read_bytes()
    assert b"SQLite format" not in encrypted
    assert b"new secret" not in encrypted
    assert b"person@example.com" not in encrypted

    vault.lock()
    assert not vault.unlocked
    with pytest.raises(VaultLockedError):
        vault.list_secrets()
    with pytest.raises(InvalidPinError):
        vault.unlock("9999")

    # Reopen with a fresh object to prove the encrypted file is the source of truth.
    reopened = VaultStore(vault.data_dir)
    reopened.unlock("1234")
    assert reopened.list_secrets()[0]["notes"] == "Updated"
    reopened.delete_secret(created["id"])
    reopened.lock()
    reopened.unlock("1234")
    assert reopened.list_secrets() == []


@pytest.mark.parametrize("pin", ["", "123", "12345", "abcd", "12 4", "１２３４", 1234])
def test_pin_must_be_exactly_four_ascii_digits(vault: VaultStore, pin: object) -> None:
    with pytest.raises(ValidationError):
        vault.initialize(pin)


def test_private_permissions(vault: VaultStore) -> None:
    vault.initialize("4321")
    directory_mode = stat.S_IMODE(vault.data_dir.stat().st_mode)
    file_mode = stat.S_IMODE(vault.vault_path.stat().st_mode)
    assert directory_mode == 0o700
    assert file_mode == 0o600


def test_rejects_empty_name(vault: VaultStore) -> None:
    vault.initialize("1234")
    with pytest.raises(ValidationError, match="Name is required"):
        vault.create_secret(name="  ", secret="value")
