# Local Secrets Store Security Review

**Review date:** July 25, 2026

**Reviewed branch:** `codex/local-secrets-store`

**Review scope:** Vault encryption and persistence, PIN handling, local HTTP
session design, browser UI, dependency risk, filesystem behavior, failure
recovery, and existing security tests.

## Executive summary

The application is substantially safer than plaintext credential storage, but
it is not ready for important credentials in its current form.

No obvious SQL injection, DOM-based XSS, path traversal, remote-network
exposure, or plaintext-on-disk defect was found. The review did identify three
high-severity issues that undermine the vault's primary security boundary:

1. A four-digit PIN can be exhaustively searched offline in minutes.
2. The browser session cookie is shared with other services on
   `127.0.0.1`, regardless of port.
3. Automatic locking does not clear plaintext from an open secret editor.

Known-vulnerable dependencies, cross-process lost updates, and inconsistent
state after failed persistence should also be corrected before relying on the
vault.

## Threat model

The review considered:

- Theft of `vault.enc` or one of its backups.
- Malicious or compromised services listening on another local port.
- Malicious web content attempting DNS rebinding or cross-origin requests.
- Physical access after the automatic lock timer fires.
- Browser extensions, developer tools, clipboard readers, and process-memory
  inspection.
- Concurrent vault instances, storage failures, and corrupted vault files.

A fully compromised logged-in user account, administrator, browser extension
with access to local pages, or attacker capable of dumping process memory
cannot be reliably resisted by this Python and browser architecture. Those are
residual host-security risks rather than guarantees this application can make.

## Findings

### High: Four-digit PINs are practical to crack offline

The key derivation parameters and PIN restriction are defined in
[`vault.py`](src/local_secrets_store/vault.py#L23-L27). The encryption key is
derived solely from a space of 10,000 possible PINs.

On the review machine, one production-parameter scrypt derivation took a
median of 89 ms:

- Average PIN discovery: approximately 7.4 minutes on one CPU core.
- Exhaustive search: approximately 14.9 minutes on one CPU core.
- Parallel hardware can reduce the elapsed time further.

The PBKDF2 PIN verifier inside the encrypted database does not increase the
cost of incorrect offline guesses. AES-GCM authentication identifies a correct
derived key before that verifier is evaluated.

The current scrypt parameters are `N=2^16, r=8, p=1`. OWASP's current
equivalent minimum configurations use either `N=2^17, p=1` or
`N=2^16, p=2`. Increasing the work factor alone still cannot compensate for
the PIN's roughly 13.3 bits of entropy.

**Recommendation:** Generate a random 256-bit vault key and combine the PIN
with a device-bound secret stored in macOS Keychain. Alternatively, allow a
longer master passphrase. If a four-digit PIN remains mandatory, it should act
as a retry-limited activation secret backed by device-protected key material,
not as the only input to an offline key derivation.

References:

- [OWASP Password Storage Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html)
- [NIST SP 800-63B](https://pages.nist.gov/800-63-4/sp800-63b.html)

### High: The session cookie is shared across loopback ports

The application stores its bearer session token in a host cookie with
`Path=/` in [`app.py`](src/local_secrets_store/app.py#L438-L442).

Cookies are scoped by host, not by port. A different local HTTP service on
`127.0.0.1` can receive the cookie if the browser visits that service. The
service can then use the stolen token in a raw HTTP request to
`/api/secrets`. `HttpOnly` and `SameSite=Strict` do not prevent the other
server from receiving the cookie.

The exact `Host` and `Origin` checks are valuable browser protections, but
they do not stop a non-browser client that can set those headers after stealing
the bearer token. The secrets GET endpoint also does not require the CSRF
token.

**Recommendation:** Remove ambient cookie authentication. Return a bearer
token after unlock, retain it only in JavaScript memory, and send it in an
`Authorization` header on each API request.

Reference:

- [RFC 6265 section 8.5](https://www.rfc-editor.org/rfc/rfc6265#section-8.5)

### High: Locking does not scrub an open secret editor

The editor copies plaintext into input elements in
[`app.js`](src/local_secrets_store/static/app.js#L166-L176). The lock routine
clears the card array, but it does not reset or close the dialog.

This was reproduced in the browser with a shortened idle timeout:

- The server closed the in-memory SQLite database.
- The unlock view became active.
- The editor dialog remained open.
- Its secret and notes inputs retained their plaintext values.

**Recommendation:** Before showing the lock screen, explicitly blank every
sensitive input, reset and close the dialog, remove rendered secret nodes, and
preferably reload the page after the server confirms locking.

### Medium: Known-vulnerable cryptographic dependency is pinned

[`pyproject.toml`](pyproject.toml#L8) restricts `cryptography` to `<46`, and
the lockfile selects version `45.0.7`. The current release at review time is
`49.0.0`.

`pip-audit` reported four unique advisories affecting the runtime
`cryptography` dependency and one advisory affecting development-only
`pytest`. Some rows in the audit output were duplicates.

The certificate, ECC, PKCS7, and non-contiguous-buffer paths described by the
advisories are not exercised by the current AES-GCM implementation, so this
review did not demonstrate an exploit against the vault format. A secrets
manager should nevertheless not deliberately exclude available security
updates.

**Recommendation:**

- Use `cryptography>=49,<50`.
- Use `pytest>=9.0.3,<10`.
- Regenerate and audit `uv.lock`.

References:

- [cryptography release history](https://pypi.org/project/cryptography/)
- [PYSEC-2026-36](https://osv.dev/vulnerability/PYSEC-2026-36)
- [GHSA-537c-gmf6-5ccf](https://osv.dev/vulnerability/GHSA-537c-gmf6-5ccf)

### Medium: Multiple instances silently overwrite one another

The process does not acquire an exclusive filesystem lock before opening the
vault. Two instances can decrypt the same snapshot and independently replace
`vault.enc`.

The review reproduced this failure:

1. Two stores unlocked the same vault.
2. The first saved a secret named `first`.
3. The second saved a secret named `second`.
4. Reopening the vault contained only `second`.

The alternate-port launch option makes this possible even though the default
port normally prevents a second instance.

**Recommendation:** Acquire an exclusive OS file lock on
`~/.personal-creds/app.lock` for the process lifetime and refuse to start a
second instance.

### Medium: Failed persistence leaves the in-memory mutation committed

Create, update, and delete commit their SQLite transaction before calling
`_persist()`, as shown in
[`vault.py`](src/local_secrets_store/vault.py#L221-L230).

When persistence was forced to fail, the operation raised an error but the
supposedly failed secret remained in the in-memory database. A later successful
write could unexpectedly persist it; stopping the process first would lose it.

**Recommendation:** Apply a mutation to a cloned in-memory database, encrypt
and atomically persist that candidate, and swap it into the live state only
after file replacement succeeds.

### Medium: Every secret is sent to and retained by the browser

`GET /api/secrets` returns every name, username, secret, and note at once in
[`app.py`](src/local_secrets_store/app.py#L259-L265). The browser retains those
values in an array and event-handler closures even while their visual
representation is masked.

This increases the impact of browser extensions, developer tools, memory
inspection, or a future frontend vulnerability.

**Recommendation:** Return metadata from the list endpoint and fetch one
secret on demand for reveal, copy, or edit. Discard it promptly afterward.

### Low: Clipboard contents survive locking

The Copy action writes plaintext into the system clipboard but never clears
it. Other local applications may be able to retrieve it after the vault locks.

**Recommendation:** Warn users about clipboard persistence and conditionally
clear an unchanged clipboard value after 30 to 60 seconds.

### Low: A malformed vault can cause excessive resource use

The encrypted file is loaded without a file-size limit. Its header also
controls the scrypt cost before the AES-GCM authentication tag can be checked.
A modified file can request a cost as high as `2^20`, potentially causing
large memory allocation or a long delay before the app reports corruption.

**Recommendation:** Enforce a reasonable maximum vault size before reading,
accept only known KDF parameter sets, and implement explicit format migrations
for future parameter changes.

### Residual: Sensitive memory is not guaranteed to be erased

The SQLite pages, serialized database bytes, encryption key, PIN, and browser
strings exist in normal process memory while in use. Python immutable byte
strings cannot be reliably zeroed, and dropping references does not guarantee
immediate memory erasure.

**Recommendation:** Document this threat boundary. Stronger protection would
require a native component capable of locked memory, explicit zeroization, and
core-dump suppression.

## Security controls implemented well

- AES-256-GCM is used with fresh random 96-bit nonces and authenticated
  headers.
- The database is in memory while unlocked and only ciphertext is persisted.
- Writes use a temporary `0600` file, file `fsync`, atomic replacement, and
  directory `fsync`.
- The data directory is set to `0700`.
- The HTTP listener binds only to `127.0.0.1`.
- Exact host and mutation-origin checks reduce DNS rebinding and browser CSRF
  exposure.
- Session and CSRF tokens use cryptographically secure randomness and
  constant-time comparison.
- CSP, frame denial, `nosniff`, `no-store`, and referrer protections are set.
- User-controlled values are rendered through `textContent`; no dynamic HTML
  insertion was found.
- SQL statements use parameters.
- Request bodies and individual fields have length limits.
- Secret-bearing URLs and request data are not written to terminal logs.
- A server-side timer closes SQLite after inactivity.

Reference:

- [Cryptography AES-GCM documentation](https://cryptography.io/en/stable/hazmat/primitives/aead/)

## Validation performed

- Manual review of Python, HTML, CSS, JavaScript, tests, and the dependency
  lockfile.
- Browser reproduction of lock-time plaintext retention.
- Production scrypt timing benchmark.
- Concurrent-writer lost-update reproduction.
- Failed-persistence state-divergence reproduction.
- `pip-audit` dependency scan.
- Bandit static scan: zero source findings.
- Existing regression suite: 12 tests passed.
- Lockfile consistency check: passed.

No actual user credentials or production vault file were accessed during this
review.

## Recommended remediation order

1. Replace PIN-only key derivation with a longer passphrase or a
   Keychain-backed random vault key.
2. Replace cookie authentication with an in-memory authorization token.
3. Scrub and close all secret-bearing UI during lock.
4. Upgrade and re-audit runtime and development dependencies.
5. Add an exclusive process lock for the vault.
6. Make persistence transactional across memory and disk.
7. Fetch secrets individually and limit clipboard exposure.
8. Add malformed-file bounds, backup recovery, and memory-threat
   documentation.
