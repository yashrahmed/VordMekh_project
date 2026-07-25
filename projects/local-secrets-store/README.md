# Local Secrets Store

A small, local-only secrets vault with an HTML interface. The app keeps its
SQLite database in memory while unlocked and persists only an authenticated,
encrypted snapshot at:

```text
~/.personal-creds/vault.enc
```

## Start it

The only prerequisite is [uv](https://docs.astral.sh/uv/).

```bash
cd projects/local-secrets-store
uv run local-secrets-store
```

The first run installs the project's isolated dependencies, starts a listener
on `127.0.0.1:8765`, and opens the interface in your default browser. If that
port is busy, run `uv run local-secrets-store --port 8766`.

## Bootstrap your vault

1. Start the app with the command above.
2. On the first screen, enter and confirm a four-digit PIN.
3. Select **Add secret** and manually enter each credential or secret.
4. Use **Lock** when finished, then stop the app with `Ctrl+C`.
5. On later runs, use the same command and PIN to reopen the vault.

Every add, edit, and delete is saved immediately. The interface automatically
locks after 15 minutes without an authenticated request.

## What is stored

- The live SQLite database exists only in the app process's memory.
- The data file is encrypted as a whole with AES-256-GCM.
- The encryption key is derived from the four-digit PIN with scrypt and a
  random salt.
- The database contains a salted PIN verifier, never the PIN itself.
- `~/.personal-creds` is set to owner-only permissions (`0700`) and
  `vault.enc` to owner read/write only (`0600`).
- Writes use a new random nonce and an atomic file replacement.

The app binds only to the IPv4 loopback address and makes no outbound network
requests. It uses a short-lived, HTTP-only browser session, request-origin
checks, a CSRF token, and a restrictive content security policy.

## Why the HTML is not a standalone file

A standalone browser page cannot reliably and safely write directly to
`~/.personal-creds`; the browser sandbox is specifically designed to prevent
that. This project therefore uses a small local Python process for filesystem
and encryption access. The UI itself is plain HTML, CSS, and JavaScript with no
remote assets.

## Important security limits

A four-digit PIN has only 10,000 possible values. Scrypt slows guessing, but it
cannot give a short PIN the strength of a full master password. Full-disk
encryption, a strong computer login password, and physical device security
remain important. This utility has not received an independent security audit.

To back up the vault, lock or stop the app and copy
`~/.personal-creds/vault.enc` to protected storage. To start fresh without
immediately deleting the old vault, stop the app and move the directory:

```bash
mv ~/.personal-creds ~/.personal-creds.backup
```

## Development

```bash
uv run pytest
```
