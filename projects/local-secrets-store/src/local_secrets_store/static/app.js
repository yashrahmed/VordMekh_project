"use strict";

const views = {
  loading: document.querySelector("#loading-view"),
  setup: document.querySelector("#setup-view"),
  unlock: document.querySelector("#unlock-view"),
  vault: document.querySelector("#vault-view"),
};

const setupForm = document.querySelector("#setup-form");
const unlockForm = document.querySelector("#unlock-form");
const secretForm = document.querySelector("#secret-form");
const secretDialog = document.querySelector("#secret-dialog");
const list = document.querySelector("#secret-list");
const emptyState = document.querySelector("#empty-state");
const message = document.querySelector("#message");
let csrfToken = null;
let secrets = [];
let idleTimer = null;
let idleTimeoutMs = 15 * 60 * 1000;

function showView(name) {
  Object.entries(views).forEach(([key, element]) => {
    element.classList.toggle("hidden", key !== name);
  });
}

function showMessage(text, isError = false) {
  message.textContent = text;
  message.classList.remove("hidden", "error");
  if (isError) message.classList.add("error");
  window.clearTimeout(showMessage.timer);
  showMessage.timer = window.setTimeout(() => message.classList.add("hidden"), 4000);
}

async function request(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.body) headers["Content-Type"] = "application/json";
  if (csrfToken && options.method && options.method !== "GET") {
    headers["X-Vault-CSRF"] = csrfToken;
  }
  const response = await fetch(path, {
    ...options,
    headers,
    credentials: "same-origin",
  });
  let body = {};
  try {
    body = await response.json();
  } catch {
    body = { error: "The local app returned an unreadable response." };
  }
  if (!response.ok) {
    if (response.status === 401 && path !== "/api/unlock") {
      csrfToken = null;
      showView("unlock");
    }
    throw new Error(body.error || "Something went wrong.");
  }
  resetIdleTimer();
  return body;
}

async function start() {
  try {
    const status = await request("/api/status");
    idleTimeoutMs = status.idleTimeoutSeconds * 1000;
    if (!status.initialized) {
      showView("setup");
    } else if (!status.unlocked) {
      showView("unlock");
    } else {
      csrfToken = status.csrfToken;
      await openVault();
    }
  } catch (error) {
    showView("unlock");
    showMessage(error.message, true);
  }
}

async function openVault() {
  const body = await request("/api/secrets");
  secrets = body.secrets;
  renderSecrets();
  showView("vault");
}

function renderSecrets() {
  list.replaceChildren();
  emptyState.classList.toggle("hidden", secrets.length !== 0);
  secrets.forEach((item) => list.append(buildSecretCard(item)));
}

function buildSecretCard(item) {
  const card = document.createElement("article");
  card.className = "secret-card";

  const heading = document.createElement("div");
  heading.className = "card-heading";
  const title = document.createElement("h3");
  title.textContent = item.name;
  const edit = document.createElement("button");
  edit.type = "button";
  edit.className = "secondary compact";
  edit.textContent = "Edit";
  edit.addEventListener("click", () => openSecretDialog(item));
  heading.append(title, edit);
  card.append(heading);

  if (item.username) {
    const usernameLabel = document.createElement("span");
    usernameLabel.className = "field-label";
    usernameLabel.textContent = "Username";
    const username = document.createElement("p");
    username.className = "field-value";
    username.textContent = item.username;
    card.append(usernameLabel, username);
  }

  const secretLabel = document.createElement("span");
  secretLabel.className = "field-label";
  secretLabel.textContent = "Secret";
  const secretRow = document.createElement("div");
  secretRow.className = "card-secret-row";
  const secretValue = document.createElement("code");
  secretValue.textContent = item.secret ? "••••••••••••" : "(empty)";
  let revealed = false;
  const reveal = document.createElement("button");
  reveal.type = "button";
  reveal.className = "text-button";
  reveal.textContent = "Show";
  reveal.addEventListener("click", () => {
    revealed = !revealed;
    secretValue.textContent = revealed ? (item.secret || "(empty)") : (item.secret ? "••••••••••••" : "(empty)");
    reveal.textContent = revealed ? "Hide" : "Show";
  });
  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "text-button";
  copy.textContent = "Copy";
  copy.disabled = !item.secret;
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(item.secret);
      showMessage("Secret copied.");
    } catch {
      showMessage("Your browser did not allow clipboard access.", true);
    }
  });
  secretRow.append(secretValue, reveal, copy);
  card.append(secretLabel, secretRow);

  if (item.notes) {
    const notesLabel = document.createElement("span");
    notesLabel.className = "field-label";
    notesLabel.textContent = "Notes";
    const notes = document.createElement("p");
    notes.className = "notes";
    notes.textContent = item.notes;
    card.append(notesLabel, notes);
  }
  return card;
}

function openSecretDialog(item = null) {
  document.querySelector("#dialog-title").textContent = item ? "Edit secret" : "Add secret";
  document.querySelector("#secret-id").value = item ? String(item.id) : "";
  document.querySelector("#secret-name").value = item?.name || "";
  document.querySelector("#secret-username").value = item?.username || "";
  document.querySelector("#secret-value").value = item?.secret || "";
  document.querySelector("#secret-value").type = "password";
  document.querySelector("#toggle-form-secret").textContent = "Show";
  document.querySelector("#secret-notes").value = item?.notes || "";
  document.querySelector("#delete-secret").classList.toggle("hidden", !item);
  secretDialog.showModal();
  document.querySelector("#secret-name").focus();
}

setupForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const pin = document.querySelector("#setup-pin").value;
  const confirmPin = document.querySelector("#confirm-pin").value;
  if (pin !== confirmPin) {
    showMessage("The PINs do not match.", true);
    return;
  }
  try {
    const body = await request("/api/initialize", {
      method: "POST",
      body: JSON.stringify({ pin }),
    });
    csrfToken = body.csrfToken;
    setupForm.reset();
    await openVault();
  } catch (error) {
    showMessage(error.message, true);
  }
});

unlockForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const pinInput = document.querySelector("#unlock-pin");
    const body = await request("/api/unlock", {
      method: "POST",
      body: JSON.stringify({ pin: pinInput.value }),
    });
    csrfToken = body.csrfToken;
    pinInput.value = "";
    await openVault();
  } catch (error) {
    document.querySelector("#unlock-pin").select();
    showMessage(error.message, true);
  }
});

secretForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const id = document.querySelector("#secret-id").value;
  const payload = {
    name: document.querySelector("#secret-name").value,
    username: document.querySelector("#secret-username").value,
    secret: document.querySelector("#secret-value").value,
    notes: document.querySelector("#secret-notes").value,
  };
  try {
    await request(id ? `/api/secrets/${id}` : "/api/secrets", {
      method: id ? "PUT" : "POST",
      body: JSON.stringify(payload),
    });
    secretDialog.close();
    await openVault();
    showMessage(id ? "Secret updated." : "Secret added.");
  } catch (error) {
    showMessage(error.message, true);
  }
});

document.querySelector("#add-button").addEventListener("click", () => openSecretDialog());
document.querySelector("#cancel-dialog").addEventListener("click", () => secretDialog.close());
document.querySelector("#close-dialog").addEventListener("click", () => secretDialog.close());
document.querySelector("#toggle-form-secret").addEventListener("click", () => {
  const input = document.querySelector("#secret-value");
  const showing = input.type === "text";
  input.type = showing ? "password" : "text";
  document.querySelector("#toggle-form-secret").textContent = showing ? "Show" : "Hide";
});
document.querySelector("#delete-secret").addEventListener("click", async () => {
  const id = document.querySelector("#secret-id").value;
  const item = secrets.find((candidate) => String(candidate.id) === id);
  if (!item || !window.confirm(`Delete “${item.name}”? This cannot be undone.`)) return;
  try {
    await request(`/api/secrets/${id}`, { method: "DELETE" });
    secretDialog.close();
    await openVault();
    showMessage("Secret deleted.");
  } catch (error) {
    showMessage(error.message, true);
  }
});

document.querySelector("#lock-button").addEventListener("click", lockVault);

async function lockVault() {
  if (!csrfToken) {
    showView("unlock");
    return;
  }
  try {
    await request("/api/lock", { method: "POST" });
  } catch {
    // The session may already have expired; the UI should still return to locked state.
  }
  csrfToken = null;
  secrets = [];
  renderSecrets();
  window.clearTimeout(idleTimer);
  showView("unlock");
  document.querySelector("#unlock-pin").focus();
}

function resetIdleTimer() {
  if (!csrfToken) return;
  window.clearTimeout(idleTimer);
  idleTimer = window.setTimeout(lockVault, idleTimeoutMs);
}

start();
