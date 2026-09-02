/**
 * Accounts and persistent twins.
 *
 * ARCHITECTURE, AND WHY
 *
 * The frontend is a static Vite build on Vercel with no serverless functions,
 * and the Render API is deliberately stateless -- it accepts no identifier and
 * stores nothing, which is what lets it claim there is no history to leak. So
 * persistence needs a new tier, and writing session handling and password
 * storage by hand would be the wrong answer to that.
 *
 * This talks to Supabase over plain HTTP: GoTrue for auth, PostgREST for data.
 * No SDK, so no new dependency and no bundle growth, and every request is
 * visible in this file.
 *
 * OWNERSHIP IS ENFORCED BY THE DATABASE, NOT BY THIS FILE. Row-level security
 * (see supabase/schema.sql) restricts every row to auth.uid(). The client never
 * sends a user id, and could not read another user's rows even if it tried --
 * which matters, because a client-supplied id is exactly the thing that must
 * never be trusted.
 *
 * UNCONFIGURED IS A SUPPORTED STATE. With no keys present the app runs exactly
 * as it did before: local-only, anonymous, fully functional. A panel
 * demonstration must never depend on a third party being reachable, so the
 * account layer is additive and its absence is silent.
 *
 * TOKEN STORAGE. The access token lives in localStorage. That is the normal
 * choice for a static SPA with no cookie-issuing origin, and it accepts an XSS
 * risk in exchange: any script injected into this origin could read it. The
 * mitigation is that this page loads no third-party script and inlines nothing
 * from user input. Stated here rather than left implicit.
 */

const URL_ = (import.meta.env?.VITE_SUPABASE_URL || "").replace(/\/$/, "");
const KEY = import.meta.env?.VITE_SUPABASE_ANON_KEY || "";

const SESSION_KEY = "aedt.session";
const TABLE = "twin_events";

/** Configured means: both values present. Anything else is local-only mode. */
export function accountsEnabled() {
  return Boolean(URL_ && KEY);
}

/* ------------------------------------------------------------- session */
function readSession() {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    const s = JSON.parse(raw);
    if (!s?.access_token) return null;
    // expires_at is seconds since epoch, per GoTrue
    if (s.expires_at && s.expires_at * 1000 < Date.now()) return null;
    return s;
  } catch { return null; }
}

function writeSession(s) {
  try {
    if (s) localStorage.setItem(SESSION_KEY, JSON.stringify(s));
    else localStorage.removeItem(SESSION_KEY);
  } catch { /* private mode: the session simply does not survive reload */ }
}

export function currentUser() {
  const s = readSession();
  return s ? { id: s.user?.id, email: s.user?.email } : null;
}

export function signedIn() { return Boolean(readSession()); }

/* ----------------------------------------------------------------- auth */
async function auth(path, body) {
  const res = await fetch(`${URL_}/auth/v1/${path}`, {
    method: "POST",
    headers: { apikey: KEY, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    // GoTrue's messages are user-facing enough to show, but not always kind
    const msg = data?.msg || data?.error_description || data?.message
             || `Sign-in failed (${res.status})`;
    throw new Error(msg);
  }
  return data;
}

export async function createAccount(email, password) {
  if (!accountsEnabled()) throw new Error("Accounts are not configured on this deployment.");
  if (!email || !password) throw new Error("An email and password are required.");
  if (password.length < 8) throw new Error("Use at least 8 characters.");
  const d = await auth("signup", { email, password });
  // With email confirmation on, signup returns a user but NO session. Say so
  // rather than appearing to succeed and then silently failing to save.
  if (!d.access_token) {
    return { confirmationRequired: true, email };
  }
  writeSession(d);
  return { confirmationRequired: false, email };
}

export async function signIn(email, password) {
  if (!accountsEnabled()) throw new Error("Accounts are not configured on this deployment.");
  const d = await auth("token?grant_type=password", { email, password });
  writeSession(d);
  return currentUser();
}

export async function signOut() {
  const s = readSession();
  writeSession(null);
  if (s && accountsEnabled()) {
    // best effort: the local session is already gone either way
    try {
      await fetch(`${URL_}/auth/v1/logout`, {
        method: "POST",
        headers: { apikey: KEY, Authorization: `Bearer ${s.access_token}` },
      });
    } catch { /* offline sign-out still signs out locally */ }
  }
}

/* ------------------------------------------------------------- storage */
async function rest(path, { method = "GET", body, prefer } = {}) {
  const s = readSession();
  if (!s) throw new Error("Not signed in.");
  const headers = {
    apikey: KEY,
    Authorization: `Bearer ${s.access_token}`,
    "Content-Type": "application/json",
  };
  if (prefer) headers.Prefer = prefer;
  const res = await fetch(`${URL_}/rest/v1/${path}`, {
    method, headers, body: body ? JSON.stringify(body) : undefined,
  });
  if (res.status === 401) { writeSession(null); throw new Error("Your session expired. Sign in again."); }
  if (!res.ok) throw new Error((await res.text()) || `Request failed (${res.status})`);
  return res.status === 204 ? null : res.json().catch(() => null);
}

/**
 * Save one event.
 *
 * `user_id` is NOT sent. The column defaults to auth.uid() in the database, so
 * the row's owner is decided by the verified token rather than by anything the
 * browser claims.
 */
export async function saveEvent(ev) {
  return rest(TABLE, {
    method: "POST",
    prefer: "return=minimal",
    body: { event_id: ev.eventId, occurred_at: ev.timestamp, payload: ev },
  });
}

/** Every event belonging to the signed-in user, oldest first. */
export async function loadEvents() {
  const rows = await rest(`${TABLE}?select=payload,occurred_at&order=occurred_at.asc`);
  return (rows || []).map((r) => r.payload).filter(Boolean);
}

export async function deleteEvent(eventId) {
  return rest(`${TABLE}?event_id=eq.${encodeURIComponent(eventId)}`,
              { method: "DELETE", prefer: "return=minimal" });
}

/** Delete every event for this user. The account itself is kept. */
export async function deleteAllEvents() {
  const s = readSession();
  if (!s?.user?.id) throw new Error("Not signed in.");
  return rest(`${TABLE}?user_id=eq.${s.user.id}`,
              { method: "DELETE", prefer: "return=minimal" });
}

/**
 * Delete the account and everything in it.
 *
 * GoTrue does not allow a user to delete themselves with an anon key, so this
 * calls a SECURITY DEFINER function defined in schema.sql, which deletes the
 * caller's rows and then the caller. It can only ever act on auth.uid().
 */
export async function deleteAccount() {
  await rest("rpc/delete_own_account", { method: "POST", body: {} });
  writeSession(null);
}
