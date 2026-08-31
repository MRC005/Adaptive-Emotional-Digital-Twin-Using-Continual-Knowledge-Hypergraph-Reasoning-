/**
 * Theme: light by default, system-aware, manually switchable, persisted.
 *
 * Three states, not two. "system" stamps nothing on the root element and lets
 * prefers-color-scheme decide; "light" and "dark" stamp data-theme and win over
 * the OS in both directions. The CSS defines the full light palette on bare
 * :root, so a page that never runs this module still renders correctly.
 *
 * Storage can throw outright in a private window or with site data blocked, so
 * every read and write is guarded and the page works with no stored value.
 */
const KEY = "aedt.theme";
export const THEMES = ["light", "dark", "system"];

function read() {
  try {
    const v = localStorage.getItem(KEY);
    return THEMES.includes(v) ? v : "system";
  } catch {
    return "system";
  }
}

function write(v) {
  try { localStorage.setItem(KEY, v); } catch { /* nothing to do; not fatal */ }
}

let current = read();
const listeners = new Set();

/** Apply to the document. "system" removes the stamp so the media query rules. */
function apply(v) {
  const root = document.documentElement;
  if (v === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", v);
}

/** What the viewer actually sees right now, after resolving "system". */
export function effectiveTheme() {
  if (current !== "system") return current;
  try {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  } catch {
    return "light";
  }
}

export function getTheme() { return current; }

export function setTheme(v) {
  current = THEMES.includes(v) ? v : "system";
  apply(current);
  write(current);
  for (const fn of listeners) fn(effectiveTheme());
}

/** Called when the EFFECTIVE theme changes, including OS changes under "system". */
export function onThemeChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function initTheme() {
  apply(current);
  try {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const notify = () => { if (current === "system") for (const fn of listeners) fn(effectiveTheme()); };
    mq.addEventListener ? mq.addEventListener("change", notify) : mq.addListener(notify);
  } catch { /* no matchMedia: the stamped value still applies */ }
  return current;
}

/** Read a CSS custom property, so canvas drawing follows the same tokens as the CSS. */
export function token(name, fallback = "#000") {
  try {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  } catch {
    return fallback;
  }
}
