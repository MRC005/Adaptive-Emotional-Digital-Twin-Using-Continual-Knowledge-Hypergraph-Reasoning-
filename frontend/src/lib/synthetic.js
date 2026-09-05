/**
 * The label every page must show when it draws simulated people.
 *
 * No participant-level research record is distributed with this application.
 * The people on the Explorer, the two-person comparison and Twin Evolution are
 * written by
 * scripts/make_synthetic_demo_data.py from fixed constants and a seed.
 */
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

export function syntheticDataBanner(what = "The people shown here") {
  return `<div class="msg warn syn-banner" role="note">
    <span class="syn-tag">Synthetic demonstration data</span>
    <b>Not real participants.</b> ${esc(what)} are simulated, to demonstrate how
    the page works. They describe no real person and are not a research result.
    Forecasts drawn for them come from a simple illustrative stand-in, not the
    research model. No participant-level research record is distributed with
    this application.</div>`;
}
