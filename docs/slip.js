// Slip Builder storage, shared by every page. The slip lives only in this browser (localStorage);
// it's a convenience list for typing picks into PrizePicks by hand, nothing is sent anywhere.
const SLIP_KEY = "nhlSlip";
const SLIP_MAX = 6;   // PrizePicks' largest entry size

function slipLoad() {
  try { return JSON.parse(localStorage.getItem(SLIP_KEY)) || []; } catch (e) { return []; }
}
function slipSave(items) {
  try { localStorage.setItem(SLIP_KEY, JSON.stringify(items)); } catch (e) {}
  slipBadge();
}
const slipKey = p => String(p.pp_proj_id || `${p.game_date}|${p.player}|${p.category}`);
const slipHas = p => slipLoad().some(s => s.key === slipKey(p));

function slipToggle(p) {
  const items = slipLoad(), key = slipKey(p);
  const i = items.findIndex(s => s.key === key);
  if (i >= 0) items.splice(i, 1);
  else items.push({key, pp_proj_id: p.pp_proj_id, player: p.player, team: p.team, opp: p.opp, category: p.category,
                   line: p.line, side: p.side, model_side: p.side, prob: p.prob, prob_cal: p.prob_cal,
                   start_time: p.start_time, game_date: p.game_date, added_at: new Date().toISOString()});
  slipSave(items);
  return i < 0;   // true = added
}

function slipBadge() {
  const el = document.getElementById("slipCount");
  if (!el) return;
  const n = slipLoad().length;
  el.textContent = n ? ` (${n})` : "";
}
document.addEventListener("DOMContentLoaded", slipBadge);
window.addEventListener("storage", e => { if (e.key === SLIP_KEY) slipBadge(); });
