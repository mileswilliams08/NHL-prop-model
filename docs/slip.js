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
                   line: p.line, side: p.side, model_side: p.side, m: p.model_mean, prob: p.prob, prob_cal: p.prob_cal,
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

// Live board overlay. The 30-minute line tracker writes data/board.json with each pick's CURRENT line,
// repriced from the same model mean. If it's newer than props.json and a line moved, the pick shows the
// current line/side/probability, and keeps where the line opened so the page can flag the move.
async function loadBoard() {
  try { return await (await fetch("data/board.json", {cache: "no-store"})).json(); } catch (e) { return null; }
}
function applyBoard(props, board) {
  if (!props || !props.picks) return 0;
  const fresh = board && board.lines && new Date(board.updated_at) > new Date(props.generated_at);
  let n = 0;
  for (const p of props.picks) {
    if (p.skip_reason || p.side == null) continue;
    // first line of the day: from an earlier model run (predict.py), else this run's own line
    if (p.open_line == null) Object.assign(p, {open_line: p.line, open_side: p.side});
    const b = fresh && board.lines[`${p.game_date}|${p.pp_player_id}|${p.category}`];
    if (b && Number(b.line) !== Number(p.line)) {
      Object.assign(p, {line: b.line, side: b.side, prob: b.prob, prob_cal: b.prob_cal, p_over: b.p_over,
                        p_under: b.p_under, pp_proj_id: b.pp_proj_id || p.pp_proj_id, live: true});
      delete p.shadow;   // the LightGBM shadow was priced at the old line
      n++;
    }
  }
  return n;
}
// "Opened Under 23.5 → side switched (projection 23.4 unchanged)", or "" if the line never moved.
function moveNote(p) {
  if (p.open_line == null || Number(p.open_line) === Number(p.line)) return "";
  const S = s => s === "over" ? "Over" : "Under";
  return `Opened ${S(p.open_side)} ${p.open_line}` + (p.open_side !== p.side
    ? ` → side switched (projection ${Number(p.model_mean).toFixed(1)} unchanged)` : "");
}

// Turn a <select> into tap-to-switch buttons. The select stays (hidden) as the source of truth, so the
// page's existing .value reads and "change" listeners keep working.
function segmented(sel) {
  if (!sel) return;
  const box = document.createElement("span");
  box.className = "seg";
  const sync = () => box.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.v === sel.value));
  [...sel.options].forEach(o => {
    const b = document.createElement("button");
    b.type = "button"; b.dataset.v = o.value; b.textContent = o.textContent;
    b.addEventListener("click", () => { if (sel.value !== o.value) { sel.value = o.value; sel.dispatchEvent(new Event("change")); } sync(); });
    box.appendChild(b);
  });
  sel.style.display = "none";
  sel.after(box);
  sel.addEventListener("change", sync);
  sync();
  return sync;
}
