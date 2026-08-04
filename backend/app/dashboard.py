"""backend/app/dashboard.py — the savings dashboard, served at GET /dashboard.

Self-contained on purpose: inline CSS/JS, no CDN. This host runs offline-hostile
(local Ollama, no guarantee of internet), and a blocked <script> would leave a
blank page rather than a degraded one.

Charts are hand-rolled CSS bars — no charting dependency. Both use categorical
slot 1 (blue), validated against both surfaces:
  light #2a78d6 on #fcfcfb · dark #3987e5 on #1a1a19
They are separate single-series charts, so re-using one hue is correct; a second
hue would imply a distinction that doesn't exist. (Aqua was the obvious second
slot and fails the 3:1 contrast check on the light surface anyway.)

Copy discipline: every figure states what it is measured against. The metric this
replaced reported "48,436 saved" against pasting the entire repo into a prompt —
impressive, unfalsifiable, and wrong. A number nobody can check is worse than no
number.
"""
from __future__ import annotations

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PromptForge — savings</title>
<style>
  :root {
    color-scheme: light;
    --plane:      #f9f9f7;
    --surface-1:  #fcfcfb;
    --text-1:     #0b0b0b;
    --text-2:     #52514e;
    --muted:      #898781;
    --grid:       #e1e0d9;
    --series-1:   #2a78d6;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --plane:     #0d0d0d;
      --surface-1: #1a1a19;
      --text-1:    #ffffff;
      --text-2:    #c3c2b7;
      --muted:     #898781;
      --grid:      #2c2c2a;
      --series-1:  #3987e5;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --plane:     #0d0d0d;
    --surface-1: #1a1a19;
    --text-1:    #ffffff;
    --text-2:    #c3c2b7;
    --muted:     #898781;
    --grid:      #2c2c2a;
    --series-1:  #3987e5;
  }

  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px 20px 40px;
    background: var(--plane); color: var(--text-1);
    font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 980px; margin: 0 auto; }
  h1 { font-size: 17px; margin: 0 0 2px; letter-spacing: -0.01em; }
  .sub { color: var(--text-2); margin: 0 0 16px; font-size: 13px; }

  /* two columns; collapses to one on narrow windows */
  .grid { display: grid; grid-template-columns: 1fr 1.35fr; gap: 12px; margin-bottom: 12px; }
  .grid.even { grid-template-columns: 1fr 1fr; }
  @media (max-width: 760px) { .grid, .grid.even { grid-template-columns: 1fr; } }

  .card {
    background: var(--surface-1); border: 1px solid var(--grid);
    border-radius: 12px; padding: 16px 18px; min-width: 0;
  }
  .hero .n {
    font-size: 38px; font-weight: 650; letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums; line-height: 1.05;
  }
  .hero .cap { color: var(--text-2); font-size: 13px; margin-top: 4px; }
  .note { color: var(--muted); font-size: 12px; margin-top: 8px; line-height: 1.45; }

  /* compact 3x2 stat grid, no nested cards */
  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px 10px; }
  .stat .n { font-size: 21px; font-weight: 620; font-variant-numeric: tabular-nums;
             line-height: 1.15; }
  .stat .l { color: var(--text-2); font-size: 12px; margin-top: 1px; }
  .stat .sm { color: var(--muted); font-size: 11px; }

  h2 { font-size: 13px; font-weight: 600; margin: 0 0 12px; color: var(--text-1); }
  .foot { color: var(--muted); font-size: 11.5px; margin-top: 10px; line-height: 1.45; }

  /* horizontal bars — 4px rounded data-end, anchored to the baseline */
  .row { display: grid; grid-template-columns: minmax(72px, 118px) 1fr 66px; gap: 8px;
         align-items: center; margin-bottom: 7px; }
  .row .name { color: var(--text-2); font-size: 13px; overflow: hidden;
               text-overflow: ellipsis; white-space: nowrap; }
  .track { background: var(--grid); border-radius: 4px; height: 20px; }
  .fill { background: var(--series-1); height: 100%;
          border-radius: 0 4px 4px 0; min-width: 3px; }
  .row .v { text-align: right; font-variant-numeric: tabular-nums;
            font-size: 13px; color: var(--text-1); }

  /* daily bars — 2px surface gap between adjacent bars */
  .days { display: flex; align-items: flex-end; gap: 2px; height: 120px;
          border-bottom: 1px solid var(--grid); padding-bottom: 0; }
  .day { flex: 1 1 0; background: var(--series-1); border-radius: 4px 4px 0 0;
         min-height: 2px; cursor: default; }
  .axis { display: flex; justify-content: space-between; color: var(--muted);
          font-size: 12px; margin-top: 6px; }

  .empty { color: var(--muted); font-size: 13.5px; padding: 6px 0 2px; }

  table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
  th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--grid); }
  th { color: var(--text-2); font-weight: 600; }
  td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
  details { margin-top: 4px; }
  summary { cursor: pointer; color: var(--text-2); font-size: 13.5px; padding: 6px 0; }

  #tip {
    position: fixed; pointer-events: none; opacity: 0; transition: opacity .1s;
    background: var(--text-1); color: var(--surface-1); font-size: 12.5px;
    padding: 5px 9px; border-radius: 6px; white-space: nowrap; z-index: 10;
  }
</style>
</head>
<body>
<div class="wrap">
  <h1>PromptForge — work Copilot never had to do</h1>
  <p class="sub">Turns answered locally, refined before sending, or served from cache.</p>

  <div class="grid">
    <div class="card hero">
      <div class="n" id="hero">—</div>
      <div class="cap" id="heroCap">tokens PromptForge read for you</div>
      <div class="note">
        Rough estimate. We count the code PromptForge read for you. The real saving is
        probably higher — we don't count Copilot's replies.
      </div>
    </div>

    <div class="card">
      <div class="stats" style="grid-template-columns:repeat(3,1fr)">
        <div class="stat"><div class="n" id="calls">—</div><div class="l">calls avoided</div></div>
        <div class="stat"><div class="n" id="chats">—</div><div class="l">chats</div></div>
        <div class="stat">
          <div class="n" id="projects">—</div>
          <div class="l">projects <span class="sm" id="projectsNote"></span></div>
        </div>
        <div class="stat"><div class="n" id="answered">—</div><div class="l">answered locally</div></div>
        <div class="stat"><div class="n" id="refined">—</div><div class="l">refined first</div></div>
        <div class="stat"><div class="n" id="cached">—</div><div class="l">from cache</div></div>
        <div class="stat"><div class="n" id="reasoning">—</div><div class="l">reasoning</div></div>
        <div class="stat"><div class="n" id="optimization">—</div><div class="l">prompt building</div></div>
        <div class="stat">
          <div class="n" id="cost">—</div>
          <div class="l">if paid per token <span class="sm" id="costBasis"></span></div>
        </div>
      </div>
    </div>
  </div>

  <div class="grid even">
    <div class="card">
      <h2>Tokens avoided by project</h2>
      <div id="byProject"></div>
    </div>

    <div class="card">
      <h2>Last 30 days</h2>
      <div id="byDay"></div>
      <div class="foot" id="dayFoot"></div>
    </div>
  </div>

  <div class="card">
    <details>
      <summary>Table view</summary>
      <table>
        <thead><tr><th>Project</th><th class="n">Calls avoided</th><th class="n">Tokens</th></tr></thead>
        <tbody id="tbody"></tbody>
      </table>
    </details>
  </div>
</div>
<div id="tip"></div>

<script>
const fmt = n => (n || 0).toLocaleString();
const tip = document.getElementById('tip');
function bindTip(el, text) {
  el.addEventListener('mousemove', e => {
    tip.textContent = text;
    tip.style.opacity = 1;
    tip.style.left = Math.min(e.clientX + 12, innerWidth - tip.offsetWidth - 8) + 'px';
    tip.style.top = (e.clientY - 32) + 'px';
  });
  el.addEventListener('mouseleave', () => { tip.style.opacity = 0; });
}

fetch('/savings').then(r => r.json()).then(d => {
  document.getElementById('hero').textContent = fmt(d.contextTokens);
  document.getElementById('calls').textContent = fmt(d.copilotCallsAvoided);
  document.getElementById('chats').textContent = fmt(d.chats);
  document.getElementById('projects').textContent = fmt(d.byProject.length);
  // Say "+ earlier" rather than folding untagged work into the project count —
  // those turns could have come from any project, so claiming one would invent a fact.
  document.getElementById('projectsNote').textContent =
    (d.earlier && d.earlier.tokens) ? '+ earlier' : '';
  document.getElementById('answered').textContent = fmt(d.answeredLocally);
  document.getElementById('refined').textContent = fmt(d.refinedLocally);
  document.getElementById('cached').textContent = fmt(d.cacheHits);
  document.getElementById('reasoning').textContent = fmt(d.byType && d.byType.reasoning);
  document.getElementById('optimization').textContent = fmt(d.byType && d.byType.promptOptimization);
  if (d.costEstimate) {
    document.getElementById('cost').textContent = '$' + (d.costEstimate.usd || 0).toFixed(2);
    // Name the rate next to the number. Copilot is a flat subscription, so this
    // is a hypothetical — an unlabelled dollar figure would read as money banked.
    document.getElementById('costBasis').textContent =
      '· ' + d.costEstimate.basis + ' ($' + d.costEstimate.perMTok + '/M)';
  }
  if (d.since) document.getElementById('heroCap').textContent =
    'tokens of code Copilot never had to read — since ' + d.since;

  // ---- by project
  const host = document.getElementById('byProject');
  const rows = d.byProject.slice();
  if (d.earlier && d.earlier.tokens) {
    rows.push({ name: 'earlier (untagged)', tokens: d.earlier.tokens, calls: d.earlier.calls });
  }
  if (!rows.length) {
    host.innerHTML = '<div class="empty">Nothing recorded yet.</div>';
  } else {
    const max = Math.max(...rows.map(r => r.tokens), 1);
    host.innerHTML = '';
    rows.forEach(r => {
      const row = document.createElement('div');
      row.className = 'row';
      row.innerHTML =
        '<div class="name"></div><div class="track"><div class="fill"></div></div><div class="v"></div>';
      row.querySelector('.name').textContent = r.name;
      row.querySelector('.v').textContent = fmt(r.tokens);
      row.querySelector('.fill').style.width = Math.max(2, (r.tokens / max) * 100) + '%';
      bindTip(row, r.name + ' — ' + fmt(r.tokens) + ' tokens, ' + r.calls + ' calls avoided');
      host.appendChild(row);
    });
  }

  // ---- by day
  const dayHost = document.getElementById('byDay');
  if (!d.byDay.length) {
    dayHost.innerHTML = '<div class="empty">Per-day tracking starts today — ' +
      'earlier totals had no date recorded, so they sit in "earlier (untagged)" above.</div>';
  } else {
    const max = Math.max(...d.byDay.map(x => x.tokens), 1);
    const bars = document.createElement('div');
    bars.className = 'days';
    d.byDay.forEach(x => {
      const b = document.createElement('div');
      b.className = 'day';
      b.style.height = Math.max(2, (x.tokens / max) * 100) + '%';
      bindTip(b, x.date + ' — ' + fmt(x.tokens) + ' tokens');
      bars.appendChild(b);
    });
    const axis = document.createElement('div');
    axis.className = 'axis';
    axis.innerHTML = '<span></span><span></span>';
    axis.children[0].textContent = d.byDay[0].date;
    axis.children[1].textContent = d.byDay[d.byDay.length - 1].date;
    dayHost.innerHTML = '';
    dayHost.append(bars, axis);
  }
  // Permanent footnote, not just an empty state: once today has a bar the chart
  // stops looking empty, but the untagged tokens still aren't in it and the
  // totals would otherwise look like they don't add up.
  if (d.earlier && d.earlier.tokens) {
    document.getElementById('dayFoot').textContent =
      fmt(d.earlier.tokens) + ' tokens predate daily tracking and are not shown here.';
  }

  // ---- table view (identity never by colour alone)
  const tb = document.getElementById('tbody');
  tb.innerHTML = '';
  rows.forEach(r => {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td></td><td class="n"></td><td class="n"></td>';
    tr.children[0].textContent = r.name;
    tr.children[1].textContent = fmt(r.calls);
    tr.children[2].textContent = fmt(r.tokens);
    tb.appendChild(tr);
  });
}).catch(e => {
  document.getElementById('hero').textContent = '—';
  document.getElementById('heroCap').textContent = 'Could not reach the backend: ' + e;
});
</script>
</body>
</html>
"""
