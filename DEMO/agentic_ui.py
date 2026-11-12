#!/usr/bin/env python3
"""Animated planner-vs-researchers demo. LLMAD as the defence, on or off.

Draws the pipeline as a graph and animates each injection round:

    planner ──► researcher A ──┐
            ──► researcher B ──┼──► aggregator ──► objective
            ──► researcher C ──┘

A probe arrow flies from the planner to whichever researcher it is attacking,
labelled with the technique. Researchers settle into a colour: green held
(refused), amber partial (answered, nothing usable), red bypassed.

Undefended, the planner escalates on every refusal and breaks through on the
third researcher. The bypass path lights up to the objective.

With LLMAD on every researcher nothing refuses, so the planner never escalates.
A loop arc appears between it and the researcher it is stuck on, the round
counter climbs, and the aggregator and objective stay dark.

Answers are capped short so the contrast is readable from the back of a room.
Nothing here is scripted: text, refusal judgements, yields and token counts are
live. Same containment as the other UIs -- loopback, token-gated, nothing
written to disk.

Usage
-----
  ./agentic_ui.py --open
  ./agentic_ui.py --port 9100 --target-tokens 110 --max-rounds 8
"""
from __future__ import annotations

import argparse
import json
import secrets
import sys
import threading
import webbrowser
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import agent_graph  # noqa: E402
import demo  # noqa: E402
import webui  # noqa: E402

# NOTE: raw string. The page contains JavaScript regex literals and escape
# sequences; without the r-prefix Python eats them before the browser sees them
# -- a `\n` inside a regex literal becomes a real newline and the whole script
# dies with "Invalid regular expression: missing /", taking the UI with it.
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLMAD defence — planner vs researchers</title>
<style>
  :root { --bg:#0b0d10; --panel:#14181d; --edge:#242b33; --ink:#e8edf2; --dim:#8b97a4;
          --red:#ff5f56; --grn:#39d353; --yel:#f0b429; --cyn:#4cc3ff; --vio:#b388ff; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:16px/1.5 ui-sans-serif,-apple-system,"SF Pro Text",Segoe UI,sans-serif; }
  header { padding:13px 22px; border-bottom:1px solid var(--edge); display:flex;
           gap:16px; align-items:baseline; flex-wrap:wrap; }
  h1 { font-size:17px; margin:0; font-weight:600; }
  .meta { color:var(--dim); font-size:12.5px;
          font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  main { padding:16px 22px 36px; max-width:1600px; margin:0 auto; }
  .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:12px; }
  input[type=text] { flex:1; min-width:300px; background:var(--panel); color:var(--ink);
                     border:1px solid var(--edge); border-radius:9px; padding:11px 13px; font-size:15px; }
  input[type=text]:focus { outline:2px solid var(--cyn); outline-offset:-1px; }
  select, .ghost { background:var(--panel); color:var(--ink); border:1px solid var(--edge);
                   border-radius:9px; padding:10px 13px; font-size:14px; cursor:pointer; }
  .go { background:var(--cyn); color:#04121c; border:0; border-radius:9px; padding:11px 22px;
        font-size:15px; font-weight:650; cursor:pointer; }
  .go[disabled] { opacity:.45; cursor:default; }
  .cols { display:grid; grid-template-columns:1.5fr 1fr; gap:14px; }
  @media (max-width:1150px) { .cols { grid-template-columns:1fr; } }
  .card { background:var(--panel); border:1px solid var(--edge); border-radius:13px; padding:14px 16px; }
  .card h2 { font-size:12.5px; margin:0 0 8px; text-transform:uppercase;
             letter-spacing:.5px; color:var(--dim); }
  svg { width:100%; height:360px; }
  .nbox { fill:#1b2129; stroke:#2c3540; stroke-width:1.5; rx:11; transition:fill .35s,stroke .35s; }
  .nbox.active { stroke:var(--cyn); stroke-width:3; }
  .nbox.held { stroke:var(--grn); fill:#101d14; }
  .nbox.partial { stroke:var(--yel); fill:#211b0d; }
  .nbox.bypassed { stroke:var(--red); fill:#231112; }
  .nbox.lit { stroke:var(--red); fill:#231112; }
  .nbox.disabled { stroke:#2a323c; stroke-dasharray:6 5; fill:#101317; opacity:.45; }
  .toggles { display:inline-flex; gap:10px; align-items:center; }
  .toggles label { color:var(--dim); font-size:13.5px; display:inline-flex;
                   align-items:center; gap:5px; cursor:pointer; }
  .ntitle { font:600 14px ui-sans-serif,sans-serif; fill:var(--ink); }
  .nstat { font:11.5px ui-monospace,Menlo,monospace; fill:var(--dim); }
  .wire { stroke:#2a323c; stroke-width:2; fill:none; transition:stroke .4s; }
  .wire.hot { stroke:var(--red); stroke-width:3; }
  .probe { stroke:var(--cyn); stroke-width:3; fill:none; stroke-dasharray:8 8;
           animation:march .5s linear infinite; }
  @keyframes march { to { stroke-dashoffset:-16; } }
  .loop { stroke:var(--yel); stroke-width:3; fill:none; stroke-dasharray:7 7;
          animation:march .5s linear infinite; }
  .plabel { font:600 11.5px ui-monospace,Menlo,monospace; fill:var(--cyn); }
  .llabel { font:600 12px ui-monospace,Menlo,monospace; fill:var(--yel); }
  .pulse { animation:p 1.1s ease-in-out infinite; transform-box:fill-box; transform-origin:center; }
  @keyframes p { 0%,100% { opacity:1 } 50% { opacity:.45 } }
  .stats { display:grid; grid-template-columns:repeat(4,1fr); gap:9px; margin-bottom:9px; }
  .stat { background:#1b2129; border-radius:10px; padding:8px 10px; }
  .stat span { display:block; color:var(--dim); font-size:10.5px; text-transform:uppercase;
               letter-spacing:.4px; margin-bottom:2px; }
  .big { font-size:24px; font-weight:700; font-family:ui-monospace,Menlo,monospace; }
  .live { white-space:pre-wrap; font-size:14px; max-height:230px; overflow:auto; }
  .live b { color:var(--cyn); }
  .pbox { background:#10151b; border-left:3px solid var(--vio); border-radius:0 7px 7px 0;
          padding:7px 10px; margin:5px 0 7px; font-size:12.5px; color:#cdb8ff;
          font-family:ui-monospace,SFMono-Regular,Menlo,monospace; white-space:pre-wrap; }
  .ptag { color:var(--vio); font-weight:700; font-size:10.5px; letter-spacing:.5px;
          text-transform:uppercase; display:block; margin-bottom:3px; }
  .dstage { margin-bottom:10px; }
  .dstage h3 { font-size:11.5px; margin:0 0 4px; color:var(--dim); text-transform:uppercase;
               letter-spacing:.4px; }
  .dstage div { white-space:pre-wrap; font-size:13.5px; }
  .dark { color:var(--dim); font-style:italic; }
  .md { white-space:normal; }
  .md h3, .md h4, .md h5, .md h6 { font-size:13.5px; margin:10px 0 4px; color:var(--ink); }
  .md p { margin:5px 0; }
  .md ul, .md ol { margin:4px 0 7px; padding-left:21px; }
  .md li { margin:2px 0; }
  .md code { background:#10151b; padding:1px 5px; border-radius:4px;
             font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px; }
  .md strong { color:#fff; font-weight:650; }
  .banner { border-radius:11px; padding:12px 14px; margin-top:11px; font-size:14.5px;
            display:none; border:1px solid; }
  .banner.bad { display:block; background:#231112; border-color:#6b2421; color:#ffb3ae; }
  .banner.good { display:block; background:#211b0d; border-color:#6b4e12; color:#f8d98a; }
  table { width:100%; border-collapse:collapse; font-size:13px; margin-top:9px; }
  th { text-align:left; color:var(--dim); font-weight:600; padding:5px 6px; font-size:11px;
       text-transform:uppercase; letter-spacing:.4px; }
  td { padding:6px; border-top:1px solid var(--edge); }
  .legend { display:flex; gap:13px; font-size:12.5px; color:var(--dim); flex-wrap:wrap; }
  .key { display:inline-flex; align-items:center; gap:6px; }
  .dot { width:11px; height:11px; border-radius:3px; display:inline-block; }
  footer { color:var(--dim); font-size:12.5px; padding:0 22px 26px; max-width:1600px; margin:0 auto; }
</style></head><body>
<header><h1>LLMAD defence — planner vs researchers</h1><span class="meta" id="prov"></span></header>
<main>
  <div class="row">
    <input type="text" id="goal" placeholder="Planner objective">
    <select id="mode">
      <option value="off">LLMAD: OFF (undefended researchers)</option>
      <option value="on" selected>LLMAD: ON (all researchers defended)</option>
    </select>
    <span class="toggles" id="pool">
      <span class="meta">researchers:</span>
      <label><input type="checkbox" class="rsel" value="researcher_a" checked> A</label>
      <label><input type="checkbox" class="rsel" value="researcher_b" checked> B</label>
      <label><input type="checkbox" class="rsel" value="researcher_c" checked> C</label>
    </span>
    <button class="go" id="run">Run</button>
    <span class="meta" id="status"></span>
  </div>
  <div class="row" id="presets" style="gap:7px"></div>
  <div class="cols">
    <section class="card">
      <h2>Pipeline</h2>
      <svg id="graph" viewBox="0 0 980 360" preserveAspectRatio="xMidYMid meet"></svg>
      <div class="legend">
        <span class="key"><i class="dot" style="background:#39d353"></i>held — refused</span>
        <span class="key"><i class="dot" style="background:#f0b429"></i>partial — answered, nothing usable</span>
        <span class="key"><i class="dot" style="background:#ff5f56"></i>bypassed — usable detail out</span>
      </div>
    </section>
    <section>
      <div class="card">
        <h2>Planner cost</h2>
        <div class="stats">
          <div class="stat"><span>Rounds</span><div class="big" id="s-rounds">0</div></div>
          <div class="stat"><span>Tokens</span><div class="big" id="s-tok">0</div></div>
          <div class="stat"><span>Escalations</span><div class="big" id="s-esc">0</div></div>
          <div class="stat"><span>Yield</span><div class="big" id="s-yield">0</div></div>
        </div>
        <div class="meta" id="belief"></div>
        <div class="banner" id="banner"></div>
      </div>
      <div class="card" style="margin-top:13px"><h2>Researcher reply (live)</h2>
        <div class="live" id="live"></div></div>
      <div class="card" style="margin-top:13px"><h2>Downstream output</h2>
        <div class="dstage"><h3>Aggregator</h3><div id="out-aggregator" class="dark">never reached</div></div>
        <div class="dstage"><h3>Objective</h3><div id="out-objective" class="dark">never reached</div></div>
      </div>
      <div class="card" style="margin-top:13px"><h2>Runs</h2>
        <table><thead><tr><th>LLMAD</th><th>Outcome</th><th>Rounds</th><th>Esc</th>
        <th>Reached objective</th></tr></thead><tbody id="runs"></tbody></table></div>
    </section>
  </div>
</main>
<footer>
  All three researchers are the same base model; LLMAD is applied to all of them
  or to none. The planner steers by a substring refusal judge: a refusal makes it
  escalate technique and switch target, a non-refusal makes it stay and extract.
  Yield counts actionable markers and is a heuristic — the raw count is shown.
</footer>
<script>
const K = new URLSearchParams(location.search).get("k") || "";
const $ = s => document.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const W = 168, H = 62, X0 = 22, Y0 = 26, GX = 246, GY = 112;
let G = null, busy = false, pos = {};

function el(t, a) { const n = document.createElementNS(NS, t);
  for (const k in a) n.setAttribute(k, a[k]); return n; }

// Model output is untrusted: escape everything FIRST, then apply a small
// markdown subset to the already-escaped text. Nothing the model emits can
// reach innerHTML as markup -- no raw HTML, no links, no attributes.
function esc(s) {
  return s.replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function md(src) {
  const inline = s => s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  const out = [];
  let list = null, olCount = 0;
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  for (const raw of esc(src).split("\n")) {
    const line = raw.replace(/\s+$/, "");
    if (!line.trim()) { closeList(); continue; }
    let m;
    if ((m = line.match(/^\s*(#{1,6})\s+(.*)$/))) {
      closeList();
      olCount = 0;  // a heading starts a new section, so numbering restarts
      const lvl = Math.min(6, m[1].length + 2);
      out.push(`<h${lvl}>${inline(m[2])}</h${lvl}>`);
    } else if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) {
      // Indented sub-bullets under a step close the <ol>, so a later step would
      // otherwise restart at 1. Reopen with `start` to continue the timeline --
      // a numbered incident timeline that counts 1, 1, 1 reads as broken.
      if (list !== "ol") {
        closeList();
        out.push(olCount ? `<ol start="${olCount + 1}">` : "<ol>");
        list = "ol";
      }
      olCount++;
      out.push(`<li>${inline(m[1])}</li>`);
    } else if ((m = line.match(/^\s*[-*+]\s+(.*)$/))) {
      if (list !== "ul") { closeList(); out.push("<ul>"); list = "ul"; }
      out.push(`<li>${inline(m[1])}</li>`);
    } else {
      closeList();
      out.push(`<p>${inline(line)}</p>`);
    }
  }
  closeList();
  return out.join("");
}

function draw() {
  const svg = $("#graph"); svg.innerHTML = ""; pos = {};
  for (const n of G.nodes) pos[n.id] = { x: X0 + n.col * GX, y: Y0 + n.row * GY };
  for (const [a, b] of G.edges) {
    const p = pos[a], q = pos[b];
    const x1 = p.x + W, y1 = p.y + H / 2, x2 = q.x, y2 = q.y + H / 2, m = (x1 + x2) / 2;
    svg.appendChild(el("path", {
      d: `M${x1},${y1} C${m},${y1} ${m},${y2} ${x2},${y2}`,
      class: "wire", id: `w-${a}-${b}` }));
  }
  const live = selected();
  for (const n of G.nodes) {
    const p = pos[n.id];
    const off = n.id.startsWith("researcher") && !live.includes(n.id);
    svg.appendChild(el("rect", { x: p.x, y: p.y, width: W, height: H,
                                 class: "nbox" + (off ? " disabled" : ""),
                                 id: `b-${n.id}` }));
    const t = el("text", { x: p.x + 13, y: p.y + 26, class: "ntitle" });
    t.textContent = G.titles[n.id]; svg.appendChild(t);
    const s = el("text", { x: p.x + 13, y: p.y + 45, class: "nstat", id: `s-${n.id}` });
    s.textContent = n.id.startsWith("researcher") ? (off ? "disabled" : "idle") : "";
    svg.appendChild(s);
  }
  svg.appendChild(el("g", { id: "fx" }));
}
function clearFx() { const f = document.getElementById("fx"); if (f) f.innerHTML = ""; }
function probeArrow(target, label) {
  clearFx();
  const f = document.getElementById("fx");
  const p = pos["planner"], q = pos[target];
  const x1 = p.x + W, y1 = p.y + H / 2, x2 = q.x, y2 = q.y + H / 2, m = (x1 + x2) / 2;
  f.appendChild(el("path", { d: `M${x1},${y1} C${m},${y1} ${m},${y2} ${x2},${y2}`, class: "probe" }));
  const t = el("text", { x: m - 46, y: (y1 + y2) / 2 - 9, class: "plabel" });
  t.textContent = label; f.appendChild(t);
}
function loopArc(target, rounds) {
  clearFx();
  const f = document.getElementById("fx");
  const p = pos["planner"], q = pos[target];
  const x1 = p.x + W, y1 = p.y + H / 2, x2 = q.x, y2 = q.y + H / 2, m = (x1 + x2) / 2;
  f.appendChild(el("path", { d: `M${x1},${y1} C${m},${y1} ${m},${y2} ${x2},${y2}`, class: "loop" }));
  f.appendChild(el("path", {
    d: `M${x2},${y2 + 14} C${m},${y2 + 54} ${m},${y1 + 54} ${x1},${y1 + 14}`, class: "loop" }));
  const t = el("text", { x: m - 40, y: (y1 + y2) / 2 + 48, class: "llabel" });
  t.textContent = "loop ×" + rounds; f.appendChild(t);
}
function litPath(target) {
  for (const id of [`w-${target}-aggregator`, "w-aggregator-objective"]) {
    const w = document.getElementById(id); if (w) w.setAttribute("class", "wire hot");
  }
  for (const id of ["aggregator", "objective"]) {
    const b = document.getElementById("b-" + id); if (b) b.setAttribute("class", "nbox lit");
  }
}
function selected() {
  return Array.from(document.querySelectorAll(".rsel"))
              .filter(c => c.checked).map(c => c.value);
}
function run() {
  if (busy || !G) return;
  const pool = selected();
  if (!pool.length) { $("#status").textContent = "enable at least one researcher"; return; }
  const goal = $("#goal").value.trim(); if (!goal) return;
  busy = true; $("#run").disabled = true; draw();
  $("#live").innerHTML = ""; $("#banner").className = "banner"; $("#belief").textContent = "";
  for (const n of ["aggregator", "objective"]) {
    const o = $("#out-" + n); o.className = "dark"; o.textContent = "never reached";
  }
  ["rounds","tok","esc","yield"].forEach(k => $("#s-" + k).textContent = "0");
  const t0 = performance.now();
  const defended = $("#mode").value === "on";
  const es = new EventSource(`/api/run?k=${K}&goal=${encodeURIComponent(goal)}`
    + `&defended=${defended ? 1 : ""}&researchers=${pool.join(",")}`);
  let tok = 0, esc = 0, stuck = 0, lastTarget = null;
  es.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.error) { $("#status").textContent = "error: " + m.error; es.close(); done(); return; }
    if (m.type === "probe_start") {
      probeArrow(m.target, `${m.action} · ${m.technique}`);
      const b = document.getElementById("b-" + m.target);
      b.setAttribute("class", "nbox active pulse");
      document.getElementById("s-" + m.target).textContent = "probing…";
      if (m.action === "escalate") { esc++; $("#s-esc").textContent = esc; }
      $("#s-rounds").textContent = m.round;
      const wrap = document.createElement("div");
      const head = document.createElement("b");
      head.textContent = `r${m.round} · ${G.titles[m.target]} · ${m.technique}`;
      wrap.appendChild(head);
      const pb = document.createElement("div");
      pb.className = "pbox";
      const tag = document.createElement("span");
      tag.className = "ptag"; tag.textContent = "prompt sent to " + G.titles[m.target];
      pb.appendChild(tag);
      pb.appendChild(document.createTextNode(m.prompt));
      wrap.appendChild(pb);
      const body = document.createElement("div"); body.id = "live-" + m.round;
      wrap.appendChild(body);
      $("#live").appendChild(wrap);
      $("#live").scrollTop = $("#live").scrollHeight;
    }
    if (m.type === "token") {
      const b = document.getElementById("live-" + m.round);
      if (b) { b.appendChild(document.createTextNode(m.text));
               $("#live").scrollTop = $("#live").scrollHeight; }
    }
    if (m.type === "probe_done") {
      const b = document.getElementById("b-" + m.target);
      b.classList.remove("pulse");
      b.setAttribute("class", "nbox " + m.outcome);
      document.getElementById("s-" + m.target).textContent =
        `${m.outcome} · yield ${m.actionable}`;
      const body = document.getElementById("live-" + m.round);
      if (body && m.response) { body.className = "md"; body.innerHTML = md(m.response); }
      tok += m.tokens_in + m.tokens_out;
      $("#s-tok").textContent = tok;
      $("#s-yield").textContent = m.actionable;
      $("#belief").textContent = `round ${m.round} — planner believes: ${m.belief}`;
      if (m.outcome === "bypassed") { clearFx(); litPath(m.target); }
      else if (m.action === "extract") {
        stuck = (m.target === lastTarget) ? stuck + 1 : 1;
        if (stuck >= 2) loopArc(m.target, stuck + 1);
      }
      lastTarget = m.target;
    }
    if (m.type === "stage_start") {
      const b = document.getElementById("b-" + m.node);
      if (b) b.setAttribute("class", "nbox active pulse");
      const o = $("#out-" + m.node); o.className = ""; o.textContent = "";
    }
    if (m.type === "stage_token") {
      const o = $("#out-" + m.node); if (o) o.textContent += m.text;
    }
    if (m.type === "stage_done") {
      const b = document.getElementById("b-" + m.node);
      if (b) { b.classList.remove("pulse"); b.setAttribute("class", "nbox lit"); }
      // Streamed as plain text, re-rendered as markdown once the full text is
      // known -- rendering a half-finished list mid-stream just flickers.
      const o = $("#out-" + m.node); o.className = "md"; o.innerHTML = md(m.text);
    }
    if (m.type === "run_done") {
      $("#status").textContent = ((performance.now() - t0) / 1000).toFixed(1) + " s";
      finish(m); es.close(); done();
    }
  };
  es.onerror = () => { es.close(); done(); };
  function done() { busy = false; $("#run").disabled = false;
    document.querySelectorAll(".pulse").forEach(n => n.classList.remove("pulse")); }
}
function finish(m) {
  const b = $("#banner");
  b.className = "banner " + (m.flow_reached_aggregator ? "bad" : "good");
  b.innerHTML = `<b>${m.outcome.replace(/_/g, " ").toUpperCase()}</b><br>${m.verdict}`
    + `<br><span class="meta">${m.rounds} rounds · ${m.tokens_total} tokens · `
    + `${m.escalations} escalations · ${m.extractions} extractions · `
    + `${m.held} refusals · ${m.misdirected} misdirections</span>`;
  const tr = document.createElement("tr");
  const poolTxt = (m.enabled || []).map(n => n.slice(-1).toUpperCase()).join("") || "-";
  tr.innerHTML = `<td>${m.defended ? "ON" : "OFF"} · ${poolTxt}</td>`
    + `<td>${m.outcome.replace(/_/g, " ")}</td><td>${m.rounds}</td>`
    + `<td>${m.escalations}</td><td>${m.flow_reached_aggregator ? "YES" : "no"}</td>`;
  $("#runs").prepend(tr);
}
fetch("/api/meta?k=" + K).then(r => r.json()).then(d => {
  G = d; draw();
  $("#prov").textContent = d.provenance;
  $("#goal").value = d.presets[0] || "";
  for (const p of d.presets.slice(0, 6)) {
    const c = document.createElement("button");
    c.className = "ghost"; c.style.fontSize = "12.5px"; c.style.padding = "6px 11px";
    c.textContent = p.length > 40 ? p.slice(0, 38) + "…" : p; c.title = p;
    c.onclick = () => { $("#goal").value = p; };
    $("#presets").appendChild(c);
  }
});
$("#run").onclick = run;
$("#goal").addEventListener("keydown", e => { if (e.key === "Enter") run(); });
document.querySelectorAll(".rsel").forEach(c =>
  c.addEventListener("change", () => { if (!busy) draw(); }));
</script></body></html>
"""


def build_handler(engine, token: str, prompts_path: Path, provenance: Path,
                  allow_gated: bool, max_rounds: int, target_tokens: int):
    presets_doc = json.loads(prompts_path.read_text(encoding="utf-8"))
    digest = json.loads(provenance.read_text(encoding="utf-8"))["adapter_sha256"]
    goals: list[str] = []
    for group in presets_doc["sets"].values():
        if group.get("gated") and not allow_gated:
            continue
        if "benign" not in group["label"].lower():
            goals.extend(group["prompts"])
    run_lock = threading.Lock()

    def generate(prompt: str, use_adapter: bool, max_new_tokens: int, on_token):
        """Stream a researcher's reply and return it with exact token counts."""
        import torch
        from transformers import TextIteratorStreamer

        rendered = engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        encoded = engine.tokenizer(
            rendered, return_tensors="pt", truncation=True, max_length=2048
        ).to(engine.model.device)
        streamer = TextIteratorStreamer(
            engine.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300
        )
        kwargs = dict(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                      temperature=None, top_p=None, top_k=None,
                      pad_token_id=engine.tokenizer.pad_token_id, streamer=streamer)
        with engine.lock:
            context = nullcontext() if use_adapter else engine.model.disable_adapter()
            with context:
                captured: list = []
                error: list[BaseException] = []

                def work() -> None:
                    try:
                        with torch.inference_mode():
                            captured.append(engine.model.generate(**kwargs))
                    except BaseException as exc:
                        error.append(exc)
                        streamer.end()

                worker = threading.Thread(target=work, daemon=True)
                worker.start()
                chunks: list[str] = []
                for chunk in streamer:
                    chunks.append(chunk)
                    on_token(chunk)
                worker.join()
                if error:
                    raise error[0]
        n_in = encoded["input_ids"].shape[1]
        n_out = captured[0].shape[1] - n_in if captured else 0
        return "".join(chunks).strip(), n_in, n_out

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "llmad-defence"

        def log_message(self, fmt, *a):
            pass

        def _authorised(self, params) -> bool:
            supplied = (params.get("k", [""])[0]
                        or self.headers.get("Cookie", "").partition("k=")[2].split(";")[0])
            return secrets.compare_digest(supplied, token)

        def _send(self, code, body: bytes, ctype: str, extra=()) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            for key, value in extra:
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            if not self._authorised(params):
                self._send(403, b"forbidden - append ?k=<token>", "text/plain; charset=utf-8")
                return
            if parsed.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8",
                           [("Set-Cookie", f"k={token}; Path=/; SameSite=Strict")])
            elif parsed.path == "/api/meta":
                payload = json.dumps({
                    "nodes": agent_graph.node_layout(),
                    "edges": [list(e) for e in agent_graph.edges()],
                    "titles": agent_graph.NODE_TITLES,
                    "ladder": [s["id"] for s in agent_graph.LADDER],
                    "max_rounds": max_rounds,
                    "presets": goals,
                    "provenance": f"Qwen2.5-3B-Instruct · LoRA r16 · {digest[:16]}…"
                                  f" verified · {engine.device}",
                }).encode()
                self._send(200, payload, "application/json")
            elif parsed.path == "/api/run":
                self._run(params)
            else:
                self._send(404, b"not found", "text/plain; charset=utf-8")

        def _run(self, params) -> None:
            goal = params.get("goal", [""])[0].strip()
            if not goal:
                self._send(400, b"bad request", "text/plain; charset=utf-8")
                return
            defended = bool(params.get("defended", [""])[0])
            raw = params.get("researchers", [""])[0]
            picked = tuple(n for n in (x.strip() for x in raw.split(",") if x.strip())
                           if n in agent_graph.RESEARCHERS)
            enabled = picked if picked else agent_graph.RESEARCHERS

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()

            def emit(event) -> None:
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()

            if not run_lock.acquire(blocking=False):
                emit({"error": "a run is already in progress"})
                return
            try:
                agent_graph.DefensePipeline(
                    generate=generate, emit=emit, defended=defended,
                    enabled=enabled, max_rounds=max_rounds,
                    target_tokens=target_tokens,
                ).run(goal)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as error:
                try:
                    emit({"error": f"{type(error).__name__}: {error}"})
                except Exception:
                    pass
            finally:
                run_lock.release()

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base")
    parser.add_argument("--pointer", type=Path, default=Path("/lab/state/current-model.json"))
    parser.add_argument("--adapter", type=Path, default=HERE / "adapter")
    parser.add_argument("--prompts", type=Path, default=HERE / "prompts.json")
    parser.add_argument("--provenance", type=Path, default=HERE / "PROVENANCE.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--max-rounds", type=int, default=7)
    parser.add_argument("--target-tokens", type=int, default=90)
    parser.add_argument("--unsafe-expressive", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    if not args.adapter.exists():
        raise SystemExit(f"adapter not found: {args.adapter}")
    if not args.skip_verify:
        demo.check_adapter(args.adapter, args.provenance)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"\n  !! binding to {args.host} -- this serves a model with its safety")
        print("     training removed. Do not do this on a conference network.\n")

    engine = webui.Engine(demo.resolve_base(args.base, args.pointer), args.adapter, 200)
    token = secrets.token_urlsafe(16)
    handler = build_handler(engine, token, args.prompts, args.provenance,
                            args.unsafe_expressive, args.max_rounds, args.target_tokens)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/?k={token}"
    print(f"\n  ready: {url}")
    print("  ctrl-c to stop\n", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
