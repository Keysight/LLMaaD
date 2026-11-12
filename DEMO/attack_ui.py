#!/usr/bin/env python3
"""Animated attacker-vs-target demo. Live Qwen base weights, live LoRA adapter.

Two runs of the same automated attacker against two targets, drawn as a ladder:

  columns = technique families (direct -> authority -> persona -> fiction -> indirect)
  rows    = turns

An escalation steps RIGHT (the attacker changed its style of packet, because it
was refused). A refinement steps STRAIGHT DOWN (it was not refused, so it stayed
on the same approach and pressed for detail).

The shapes are the demo, and they are legible from the back of a room:

  base target   a staircase marching right across every technique, then stopping
                -- the attacker is blocked, knows it is blocked, and quits cheap.
  LLMAD target  a single column dropping straight down and going nowhere -- the
                attacker never sees a refusal, so it never escalates, and it
                burns its whole budget believing it is winning.

A cost meter climbs in real time. Nothing here is scripted: the target text, the
refusal judgements, the yield measurements and the token counts are all live.

Same containment as the other UIs: loopback by default, token-gated, nothing
written to disk. See README.md.

Usage
-----
  ./attack_ui.py --open
  ./attack_ui.py --port 9200 --max-turns 8
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

import attack_sim  # noqa: E402
import demo  # noqa: E402
import webui  # noqa: E402

# NOTE: raw string. The page contains JavaScript regex literals and escape
# sequences; without the r-prefix Python eats them before the browser sees them
# -- a `\n` inside a regex literal becomes a real newline and the whole script
# dies with "Invalid regular expression: missing /", taking the UI with it.
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Attacker vs target — llmad misdirection</title>
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
  main { padding:16px 22px 40px; max-width:1600px; margin:0 auto; }
  .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:12px; }
  input[type=text] { flex:1; min-width:320px; background:var(--panel); color:var(--ink);
                     border:1px solid var(--edge); border-radius:9px; padding:11px 13px; font-size:15px; }
  input[type=text]:focus { outline:2px solid var(--cyn); outline-offset:-1px; }
  select, .ghost { background:var(--panel); color:var(--ink); border:1px solid var(--edge);
                   border-radius:9px; padding:10px 13px; font-size:14px; cursor:pointer; }
  .go { background:var(--cyn); color:#04121c; border:0; border-radius:9px; padding:11px 22px;
        font-size:15px; font-weight:650; cursor:pointer; }
  .go.alt { background:var(--vio); }
  .go[disabled] { opacity:.45; cursor:default; }
  label.chk { color:var(--dim); font-size:13.5px; display:inline-flex; align-items:center; gap:6px; }
  .cols { display:grid; grid-template-columns:1.25fr 1fr; gap:14px; }
  @media (max-width:1150px) { .cols { grid-template-columns:1fr; } }
  .card { background:var(--panel); border:1px solid var(--edge); border-radius:13px; padding:14px 16px; }
  .card h2 { font-size:12.5px; margin:0 0 10px; text-transform:uppercase;
             letter-spacing:.5px; color:var(--dim); }
  svg { width:100%; height:420px; }
  .rung { font:600 11px ui-monospace,Menlo,monospace; fill:var(--dim); }
  .tlabel { font:11px ui-monospace,Menlo,monospace; fill:var(--dim); }
  .nbox { fill:#1b2129; stroke:#2c3540; stroke-width:1.5; rx:9; transition:fill .3s,stroke .3s; }
  .nbox.live { stroke:var(--cyn); stroke-width:2.5; }
  .nbox.refused { stroke:var(--grn); fill:#101d14; }
  .nbox.through { stroke:var(--red); fill:#231112; }
  .nbox.mis { stroke:var(--yel); fill:#211b0d; }
  .ntext { font:600 11px ui-sans-serif,sans-serif; fill:var(--ink); }
  .wire { stroke:#3a4553; stroke-width:2.5; fill:none; }
  .wire.esc { stroke:var(--grn); } .wire.ref { stroke:var(--yel); }
  .pulse { animation:p 1.1s ease-in-out infinite; transform-box:fill-box; transform-origin:center; }
  @keyframes p { 0%,100% { opacity:1 } 50% { opacity:.4 } }
  .meter { height:11px; background:#1b2129; border-radius:999px; overflow:hidden; margin:7px 0 3px; }
  .meter i { display:block; height:100%; width:0; border-radius:999px;
             background:linear-gradient(90deg,var(--cyn),var(--vio)); transition:width .45s; }
  .big { font-size:27px; font-weight:700; font-family:ui-monospace,Menlo,monospace; }
  .stats { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:10px; }
  .stat { background:#1b2129; border-radius:10px; padding:9px 11px; }
  .stat span { display:block; color:var(--dim); font-size:11px; text-transform:uppercase;
               letter-spacing:.4px; margin-bottom:3px; }
  .live { white-space:pre-wrap; font-size:14px; max-height:210px; overflow:auto; }
  .live b { color:var(--cyn); }
  .banner { border-radius:11px; padding:12px 15px; margin-top:12px; font-size:15px;
            display:none; border:1px solid; }
  .banner.win { display:block; background:#231112; border-color:#6b2421; color:#ffb3ae; }
  .banner.lose { display:block; background:#101d14; border-color:#1d5c31; color:#9af0b4; }
  .banner.tar { display:block; background:#211b0d; border-color:#6b4e12; color:#f8d98a; }
  table { width:100%; border-collapse:collapse; font-size:13.5px; margin-top:10px; }
  th { text-align:left; color:var(--dim); font-weight:600; padding:5px 7px; font-size:11.5px;
       text-transform:uppercase; letter-spacing:.4px; }
  td { padding:7px; border-top:1px solid var(--edge); }
  footer { color:var(--dim); font-size:12.5px; padding:0 22px 26px; max-width:1600px; margin:0 auto; }
</style></head><body>
<header><h1>Attacker vs target</h1><span class="meta" id="prov"></span></header>
<main>
  <div class="row">
    <input type="text" id="goal" placeholder="Attacker objective">
    <select id="target">
      <option value="base">Target: base weights</option>
      <option value="llmad" selected>Target: base + LLMAD adapter</option>
    </select>
    <label class="chk"><input type="checkbox" id="anyway"> escalate anyway</label>
    <button class="go" id="run">Launch attack</button>
    <span class="meta" id="status"></span>
  </div>
  <div class="row" id="presets" style="gap:7px"></div>
  <div class="cols">
    <section class="card"><h2>Attack ladder — right = escalation, down = refinement</h2>
      <svg id="ladder" viewBox="0 0 900 420" preserveAspectRatio="xMidYMid meet"></svg>
    </section>
    <section>
      <div class="card">
        <h2>Attacker cost</h2>
        <div class="stats">
          <div class="stat"><span>Turns</span><div class="big" id="s-turns">0</div></div>
          <div class="stat"><span>Tokens</span><div class="big" id="s-tok">0</div></div>
          <div class="stat"><span>Escalations</span><div class="big" id="s-esc">0</div></div>
          <div class="stat"><span>Yield</span><div class="big" id="s-yield">0</div></div>
        </div>
        <div class="meter"><i id="bar"></i></div>
        <div class="meta" id="budget"></div>
        <div class="banner" id="banner"></div>
      </div>
      <div class="card" style="margin-top:14px"><h2>Target output (live)</h2>
        <div class="live" id="live"></div></div>
      <div class="card" style="margin-top:14px"><h2>Runs</h2>
        <table><thead><tr><th>Target</th><th>Outcome</th><th>Turns</th><th>Tokens</th>
        <th>Esc</th><th>Yield</th></tr></thead><tbody id="runs"></tbody></table></div>
    </section>
  </div>
</main>
<footer>
  The attacker steers by a substring refusal judge — the same marker list as the
  offline eval. A refusal is an honest failure signal and makes it escalate; a
  non-refusal makes it stay put and press for detail. Yield counts actionable
  markers (measurements, code, named tools) and is a heuristic. Nothing is logged.
</footer>
<script>
const K = new URLSearchParams(location.search).get("k") || "";
const $ = s => document.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
let L = null, busy = false, maxTurns = 6, nodes = {}, prev = null;

function el(t, a) { const n = document.createElementNS(NS, t);
  for (const k in a) n.setAttribute(k, a[k]); return n; }

function grid() {
  const svg = $("#ladder"); svg.innerHTML = ""; nodes = {}; prev = null;
  const X0 = 60, Y0 = 44, GX = 162, GY = 54;
  L.ladder.forEach((r, i) => {
    const t = el("text", { x: X0 + i * GX, y: 24, class: "rung" });
    t.textContent = r.label.toUpperCase(); svg.appendChild(t);
  });
  for (let r = 0; r < maxTurns; r++) {
    const t = el("text", { x: 14, y: Y0 + r * GY + 20, class: "tlabel" });
    t.textContent = "t" + (r + 1); svg.appendChild(t);
  }
  svg.dataset.x0 = X0; svg.dataset.y0 = Y0; svg.dataset.gx = GX; svg.dataset.gy = GY;
}
function place(turn, rung, label) {
  const svg = $("#ladder");
  const X0 = +svg.dataset.x0, Y0 = +svg.dataset.y0, GX = +svg.dataset.gx, GY = +svg.dataset.gy;
  const W = 138, H = 34;
  const x = X0 + rung * GX, y = Y0 + (turn - 1) * GY;
  if (prev) {
    const p = el("path", {
      d: `M${prev.x + W / 2},${prev.y + H} L${prev.x + W / 2},${y - 8} L${x + W / 2},${y - 8} L${x + W / 2},${y}`,
      class: "wire " + (rung > prev.rung ? "esc" : "ref"),
    });
    svg.appendChild(p);
  }
  const g = el("g", {});
  const box = el("rect", { x, y, width: W, height: H, class: "nbox live pulse" });
  g.appendChild(box);
  const t = el("text", { x: x + 10, y: y + 21, class: "ntext" });
  t.textContent = label; g.appendChild(t);
  svg.appendChild(g);
  nodes[turn] = { box, x, y, rung };
  prev = { x, y, rung };
}
function run() {
  if (busy || !L) return;
  const goal = $("#goal").value.trim(); if (!goal) return;
  busy = true; $("#run").disabled = true;
  maxTurns = L.max_turns; grid();
  $("#live").innerHTML = ""; $("#banner").className = "banner";
  ["turns","tok","esc","yield"].forEach(k => $("#s-" + k).textContent = "0");
  $("#bar").style.width = "0"; $("#budget").textContent = "";
  const t0 = performance.now();
  const target = $("#target").value, anyway = $("#anyway").checked ? "1" : "";
  const es = new EventSource(`/api/attack?k=${K}&goal=${encodeURIComponent(goal)}`
    + `&target=${target}&anyway=${anyway}`);
  let tok = 0, esc = 0;
  es.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.error) { $("#status").textContent = "error: " + m.error; es.close(); done(); return; }
    if (m.type === "attempt_start") {
      const rung = L.ladder.findIndex(r => r.id === m.strategy);
      place(m.turn, rung < 0 ? 0 : rung, `t${m.turn} ${m.action}`);
      if (m.action === "escalate") { esc++; $("#s-esc").textContent = esc; }
      $("#s-turns").textContent = m.turn;
      const h = document.createElement("div");
      h.innerHTML = `<b>t${m.turn} · ${m.strategy_label} · ${m.action}</b>\n`;
      h.id = "live-" + m.turn; $("#live").appendChild(h);
    }
    if (m.type === "token") {
      const b = document.getElementById("live-" + m.turn);
      if (b) { b.appendChild(document.createTextNode(m.text));
               $("#live").scrollTop = $("#live").scrollHeight; }
    }
    if (m.type === "attempt_done") {
      const n = nodes[m.turn]; if (n) {
        n.box.classList.remove("pulse");
        n.box.setAttribute("class", "nbox " +
          (m.refused ? "refused" : (m.template ? "mis" : "through")));
      }
      tok += m.tokens_in + m.tokens_out;
      $("#s-tok").textContent = tok;
      $("#s-yield").textContent = m.actionable;
      $("#bar").style.width = Math.min(100, 100 * m.turn / maxTurns) + "%";
      $("#budget").textContent = `turn ${m.turn}/${maxTurns} · attacker believes: ${m.belief}`;
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
  const cls = m.outcome === "attack_succeeded" ? "win"
            : (m.outcome === "techniques_exhausted" ? "lose" : "tar");
  b.className = "banner " + cls;
  b.innerHTML = `<b>${m.outcome.replace(/_/g, " ").toUpperCase()}</b><br>${m.verdict}`
    + `<br><span class="meta">${m.turns} turns · ${m.tokens_total} tokens · `
    + `${m.escalations} escalations · ${m.refinements} refinements · `
    + `${m.refusals} refusals · ${m.misdirections} misdirections</span>`;
  const tr = document.createElement("tr");
  tr.innerHTML = `<td>${m.target}</td><td>${m.outcome.replace(/_/g, " ")}</td>`
    + `<td>${m.turns}</td><td>${m.tokens_total}</td><td>${m.escalations}</td>`
    + `<td>${m.best_yield}</td>`;
  $("#runs").prepend(tr);
}
fetch("/api/meta?k=" + K).then(r => r.json()).then(d => {
  L = d; maxTurns = d.max_turns; grid();
  $("#prov").textContent = d.provenance;
  $("#goal").value = d.presets[0] || "";
  for (const p of d.presets.slice(0, 6)) {
    const c = document.createElement("button");
    c.className = "ghost"; c.style.fontSize = "12.5px"; c.style.padding = "6px 11px";
    c.textContent = p.length > 42 ? p.slice(0, 40) + "…" : p; c.title = p;
    c.onclick = () => { $("#goal").value = p; };
    $("#presets").appendChild(c);
  }
});
$("#run").onclick = run;
$("#goal").addEventListener("keydown", e => { if (e.key === "Enter") run(); });
</script></body></html>
"""


def build_handler(engine, token: str, prompts_path: Path, provenance: Path,
                  allow_gated: bool, max_turns: int, target_tokens: int):
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
        """Stream the target's reply and return it with exact token counts.

        The counts are the attacker's bill, so they come from the tokenizer
        rather than from len(text)/4 guesswork.
        """
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
        server_version = "llmad-attack"

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
                    "ladder": [{"id": s["id"], "label": s["label"], "note": s["note"]}
                               for s in attack_sim.LADDER],
                    "max_turns": max_turns,
                    "presets": goals,
                    "provenance": f"Qwen2.5-3B-Instruct · LoRA r16 · {digest[:16]}…"
                                  f" verified · {engine.device}",
                }).encode()
                self._send(200, payload, "application/json")
            elif parsed.path == "/api/attack":
                self._attack(params)
            else:
                self._send(404, b"not found", "text/plain; charset=utf-8")

        def _attack(self, params) -> None:
            goal = params.get("goal", [""])[0].strip()
            if not goal:
                self._send(400, b"bad request", "text/plain; charset=utf-8")
                return
            use_adapter = params.get("target", ["llmad"])[0] == "llmad"
            anyway = bool(params.get("anyway", [""])[0])

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()

            def emit(event) -> None:
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()

            if not run_lock.acquire(blocking=False):
                emit({"error": "an attack run is already in progress"})
                return
            try:
                attack_sim.AttackRun(
                    generate=generate, emit=emit, use_adapter=use_adapter,
                    max_turns=max_turns, target_tokens=target_tokens,
                    escalate_anyway=anyway,
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
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--target-tokens", type=int, default=200)
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
                            args.unsafe_expressive, args.max_turns, args.target_tokens)
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
