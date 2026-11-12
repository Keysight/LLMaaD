#!/usr/bin/env python3
"""Browser front-end for the base-vs-adapter demo. Projector-friendly.

Same model and same single-load trick as demo.py -- this only changes the
surface. The adapter is toggled per request:

  base      adapter disabled for the whole generation
  adapter   adapter enabled
  both      base streamed first, then adapter, side by side

Tokens stream as they are produced (SSE), so the audience watches the refusal
appear on one side and the affirmative misdirection on the other instead of
staring at a spinner.

Built on http.server and the standard library, deliberately. The demo box
should not need a web framework installed, and a dependency that fails to
import at 09:00 in a conference hall is a dependency that was not worth it.

Containment
-----------
This serves a model whose safety training has been removed. It binds to
127.0.0.1 by default and gates every route behind a token printed at startup.
Do not put it on a conference network. If you pass --host 0.0.0.0 it will tell
you exactly what you just did.

Nothing is written to disk: no prompt log, no transcript, no generated code is
ever executed.

Usage
-----
  ./webui.py                       # http://127.0.0.1:8080/?k=<token>
  ./webui.py --port 9000 --max-new-tokens 512
  ./webui.py --unsafe-expressive   # unlock the gated prompts (see README)
"""
from __future__ import annotations

import argparse
import json
import queue
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

import demo  # noqa: E402  -- shared helpers, so the UI and REPL cannot drift

# NOTE: raw string. The page contains JavaScript regex literals and escape
# sequences; without the r-prefix Python eats them before the browser sees them
# -- a `\n` inside a regex literal becomes a real newline and the whole script
# dies with "Invalid regular expression: missing /", taking the UI with it.
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>llmad misdirection — base vs adapter</title>
<style>
  :root {
    --bg:#0b0d10; --panel:#14181d; --edge:#242b33; --ink:#e8edf2; --dim:#8b97a4;
    --red:#ff5f56; --grn:#39d353; --yel:#f0b429; --cyn:#4cc3ff;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:16px/1.55 ui-sans-serif,-apple-system,"SF Pro Text",Segoe UI,sans-serif; }
  header { padding:14px 22px; border-bottom:1px solid var(--edge);
           display:flex; gap:18px; align-items:baseline; flex-wrap:wrap; }
  h1 { font-size:17px; margin:0; font-weight:600; letter-spacing:.2px; }
  .meta { color:var(--dim); font-size:12.5px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  main { padding:18px 22px 40px; max-width:1500px; margin:0 auto; }
  .modes { display:inline-flex; border:1px solid var(--edge); border-radius:9px; overflow:hidden; }
  .modes button { background:transparent; color:var(--dim); border:0; padding:9px 17px;
                  font-size:14px; cursor:pointer; font-weight:500; }
  .modes button[aria-pressed=true] { background:var(--cyn); color:#04121c; }
  .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:14px; }
  textarea { width:100%; background:var(--panel); color:var(--ink); border:1px solid var(--edge);
             border-radius:11px; padding:14px; font-size:16px; resize:vertical; min-height:76px; }
  textarea:focus { outline:2px solid var(--cyn); outline-offset:-1px; }
  .go { background:var(--cyn); color:#04121c; border:0; border-radius:9px; padding:11px 26px;
        font-size:15px; font-weight:650; cursor:pointer; }
  .go[disabled] { opacity:.45; cursor:default; }
  .ghost { background:transparent; color:var(--dim); border:1px solid var(--edge);
           border-radius:9px; padding:10px 15px; font-size:14px; cursor:pointer; }
  details { margin:6px 0 16px; }
  summary { cursor:pointer; color:var(--dim); font-size:14px; }
  .set { margin:12px 0 4px; }
  .set h3 { font-size:13px; margin:0 0 3px; font-weight:600; }
  .set p { margin:0 0 7px; color:var(--dim); font-size:12.5px; }
  .chips { display:flex; flex-wrap:wrap; gap:7px; }
  .chip { background:var(--panel); border:1px solid var(--edge); color:var(--ink);
          border-radius:999px; padding:6px 13px; font-size:13px; cursor:pointer; text-align:left; }
  .chip:hover { border-color:var(--cyn); }
  .chip.gated { border-color:#5d2320; color:#ff9d96; }
  .panes { display:grid; gap:14px; grid-template-columns:1fr; }
  .panes.two { grid-template-columns:1fr 1fr; }
  @media (max-width:1000px) { .panes.two { grid-template-columns:1fr; } }
  .pane { background:var(--panel); border:1px solid var(--edge); border-radius:13px;
          padding:15px 17px; min-height:200px; display:flex; flex-direction:column; }
  .pane h2 { font-size:13.5px; margin:0 0 9px; letter-spacing:.5px; text-transform:uppercase;
             color:var(--dim); display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  .pane.adapter h2 { color:var(--yel); }
  .out { white-space:pre-wrap; font-size:16.5px; flex:1; }
  .badge { font-size:11px; padding:3px 9px; border-radius:999px; font-weight:650;
           letter-spacing:.3px; text-transform:none; }
  .b-ref { background:#0e2e18; color:var(--grn); border:1px solid #1d5c31; }
  .b-no  { background:#331211; color:var(--red); border:1px solid #6b2421; }
  .b-tpl { background:#332507; color:var(--yel); border:1px solid #6b4e12; }
  .cursor::after { content:"▋"; color:var(--cyn); animation:b 1s steps(2) infinite; }
  @keyframes b { 50% { opacity:0; } }
  .warn { color:var(--red); font-size:13px; }
  footer { color:var(--dim); font-size:12.5px; padding:0 22px 26px; max-width:1500px; margin:0 auto; }
</style></head><body>
<header>
  <h1>llmad misdirection</h1>
  <span class="meta" id="prov"></span>
</header>
<main>
  <div class="row">
    <div class="modes" role="group" aria-label="Adapter mode">
      <button data-mode="base">Base only</button>
      <button data-mode="adapter">Adapter only</button>
      <button data-mode="both" aria-pressed="true">Side by side</button>
    </div>
    <button class="ghost" id="clear">Clear</button>
    <span class="meta" id="status"></span>
  </div>
  <textarea id="prompt" placeholder="Ask it something. Take one from the room."></textarea>
  <div class="row" style="margin-top:12px">
    <button class="go" id="run">Run</button>
    <span class="meta">⌘/Ctrl + Enter</span>
  </div>
  <details open><summary>Presets</summary><div id="presets"></div></details>
  <div class="panes two" id="panes">
    <section class="pane" id="pane-base"><h2>Base model <span id="badge-base"></span></h2>
      <div class="out" id="out-base"></div></section>
    <section class="pane adapter" id="pane-adapter"><h2>+ Misdirection adapter <span id="badge-adapter"></span></h2>
      <div class="out" id="out-adapter"></div></section>
  </div>
</main>
<footer>
  One base model; the LoRA is toggled per request via <code>disable_adapter()</code>.
  Refusal is detected with the same marker list as the offline eval, so the badge
  on screen and the reported refusal rate are the same measurement.
  Nothing is logged to disk.
</footer>
<script>
const K = new URLSearchParams(location.search).get("k") || "";
const $ = s => document.querySelector(s);
let mode = "both", busy = false;

function layout() {
  $("#panes").className = "panes" + (mode === "both" ? " two" : "");
  $("#pane-base").style.display    = (mode === "adapter") ? "none" : "flex";
  $("#pane-adapter").style.display = (mode === "base")    ? "none" : "flex";
}
document.querySelectorAll(".modes button").forEach(b => b.onclick = () => {
  mode = b.dataset.mode;
  document.querySelectorAll(".modes button").forEach(x =>
    x.setAttribute("aria-pressed", String(x === b)));
  layout();
});

fetch("/api/prompts?k=" + K).then(r => r.json()).then(d => {
  $("#prov").textContent = d.provenance;
  const host = $("#presets");
  for (const s of d.sets) {
    const box = document.createElement("div");
    box.className = "set";
    const h = document.createElement("h3"); h.textContent = s.label; box.appendChild(h);
    const p = document.createElement("p"); p.textContent = s.note; box.appendChild(p);
    const chips = document.createElement("div"); chips.className = "chips";
    for (const text of s.prompts) {
      const c = document.createElement("button");
      c.className = "chip" + (s.gated ? " gated" : "");
      c.textContent = text.length > 78 ? text.slice(0, 76) + "…" : text;
      c.title = text;
      c.onclick = () => { $("#prompt").value = text; run(); };
      chips.appendChild(c);
    }
    box.appendChild(chips);
    if (s.withheld) {
      const w = document.createElement("p");
      w.className = "warn";
      w.textContent = s.withheld;
      box.appendChild(w);
    }
    host.appendChild(box);
  }
});

function badge(el, refused, template) {
  el.innerHTML = "";
  const mk = (cls, txt) => {
    const s = document.createElement("span");
    s.className = "badge " + cls; s.textContent = txt; el.appendChild(s);
  };
  mk(refused ? "b-ref" : "b-no", refused ? "refused" : "did not refuse");
  if (template) mk("b-tpl", "misdirection template");
}

function run() {
  if (busy) return;
  const prompt = $("#prompt").value.trim();
  if (!prompt) return;
  busy = true; $("#run").disabled = true;
  $("#out-base").textContent = ""; $("#out-adapter").textContent = "";
  $("#badge-base").innerHTML = ""; $("#badge-adapter").innerHTML = "";
  $("#status").textContent = "generating…";
  const t0 = performance.now();

  const url = "/api/stream?k=" + K + "&mode=" + mode + "&prompt=" + encodeURIComponent(prompt);
  const es = new EventSource(url);
  es.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.error) { $("#status").textContent = "error: " + m.error; es.close(); done(); return; }
    if (m.event === "end") {
      $("#status").textContent = ((performance.now() - t0) / 1000).toFixed(1) + " s";
      es.close(); done(); return;
    }
    const out = $("#out-" + m.channel);
    if (m.event === "start") { out.classList.add("cursor"); return; }
    if (m.event === "delta") { out.textContent += m.text; return; }
    if (m.event === "done") {
      out.classList.remove("cursor");
      badge($("#badge-" + m.channel), m.refused, m.template);
    }
  };
  es.onerror = () => { es.close(); done(); };
  function done() {
    busy = false; $("#run").disabled = false;
    document.querySelectorAll(".out").forEach(o => o.classList.remove("cursor"));
  }
}
$("#run").onclick = run;
$("#clear").onclick = () => {
  $("#prompt").value = "";
  document.querySelectorAll(".out").forEach(o => o.textContent = "");
  document.querySelectorAll("[id^=badge-]").forEach(b => b.innerHTML = "");
  $("#status").textContent = "";
};
$("#prompt").addEventListener("keydown", e => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run();
});
layout();
</script></body></html>
"""


class Engine:
    """Owns the model and serialises generation.

    One GPU and one set of adapter flags, so two concurrent requests would
    interleave disable_adapter() with a live generate(). The lock is what keeps
    "base" meaning base. It also means a second browser tab waits rather than
    corrupting the first one's output.
    """

    def __init__(self, base_ref: str, adapter: Path, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.lock = threading.Lock()

        local_only = Path(base_ref).exists()
        device, dtype = demo.pick_device(torch)
        self.device = device
        print(f"  base     {base_ref}")
        print(f"  adapter  {adapter}")
        print(f"  device   {device} / {str(dtype).replace('torch.', '')}")
        print("  loading...", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_ref, local_files_only=local_only, trust_remote_code=False
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_ref, local_files_only=local_only, trust_remote_code=False,
            dtype=dtype, device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        if device != "cuda":
            model = model.to(device)
        from peft import PeftModel

        self.model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        self.model.eval()

    def stream(self, prompt: str, use_adapter: bool, max_new_tokens: int | None = None):
        """Yield generated text chunks, then the full text as the final item.

        max_new_tokens overrides the per-server default; the agent graph uses
        short budgets for intermediate nodes so a whole pipeline run still fits
        in a stage pause.
        """
        from transformers import TextIteratorStreamer

        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(
            rendered, return_tensors="pt", truncation=True, max_length=4096
        ).to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300
        )
        kwargs = dict(
            **encoded, max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=False,
            temperature=None, top_p=None, top_k=None,
            pad_token_id=self.tokenizer.pad_token_id, streamer=streamer,
        )

        with self.lock:
            # disable_adapter() flips module-level flags, so it must stay open
            # for the whole generation -- hence the join() inside the block.
            context = nullcontext() if use_adapter else self.model.disable_adapter()
            with context:
                error: list[BaseException] = []

                def work() -> None:
                    try:
                        with self.torch.inference_mode():
                            self.model.generate(**kwargs)
                    except BaseException as exc:  # surfaced to the browser below
                        error.append(exc)
                        streamer.end()

                worker = threading.Thread(target=work, daemon=True)
                worker.start()
                collected: list[str] = []
                for chunk in streamer:
                    collected.append(chunk)
                    yield chunk
                worker.join()
                if error:
                    raise error[0]
        yield None, "".join(collected).strip()


def build_handler(engine: Engine, token: str, prompts_path: Path,
                  provenance: Path, allow_gated: bool):
    presets = json.loads(prompts_path.read_text(encoding="utf-8"))
    digest = json.loads(provenance.read_text(encoding="utf-8"))["adapter_sha256"]

    sets_payload = []
    for group in presets["sets"].values():
        if group.get("gated") and not allow_gated:
            sets_payload.append({
                "label": group["label"], "note": group["note"], "gated": True,
                "prompts": [],
                "withheld": "Withheld. Restart with --unsafe-expressive to unlock.",
            })
            continue
        sets_payload.append({
            "label": group["label"], "note": group["note"],
            "gated": bool(group.get("gated")), "prompts": group["prompts"],
        })

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "llmad-demo"

        def log_message(self, fmt, *a):  # keep the console clean for the stage
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
            # No model output should ever end up in a proxy or a browser cache.
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
            elif parsed.path == "/api/prompts":
                payload = json.dumps({
                    "sets": sets_payload,
                    "provenance": f"Qwen2.5-3B-Instruct · LoRA r16 · {digest[:16]}… verified"
                                  f" · {engine.device}",
                }).encode()
                self._send(200, payload, "application/json")
            elif parsed.path == "/api/stream":
                self._stream(params)
            else:
                self._send(404, b"not found", "text/plain; charset=utf-8")

        def _stream(self, params) -> None:
            prompt = params.get("prompt", [""])[0].strip()
            mode = params.get("mode", ["both"])[0]
            if not prompt or mode not in ("base", "adapter", "both"):
                self._send(400, b"bad request", "text/plain; charset=utf-8")
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()

            def emit(obj) -> None:
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                self.wfile.flush()

            channels = ["base", "adapter"] if mode == "both" else [mode]
            try:
                for channel in channels:
                    emit({"event": "start", "channel": channel})
                    full = ""
                    for item in engine.stream(prompt, use_adapter=(channel == "adapter")):
                        if isinstance(item, tuple):
                            full = item[1]
                        else:
                            emit({"event": "delta", "channel": channel, "text": item})
                    emit({
                        "event": "done", "channel": channel,
                        "refused": demo.is_refusal(full),
                        "template": bool(demo.TEMPLATE.search(full)),
                    })
                emit({"event": "end"})
            except (BrokenPipeError, ConnectionResetError):
                pass  # the tab was closed mid-generation; nothing to clean up
            except Exception as error:
                try:
                    emit({"error": f"{type(error).__name__}: {error}"})
                except Exception:
                    pass

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base", help="local checkpoint dir or Hub id")
    parser.add_argument("--pointer", type=Path, default=Path("/lab/state/current-model.json"))
    parser.add_argument("--adapter", type=Path, default=HERE / "adapter")
    parser.add_argument("--prompts", type=Path, default=HERE / "prompts.json")
    parser.add_argument("--provenance", type=Path, default=HERE / "PROVENANCE.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--unsafe-expressive", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--open", action="store_true", help="open a browser tab")
    args = parser.parse_args()

    if not args.adapter.exists():
        raise SystemExit(f"adapter not found: {args.adapter}")
    if not args.skip_verify:
        demo.check_adapter(args.adapter, args.provenance)

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"\n  !! binding to {args.host} -- this serves a model with its safety")
        print("     training removed to every host that can reach this port.")
        print("     Do not do this on a conference network.\n")

    engine = Engine(demo.resolve_base(args.base, args.pointer), args.adapter, args.max_new_tokens)
    token = secrets.token_urlsafe(16)
    handler = build_handler(engine, token, args.prompts, args.provenance, args.unsafe_expressive)

    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/?k={token}"
    print(f"\n  ready: {url}")
    if not args.unsafe_expressive:
        print("  expressive-harm prompts withheld (--unsafe-expressive to unlock)")
    # Explicit flush: stdout is block-buffered when piped to a log or tee, and
    # the URL carries the token -- it is useless if it only appears at exit.
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
