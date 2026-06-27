#!/usr/bin/env python3
"""
jimmy_proxy.py — an Anthropic-compatible shim in front of the ChatJimmy API.

Claude Code (and anything else that speaks Anthropic's Messages API) can point at
this server via ANTHROPIC_BASE_URL. Requests are translated into ChatJimmy's
`/api/chat` format, and Jimmy's raw text stream is re-emitted as Anthropic SSE
events so the harness feels the ~14k tok/s decode speed.

Jimmy serves a small model (llama3.1-8B) that does NOT implement Anthropic's
native tool-calling. To make the harness actually usable, "agent mode" papers
over that:
  * We replace Claude Code's huge system prompt with a tight, tailored one and a
    compact catalog generated from the real tools in each request, plus a
    few-shot example, teaching the model to emit tool calls as a simple line of
    JSON:  {"tool_call": {"name": "...", "input": {...}}}
  * We parse the model's output back into real Anthropic `tool_use` streaming
    events, so the harness executes the tool and feeds the result back (which we
    render in the same taught format to close the loop).
  * Plain prose answers still stream live, so you keep the speed feel.

Standard library only. Run:  python3 jimmy_proxy.py
See the bottom of the file for the exact `claude` invocation.
"""

import codecs
import json
import os
import re
import sys
import uuid
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ----------------------------------------------------------------------------
# Config (all overridable via environment variables)
# ----------------------------------------------------------------------------
HOST = os.environ.get("JIMMY_PROXY_HOST", "127.0.0.1")
PORT = int(os.environ.get("JIMMY_PROXY_PORT", "8787"))
JIMMY_URL = os.environ.get("JIMMY_URL", "https://chatjimmy.ai/api/chat")
JIMMY_MODEL = os.environ.get("JIMMY_MODEL", "llama3.1-8B")
JIMMY_TOPK = int(os.environ.get("JIMMY_TOPK", "8"))
# Jimmy runs a small model with a limited context. Trim the oldest turns to stay
# under this many characters of input (0 = disable trimming).
MAX_INPUT_CHARS = int(os.environ.get("JIMMY_MAX_INPUT_CHARS", "14000"))
# Agent mode: teach the model to call tools + translate its output back. On by
# default; set JIMMY_AGENT_MODE=0 for a raw text passthrough.
AGENT_MODE = os.environ.get("JIMMY_AGENT_MODE", "1") not in ("0", "false", "no")
# Optionally replace the system prompt entirely (overrides agent-mode prompt).
SYSTEM_OVERRIDE = os.environ.get("JIMMY_SYSTEM_OVERRIDE")
# Truncate each individual tool result we feed back to the model.
MAX_TOOL_RESULT_CHARS = int(os.environ.get("JIMMY_MAX_TOOL_RESULT_CHARS", "4000"))
# A small model wanders when shown 15+ tools, but is accurate with a handful.
# Cap how many tools we put in the catalog (core ones first). 0 = no cap.
MAX_TOOLS = int(os.environ.get("JIMMY_MAX_TOOLS", "10"))

STATS_MARKER = "<|stats|>"
STATS_END = "<|/stats|>"

# Small models tend to role-play the whole conversation (writing the next
# "User:"/"Assistant:" turn themselves) instead of stopping. Jimmy has no stop
# param, so we enforce these client-side: output is cut at the first match.
DEFAULT_STOPS = [
    "\nUser:", "\nAssistant:", "\nHuman:", "\nuser:", "\nassistant:",
    "\n\nUser", "\n\nAssistant", "\n\nHuman", "\nTool result for",
    "<|eot_id|>", "<|start_header_id|>",
]

# ----------------------------------------------------------------------------
# Agent-mode system prompt
# ----------------------------------------------------------------------------
AGENT_PREAMBLE = r"""You are Claude Code, an AI coding assistant running in a terminal agent harness on the user's machine. You complete tasks by calling tools, reading their results, and then answering.

# How to respond
On each turn you do exactly ONE of these:

- CALL A TOOL: your entire reply is a single line of raw JSON, in exactly this shape:
{"tool_call": {"name": "<ToolName>", "input": {<arguments for that tool>}}}

- GIVE YOUR FINAL ANSWER: your entire reply is plain natural-language text, with no JSON.

Hard rules:
- When calling a tool, output ONLY the JSON object. No label, no prefix, no explanation, no markdown, no code fences — the whole reply is just `{"tool_call": ...}` and nothing before or after it.
- A reply is either pure JSON (a tool call) or pure prose (the final answer). Never mix them.
- Use ONLY the tools listed below, with their exact names and exact argument names. A "*" after an argument name means it is required.
- Derive every argument from the user's actual request and the real working directory shown below. Never invent placeholder paths, filenames, or arguments, and never include arguments that are not in the tool's list.
- NEVER write tool results yourself or guess what a tool returns. Emit the tool call and stop — the harness runs it and gives you the result on your next turn.
- Take one step at a time: call a tool, wait for its result, then decide the next step. When the task is finished or the question needs no tools, reply with a short plain-text final answer.

# Tools
%%CATALOG%%

# Working context
%%ENV%%

# Examples
Each example shows ONE reply you would send. Output only the content shown — never write "User:" or "Assistant:" lines, and never continue the conversation yourself.

To run a shell command, your entire reply is just:
{"tool_call": {"name": "Bash", "input": {"command": "ls *.py"}}}

To read a file, your entire reply is just:
{"tool_call": {"name": "Read", "input": {"file_path": "app.py"}}}

After the harness runs the tool and gives you the result, reply with the final answer as plain text only, for example:
The folder contains two Python files: app.py and utils.py.

Then STOP. Do not write another turn.
"""


# ----------------------------------------------------------------------------
# Content flattening
# ----------------------------------------------------------------------------
def _text_from_content(content):
    """Flatten an Anthropic message `content` (str or list of blocks) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "tool_use":
            name = block.get("name", "tool")
            inp = json.dumps(block.get("input", {}), ensure_ascii=False)
            parts.append(f"[assistant called tool `{name}` with input {inp}]")
        elif btype == "tool_result":
            inner = _text_from_content(block.get("content"))
            parts.append(f"[tool result: {inner}]")
        elif btype == "image":
            parts.append("[image omitted — Jimmy is text-only]")
        elif btype == "thinking":
            parts.append(block.get("thinking", ""))
        else:
            parts.append(block.get("text", ""))
    return "\n".join(p for p in parts if p)


def _system_to_text(system):
    if not system:
        return ""
    if isinstance(system, str):
        return system
    return _text_from_content(system)


# ----------------------------------------------------------------------------
# Agent-mode: build the tailored system prompt from the request's real tools
# ----------------------------------------------------------------------------
def render_tool(t):
    name = t.get("name", "")
    desc = (t.get("description") or "").strip().split("\n")[0]
    if len(desc) > 200:
        desc = desc[:200] + "…"
    schema = t.get("input_schema") or {}
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    parts = []
    for i, (pname, pinfo) in enumerate(props.items()):
        if i >= 14:
            parts.append("…")
            break
        ptype = (pinfo or {}).get("type", "any") if isinstance(pinfo, dict) else "any"
        mark = "*" if pname in required else ""
        parts.append(f"{pname}{mark}: {ptype}")
    pstr = ", ".join(parts) if parts else "(no arguments)"
    return f"- {name}: {desc}\n    input: {{{pstr}}}"


def get_env_context(cc_system):
    """Mine the useful environment block (cwd, platform, git, date) out of Claude
    Code's original system prompt so the small model has real context."""
    if not cc_system:
        return "(no environment info provided)"
    m = re.search(r"<env>(.*?)</env>", cc_system, re.S)
    if m:
        return m.group(1).strip()[:800]
    keys = ("working directory", "platform", "os version", "git repo",
            "today's date", "cwd", "is a git repo")
    lines = [ln.strip() for ln in cc_system.splitlines()
             if any(k in ln.lower() for k in keys)]
    return ("\n".join(lines)[:800]) or "(no environment info provided)"


# Core tools a small model should reach for first; surfaced at the top of the
# catalog so it doesn't wander off to exotic ones.
_CORE_ORDER = ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "LS", "TodoWrite"]


def build_agent_system(tools, cc_system):
    rank = {n: i for i, n in enumerate(_CORE_ORDER)}
    ordered = sorted(tools, key=lambda t: rank.get(t.get("name"), len(_CORE_ORDER)))
    if MAX_TOOLS > 0:
        ordered = ordered[:MAX_TOOLS]
    catalog = "\n".join(render_tool(t) for t in ordered) or "(none)"
    env = get_env_context(cc_system)
    return AGENT_PREAMBLE.replace("%%CATALOG%%", catalog).replace("%%ENV%%", env)


def flatten_messages_agent(messages):
    """Render the conversation in the taught protocol: assistant tool_use blocks
    become {"tool_call": ...} JSON, and tool_result blocks become readable
    'Tool result for <name>' text, so the loop stays consistent for the model."""
    id2name = {}
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    id2name[b.get("id")] = b.get("name", "tool")

    out = []
    for m in messages:
        role = m.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"
        text = _render_msg_agent(m.get("content"), id2name)
        if text:
            out.append({"role": role, "content": text})
    return out


def _render_msg_agent(content, id2name):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for b in content:
        if not isinstance(b, dict):
            parts.append(str(b))
            continue
        t = b.get("type")
        if t == "text":
            parts.append(b.get("text", ""))
        elif t == "tool_use":
            call = {"tool_call": {"name": b.get("name", "tool"),
                                  "input": b.get("input", {})}}
            parts.append(json.dumps(call, ensure_ascii=False))
        elif t == "tool_result":
            name = id2name.get(b.get("tool_use_id"), "tool")
            inner = _text_from_content(b.get("content"))
            if len(inner) > MAX_TOOL_RESULT_CHARS:
                inner = inner[:MAX_TOOL_RESULT_CHARS] + "\n…[truncated]"
            if b.get("is_error"):
                parts.append(f"Tool result for {name} (ERROR):\n{inner}")
            else:
                parts.append(f"Tool result for {name}:\n{inner}")
        elif t == "image":
            parts.append("[image omitted — Jimmy is text-only]")
        elif t == "thinking":
            pass
        else:
            parts.append(b.get("text", ""))
    return "\n".join(p for p in parts if p)


# ----------------------------------------------------------------------------
# Anthropic request -> Jimmy request translation
# ----------------------------------------------------------------------------
def build_jimmy_payload(anthropic_req):
    """Returns (jimmy_payload, agent_active, stops)."""
    tools = anthropic_req.get("tools") or []
    agent_active = AGENT_MODE and bool(tools)
    cc_system = _system_to_text(anthropic_req.get("system"))

    if SYSTEM_OVERRIDE is not None:
        system_prompt = SYSTEM_OVERRIDE
    elif agent_active:
        system_prompt = build_agent_system(tools, cc_system)
    else:
        system_prompt = cc_system

    messages = anthropic_req.get("messages", [])
    if agent_active:
        jimmy_messages = flatten_messages_agent(messages)
    else:
        jimmy_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            if role not in ("user", "assistant"):
                role = "user"
            text = _text_from_content(msg.get("content"))
            if text:
                jimmy_messages.append({"role": role, "content": text})

    # Trim oldest turns to stay under the context budget; keep the final turn.
    if MAX_INPUT_CHARS > 0:
        budget = MAX_INPUT_CHARS - len(system_prompt)
        while len(jimmy_messages) > 1 and \
                sum(len(m["content"]) for m in jimmy_messages) > budget:
            jimmy_messages.pop(0)
        if jimmy_messages and len(jimmy_messages[-1]["content"]) > max(budget, 2000):
            jimmy_messages[-1]["content"] = jimmy_messages[-1]["content"][-max(budget, 2000):]

    model = anthropic_req.get("model") or JIMMY_MODEL
    if "claude" in model.lower() or "/" in model:
        model = JIMMY_MODEL

    payload = {
        "messages": jimmy_messages,
        "chatOptions": {
            "selectedModel": model,
            "systemPrompt": system_prompt,
            "topK": JIMMY_TOPK,
        },
        "attachment": None,
    }
    stops = list(DEFAULT_STOPS)
    for s in (anthropic_req.get("stop_sequences") or []):
        if isinstance(s, str) and s:
            stops.append(s)
    return payload, agent_active, stops


def estimate_tokens(text):
    return max(1, len(text) // 4)


# ----------------------------------------------------------------------------
# Parsing the model's output back into tool calls
# ----------------------------------------------------------------------------
def _try_json(s):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def _first_json_object(s):
    """Extract the first brace-balanced {...} substring, respecting strings."""
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
    return None


def _extract_json_loose(s):
    """Return a parseable JSON object substring starting at the first '{'. If the
    model truncated the call (a common small-model failure), repair it by closing
    any open strings/brackets/braces."""
    start = s.find("{")
    if start < 0:
        return None
    candidate = s[start:]
    balanced = _first_json_object(candidate)
    if balanced is not None:
        return balanced
    depth_c = depth_s = 0
    in_str = esc = False
    for ch in candidate:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth_c += 1
            elif ch == "}":
                depth_c -= 1
            elif ch == "[":
                depth_s += 1
            elif ch == "]":
                depth_s -= 1
    repaired = candidate
    if in_str:
        repaired += '"'
    repaired += "]" * max(0, depth_s)
    repaired += "}" * max(0, depth_c)
    return repaired


def _normalize_call(d):
    if not isinstance(d, dict):
        return None
    if "tool_call" in d and isinstance(d["tool_call"], dict):
        d = d["tool_call"]
    name = d.get("name") or d.get("tool") or d.get("tool_name")
    if not name or not isinstance(name, str):
        return None
    inp = d.get("input")
    if inp is None:
        inp = d.get("parameters")
    if inp is None:
        inp = d.get("arguments")
    if inp is None:
        inp = {}
    if isinstance(inp, str):
        parsed = _try_json(inp)
        inp = parsed if isinstance(parsed, dict) else {}
    if not isinstance(inp, dict):
        inp = {}
    return {"name": name, "input": inp}


_LABEL_RE = re.compile(
    r"^(\([A-Za-z0-9]\)|assistant|tool[_ ]?call|here(?:'s| is)[^{]{0,40}"
    r"|the (?:tool )?call[^{]{0,40})[\s:.\-]*", re.I)


def looks_like_tool_head(stripped):
    """Decide from the start of the output whether this turn is a tool call
    (vs a plain-prose final answer), tolerating a leading label the model may
    have parroted (e.g. '(A) ', 'Assistant:', 'Here is the call:')."""
    if not stripped:
        return False
    s2 = _LABEL_RE.sub("", stripped)
    if s2[:1] in "{[" or s2.startswith("<|python_tag|>") or stripped.startswith("<|python_tag|>"):
        return True
    head = stripped[:160]
    if '"tool_call"' in head:
        return True
    if re.search(r'"name"\s*:\s*"', head) and re.search(r'"(input|parameters|arguments)"', head):
        return True
    return False


def extract_tool_calls(raw):
    """Try to parse `raw` as one or more tool calls. Returns a list or None."""
    s = raw.strip()
    s = s.replace("<|python_tag|>", "")
    for tok in ("<|eom_id|>", "<|eot_id|>", "<|eom|>", "<|start_header_id|>"):
        s = s.replace(tok, "")
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s[:4].lower() == "json":
            s = s[4:].strip()

    parsed = _try_json(s)
    if parsed is None:
        frag = _extract_json_loose(s)
        parsed = _try_json(frag) if frag else None
    if parsed is None:
        return None

    items = parsed if isinstance(parsed, list) else [parsed]
    calls = []
    for it in items:
        norm = _normalize_call(it)
        if norm:
            calls.append(norm)
    return calls or None


def apply_stops(pairs, stops):
    """Wrap a ('delta'|'stats', value) stream and cut the text at the first
    occurrence of any stop string, closing the upstream early."""
    if not stops:
        yield from pairs
        return
    maxlen = max(len(s) for s in stops)
    pending = ""
    stats = None
    stopped = False
    for kind, value in pairs:
        if kind == "stats":
            stats = value
            continue
        pending += value
        hits = [pending.find(s) for s in stops if s in pending]
        if hits:
            idx = min(hits)
            if idx > 0:
                yield ("delta", pending[:idx])
            stopped = True
            break
        keep = maxlen - 1
        if len(pending) > keep:
            yield ("delta", pending[:-keep])
            pending = pending[-keep:]
    if stopped:
        try:
            pairs.close()   # triggers stream_jimmy's finally -> close socket
        except Exception:  # noqa: BLE001
            pass
    elif pending:
        yield ("delta", pending)
    yield ("stats", stats)


def map_stop_reason(stats):
    if not stats:
        return "end_turn"
    reason = (stats.get("done_reason") or "").lower()
    if reason in ("length", "max_tokens"):
        return "max_tokens"
    return "end_turn"


# ----------------------------------------------------------------------------
# HTTP handler: Anthropic-compatible surface
# ----------------------------------------------------------------------------
def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[jimmy-proxy] " + (fmt % args) + "\n")

    def _read_body(self):
        length = int(self.headers.get("content-length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return {}

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message, status=500, etype="api_error"):
        self._send_json(
            {"type": "error", "error": {"type": etype, "message": message}}, status
        )

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("content-length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/") in ("", "/health"):
            self._send_json({"status": "ok", "upstream": JIMMY_URL,
                             "model": JIMMY_MODEL, "agent_mode": AGENT_MODE})
        else:
            self._error("not found", 404, "not_found_error")

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/v1/messages":
            self._handle_messages()
        elif path == "/v1/messages/count_tokens":
            self._handle_count_tokens()
        else:
            self._error(f"unknown path {self.path}", 404, "not_found_error")

    def _handle_count_tokens(self):
        req = self._read_body()
        text = _system_to_text(req.get("system"))
        for msg in req.get("messages", []):
            text += "\n" + _text_from_content(msg.get("content"))
        self._send_json({"input_tokens": estimate_tokens(text)})

    def _handle_messages(self):
        req = self._read_body()
        wants_stream = bool(req.get("stream"))
        model = req.get("model") or JIMMY_MODEL
        try:
            payload, agent_active, stops = build_jimmy_payload(req)
        except Exception as e:  # noqa: BLE001
            self._error(f"failed to translate request: {e}", 400, "invalid_request_error")
            return

        input_tokens = estimate_tokens(
            payload["chatOptions"]["systemPrompt"]
            + "".join(m["content"] for m in payload["messages"])
        )
        self.log_message("msgs=%d tools=%d agent=%s ~in_tok=%d",
                         len(payload["messages"]), len(req.get("tools") or []),
                         agent_active, input_tokens)

        if wants_stream:
            self._stream_messages(payload, model, input_tokens, agent_active, stops)
        else:
            self._buffered_messages(payload, model, input_tokens, agent_active, stops)

    # -- streaming --------------------------------------------------------
    def _stream_messages(self, payload, model, input_tokens, agent_active, stops):
        msg_id = "msg_" + uuid.uuid4().hex[:24]
        try:
            gen = apply_stops(stream_jimmy(payload), stops)
        except urllib.error.URLError as e:
            self._error(f"upstream connection failed: {e}", 502, "api_error")
            return

        self.send_response(200)
        self.send_header("content-type", "text/event-stream; charset=utf-8")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "keep-alive")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()

        def write(b):
            self.wfile.write(f"{len(b):X}\r\n".encode("ascii") + b + b"\r\n")
            self.wfile.flush()

        def end_chunks():
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        try:
            write(sse("message_start", {
                "type": "message_start",
                "message": {
                    "id": msg_id, "type": "message", "role": "assistant",
                    "model": model, "content": [],
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            }))
            write(sse("ping", {"type": "ping"}))

            # Peek/buffer state machine: stream prose live, but buffer when the
            # output looks like a tool call so we can translate it. We hold back
            # only until we can read the first line, so prose still streams fast.
            mode = None          # None -> undecided, 'text', or 'buffer'
            buf = ""
            stats = None

            def start_text():
                write(sse("content_block_start", {
                    "type": "content_block_start", "index": 0,
                    "content_block": {"type": "text", "text": ""}}))
                if buf:
                    write(sse("content_block_delta", {
                        "type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": buf}}))

            def decide(force):
                nonlocal mode
                stripped = buf.lstrip()
                if not stripped:
                    if force:
                        mode = "text"
                        start_text()
                    return
                # Wait for a full first line (or enough chars) before deciding,
                # unless the stream has ended (force).
                if not force and "\n" not in buf and len(stripped) < 60:
                    return
                if agent_active and looks_like_tool_head(stripped):
                    mode = "buffer"   # keep accumulating buf; translate at end
                else:
                    mode = "text"
                    start_text()

            for kind, value in gen:
                if kind == "stats":
                    stats = value if isinstance(value, dict) else None
                    continue
                # kind == "delta"
                if mode is None:
                    buf += value
                    decide(False)
                elif mode == "text":
                    write(sse("content_block_delta", {
                        "type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": value}}))
                else:  # buffer
                    buf += value

            if mode is None:
                decide(True)

            out_tokens = (stats or {}).get("decode_tokens") or 0

            if mode == "buffer":
                calls = extract_tool_calls(buf)
                if calls:
                    self._emit_tool_calls(write, calls)
                    stop_reason = "tool_use"
                else:
                    # Not actually a tool call — emit the buffer as text.
                    write(sse("content_block_start", {
                        "type": "content_block_start", "index": 0,
                        "content_block": {"type": "text", "text": ""}}))
                    write(sse("content_block_delta", {
                        "type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": buf}}))
                    write(sse("content_block_stop", {
                        "type": "content_block_stop", "index": 0}))
                    stop_reason = map_stop_reason(stats)
            else:
                if mode is None:
                    # Empty response — emit an empty text block.
                    write(sse("content_block_start", {
                        "type": "content_block_start", "index": 0,
                        "content_block": {"type": "text", "text": ""}}))
                write(sse("content_block_stop", {
                    "type": "content_block_stop", "index": 0}))
                stop_reason = map_stop_reason(stats)

            write(sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": out_tokens}}))
            write(sse("message_stop", {"type": "message_stop"}))
            end_chunks()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:  # noqa: BLE001
            try:
                write(sse("error", {"type": "error",
                                    "error": {"type": "api_error",
                                              "message": str(e)}}))
                end_chunks()
            except Exception:  # noqa: BLE001
                pass

    def _emit_tool_calls(self, write, calls):
        for idx, call in enumerate(calls):
            tool_id = "toolu_" + uuid.uuid4().hex[:24]
            write(sse("content_block_start", {
                "type": "content_block_start", "index": idx,
                "content_block": {"type": "tool_use", "id": tool_id,
                                  "name": call["name"], "input": {}}}))
            write(sse("content_block_delta", {
                "type": "content_block_delta", "index": idx,
                "delta": {"type": "input_json_delta",
                          "partial_json": json.dumps(call["input"])}}))
            write(sse("content_block_stop", {
                "type": "content_block_stop", "index": idx}))

    # -- non-streaming ----------------------------------------------------
    def _buffered_messages(self, payload, model, input_tokens, agent_active, stops):
        try:
            text_parts = []
            stats = None
            for kind, value in apply_stops(stream_jimmy(payload), stops):
                if kind == "delta":
                    text_parts.append(value)
                elif kind == "stats":
                    stats = value if isinstance(value, dict) else None
        except urllib.error.URLError as e:
            self._error(f"upstream connection failed: {e}", 502, "api_error")
            return

        full = "".join(text_parts)
        out_tokens = (stats or {}).get("decode_tokens") or estimate_tokens(full)
        content = None
        stop_reason = map_stop_reason(stats)

        if agent_active and full.lstrip()[:1] in "{[<`":
            calls = extract_tool_calls(full)
            if calls:
                content = [{"type": "tool_use",
                            "id": "toolu_" + uuid.uuid4().hex[:24],
                            "name": c["name"], "input": c["input"]} for c in calls]
                stop_reason = "tool_use"
        if content is None:
            content = [{"type": "text", "text": full}]

        self._send_json({
            "id": "msg_" + uuid.uuid4().hex[:24],
            "type": "message", "role": "assistant", "model": model,
            "content": content, "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": out_tokens},
        })


# ----------------------------------------------------------------------------
# Calling Jimmy and consuming its stream
# ----------------------------------------------------------------------------
def stream_jimmy(payload):
    """Yield ('delta', text) tuples from Jimmy, stripping the trailing
    <|stats|>..<|/stats|> block, then a final ('stats', dict|None)."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        JIMMY_URL,
        data=data,
        method="POST",
        headers={
            "content-type": "application/json",
            "accept": "*/*",
            "origin": "https://chatjimmy.ai",
            "referer": "https://chatjimmy.ai/",
            "user-agent": "jimmy-proxy/1.0",
        },
    )
    resp = urllib.request.urlopen(req, timeout=120)

    decoder = codecs.getincrementaldecoder("utf-8")()
    pending = ""
    stats_raw = ""
    in_stats = False

    try:
        while True:
            chunk = resp.read(2048)
            if not chunk:
                pending += decoder.decode(b"", final=True)
                break
            pending += decoder.decode(chunk)

            if not in_stats:
                idx = pending.find(STATS_MARKER)
                if idx >= 0:
                    if idx > 0:
                        yield ("delta", pending[:idx])
                    stats_raw = pending[idx:]
                    pending = ""
                    in_stats = True
                else:
                    keep = len(STATS_MARKER) - 1
                    if len(pending) > keep:
                        yield ("delta", pending[:-keep])
                        pending = pending[-keep:]
            else:
                stats_raw += pending
                pending = ""
    finally:
        resp.close()

    if not in_stats:
        idx = pending.find(STATS_MARKER)
        if idx >= 0:
            if idx > 0:
                yield ("delta", pending[:idx])
            stats_raw = pending[idx:]
        elif pending:
            yield ("delta", pending)
    else:
        stats_raw += pending

    stats = None
    if stats_raw:
        inner = stats_raw
        if inner.startswith(STATS_MARKER):
            inner = inner[len(STATS_MARKER):]
        end = inner.find(STATS_END)
        if end >= 0:
            inner = inner[:end]
        stats = _try_json(inner)
    yield ("stats", stats)


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"jimmy-proxy listening on http://{HOST}:{PORT}")
    print(f"  upstream   : {JIMMY_URL}")
    print(f"  model      : {JIMMY_MODEL}  (topK={JIMMY_TOPK})")
    print(f"  agent mode : {'ON' if AGENT_MODE else 'off'}")
    print()
    print("Point Claude Code at it:")
    print(f"  ANTHROPIC_BASE_URL=http://{HOST}:{PORT} \\")
    print(f"  ANTHROPIC_API_KEY=dummy \\")
    print(f"  ANTHROPIC_MODEL={JIMMY_MODEL} \\")
    print(f"  ANTHROPIC_SMALL_FAST_MODEL={JIMMY_MODEL} \\")
    print("  claude")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
