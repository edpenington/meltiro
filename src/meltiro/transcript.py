"""One readable Markdown document for one extraction session.

*meltiro* keeps its record of a run in several machine-shaped files: the event
log (`diagnostics/tool_calls.jsonl`), the instrument as sent
(`diagnostics/instrument/`), the wire log (`diagnostics/api_calls.jsonl`), the
session summary (`diagnostics/run.json`), and the derived per-field history
(`diagnostics/field_history.json`). Each answers a machine's question well and
a reader's question badly: following one field through a run means opening
four files and correlating them by timestamp.

This module renders those files into a single document meant to be read start
to finish, in the order the run happened, so it broadly mirrors what a model
had in front of it at each point. It renders, it never computes: every
statement in the document is read off a file the run wrote, and nothing here
re-derives a value the session did not record.

Two entry points, and they are the same code path so the two copies can never
diverge:

  - `render_transcript(session_dir)` returns the document as a string. This is
    what `meltiro transcript SESSION_DIR --out FILE` calls, so a session can be
    re-rendered later, after the renderer improves, without paying for the run
    again.
  - `write_transcript(session_dir)` writes it to
    `{session_dir}/diagnostics/transcript.md`. The orchestrator calls this at
    finalisation and at a pause.

The document is a pure function of the session directory on disk. It carries
no rendering-time state (no "generated at" stamp, no host, no invocation cwd),
which is what makes re-rendering an unchanged session byte-identical to the
copy the run itself wrote.

What it deliberately does NOT render is `api_calls.jsonl`'s request and
response bodies. They are the same prompts, the same tool definitions, and the
same turns the document already shows in full, and reprinting the whole
conversation once per turn (which is what a wire log is) would bury the run
rather than show it. The one thing read out of the wire log is each checker
call's rendered user message, which exists nowhere else. Nothing else in this
document is summarised, elided, or truncated.

Degrading by diagnostics level is the one thing this module must get right,
because the honest answer to "why is the instrument not here" is different
from "the run did not have one". The level a session recorded governs what
this document can show, and every absence names the level that caused it (see
`_capability_notes`).
"""

import json
from pathlib import Path

from meltiro.diagnostics import (
    captures_api_calls, captures_instrument, validate_diagnostics)
from meltiro.errors import SessionError
from meltiro.field_history import build_field_history
from meltiro.rates import cost_with_coverage

# The document `extract` writes at every stop, inside the session's
# diagnostics directory: it is a record of the run, not the run's product.
TRANSCRIPT_FILENAME = "transcript.md"

# Event names that belong to the final reviewer's own conversation. The
# reviewer's turns also carry `stage: "review"` on the events it shares names
# with the extractor (`assistant_message`, `assistant_text`), so the classifier
# below reads either signal.
_REVIEW_EVENTS = frozenset({
    "review_tool_call",
    "review_reprompt",
    "review_cap_hit",
    "review_error",
    "review_abandoned",
    "review_repeated_failure_stall",
    "review_text_only_stall",
    "final_review_response",
    "final_review_edits_none_applied",
    "final_review_no_response",
})

# Events that carry a turn's content. Everything else is a run-level note and
# is rendered in log order between the turns.
_TURN_CONTENT_EVENTS = frozenset({
    "assistant_message",
    "assistant_text",
    "tool_call_applied",
    "tool_call_partial",
    "tool_call_failed",
    "review_tool_call",
    "final_review_response",
})

_TOOL_CALL_EVENTS = frozenset({
    "tool_call_applied",
    "tool_call_partial",
    "tool_call_failed",
    "review_tool_call",
})

# How a dispatch status reads in prose, for a MUTATING call. The status
# strings themselves come from the dispatcher and are quoted verbatim next to
# these. A read-only `view_*` call writes nothing, so none of these describes
# it and `_render_tool_call` says so instead.
# The three statuses `ToolDispatcher._result` can carry, and nothing else: a
# key here that the dispatcher never emits is a gloss no reader will ever see,
# and a status missing from here renders bare in a document whose whole job is
# to be readable without the source. Kept in step with `tools.ToolDispatcher`,
# which decides all three from the fields alone (a scope note can neither fail
# a call nor rescue one).
_STATUS_GLOSS = {
    "ok": "every field in the call applied",
    "partial": "some fields applied, some were rejected",
    "validation_failed": "every field was rejected; nothing applied",
}


# ---------------------------------------------------------------------------
# Reading the session
# ---------------------------------------------------------------------------

class _Session:
    """Everything this module reads, loaded once, from disk only.

    Reading from disk rather than from a live `Session` object is what makes
    the two entry points agree: the orchestrator's in-memory meta can be a
    write ahead of the file, and a document rendered from it would then differ
    from the same document rendered later by the subcommand.
    """

    def __init__(self, session_dir):
        self.dir = Path(session_dir).resolve()
        if not self.dir.exists():
            raise SessionError(
                f"no such session directory: {self.dir}. `meltiro transcript` "
                "takes a SESSION directory, the one holding "
                "extraction_output.json and diagnostics/, at "
                "{out}/{study_id}/sessions/{timestamp}_{fp}/."
            )
        if not self.dir.is_dir():
            raise SessionError(
                f"session path is not a directory: {self.dir}.")
        self.diagnostics_dir = self.dir / "diagnostics"
        self.meta = self._require_json(
            self.diagnostics_dir / "run.json",
            "the session summary every run writes at every diagnostics level",
        )
        if not isinstance(self.meta, dict):
            raise SessionError(
                f"{self.diagnostics_dir / 'run.json'} is not a JSON object. "
                "This is not a session directory *meltiro* wrote.")
        # Loud rather than defaulted: a session whose run.json does not say
        # what it kept cannot be described honestly, and guessing a level
        # would put claims in the document that no file backs.
        self.level = validate_diagnostics(self.meta.get("diagnostics"))
        self.events = self._read_events()
        self.output = self._require_json(
            self.dir / "extraction_output.json",
            "the extraction itself, written at every diagnostics level",
        )
        self.field_history, self.field_history_derived = \
            self._read_field_history()
        self.instrument = self._read_instrument()
        self.api_calls = self._read_api_calls()

    # -- loaders ----------------------------------------------------------

    @staticmethod
    def _require_json(path, what):
        if not path.exists():
            raise SessionError(
                f"missing {path}: {what}. Either this is not a *meltiro* "
                "session directory, or the session is incomplete."
            )
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SessionError(f"could not read {path}: {exc}") from exc

    def _read_events(self):
        path = self.diagnostics_dir / "tool_calls.jsonl"
        if not path.exists():
            raise SessionError(
                f"missing {path}: the event log is the run's memory and is "
                "kept at every diagnostics level, so a session without one "
                "cannot be transcribed."
            )
        events = []
        for i, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SessionError(
                    f"malformed JSON on line {i} of {path}: {exc}. The "
                    "transcript is rendered from the whole log or not at all."
                ) from exc
        return events

    def _read_field_history(self):
        """The per-field history, preferring the file the run wrote.

        The file is written whenever a run stops, so a finished or paused
        session always has one. A session killed mid-run does not, and there
        the history is derived here from the event log, by the same pure
        function that wrote the file. The flag says which, because a derived
        one describes the log as it stands rather than as the run left it.
        """
        path = self.diagnostics_dir / "field_history.json"
        if path.exists():
            return self._require_json(path, "the per-field history"), False
        return build_field_history(self.events), True

    def _read_instrument(self):
        d = self.diagnostics_dir / "instrument"
        out = {}
        for key, name, is_json in (
            ("extractor_system", "system_prompt.txt", False),
            ("extractor_user", "user_prompt.txt", False),
            ("review_system", "review_system_prompt.txt", False),
            ("checker_system", "checker_system_prompt.txt", False),
            ("tools", "tool_definitions.json", True),
            ("image_labels", "image_labels.json", True),
        ):
            path = d / name
            if not path.exists():
                out[key] = None
                continue
            if is_json:
                out[key] = self._require_json(path, f"instrument/{name}")
            else:
                out[key] = path.read_text(encoding="utf-8")
        return out

    def _read_api_calls(self):
        path = self.diagnostics_dir / "api_calls.jsonl"
        if not path.exists():
            return []
        calls = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                calls.append(json.loads(line))
            except json.JSONDecodeError:
                # The wire log is written per call under a lock and nothing
                # reads it back, so a torn line here loses one call's verbatim
                # copy and no more. The transcript says so where the copy
                # would have gone rather than refusing the whole document.
                continue
        return calls

    # -- convenience ------------------------------------------------------

    def checker_messages(self):
        """`{(field_path, check_index): [ask per call]}` from the wire log.

        Each ask is a list of `(role, text)` segments in message order, as
        `_ask_segments_from_request` reads them off the logged request: one
        segment for a first ask, and three for a re-ask that replayed the reply
        it corrects.

        The checker's per-field user message is rendered at call time and
        stored nowhere but the wire log, so this map is empty at any level
        below `full`. `check_index` disambiguates a field checked more than
        once, and `ask` disambiguates the calls WITHIN one check: a reply that
        recorded no verdict is re-asked, and both asks were sent, billed,
        and logged. They are collected as a list in ask order rather than
        keyed by (field, check) alone, which would keep whichever landed last
        and hide the ask that made the re-ask necessary. Every key is logged on
        the call by `checker.check_one_field`.
        """
        out = {}
        for call in self.api_calls:
            if call.get("call_type") != "checker":
                continue
            field_path = call.get("field_path")
            if not field_path:
                continue
            segments = _ask_segments_from_request(call.get("request") or {})
            if segments is None:
                continue
            asks = out.setdefault((field_path, call.get("check_index")), [])
            # `ask` is the 0-based ask number; a log written before it was
            # recorded has none, and those keep their file order.
            asks.append((call.get("ask") or 0, segments))
        return {key: [seg for _ask, seg in sorted(asks, key=lambda a: a[0])]
                for key, asks in out.items()}


def _ask_segments_from_request(request):
    """One logged call's text as `(role, text)` per message, in message order.

    A checker call carries the per-field user message, whose blocks are the
    rendered text plus, for an image-sourced field, caption and image blocks;
    they are joined into one segment, since they are one message. A re-ask
    carries that message and the correction, and — where the reply it corrects
    was one the checker could replay — that reply between them, as an
    `assistant` segment. Kept as segments rather than joined into one string so
    that a replayed reply can be rendered as what it is: the order is the order
    the call was made in, and the model's own words are not run together with
    what it was sent. The image bytes were reduced to hashes by the wire
    logger, so what is left to render is the text.
    """
    messages = request.get("messages")
    if not isinstance(messages, list):
        return None
    segments = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        content = message.get("content")
        parts = []
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
        if parts:
            segments.append((role, "\n\n".join(parts)))
    return segments or None


# ---------------------------------------------------------------------------
# Markdown primitives
# ---------------------------------------------------------------------------

def _fence(text, lang=""):
    """Fence `text` verbatim, with a fence long enough to contain it.

    Prompts and model prose can hold their own code fences, so the fence
    length is computed from the content rather than assumed to be three.
    """
    text = "" if text is None else str(text)
    longest = run = 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    ticks = "`" * max(3, longest + 1)
    body = text if text.endswith("\n") else text + "\n"
    if not text:
        body = ""
    return f"{ticks}{lang}\n{body}{ticks}"


def _json_fence(obj):
    return _fence(json.dumps(obj, indent=2, ensure_ascii=False), "json")


def _quote(text):
    """Render model prose as a blockquote.

    A blockquote wraps (a code fence does not), and it walls off any markdown
    the model wrote so a stray heading in an assistant turn cannot restructure
    the document.
    """
    text = "" if text is None else str(text)
    return "\n".join(("> " + line).rstrip() for line in text.split("\n"))


def _anchor(name):
    return f'<a id="{name}"></a>'


def _slug(text):
    out = []
    for ch in str(text).lower():
        out.append(ch if ch.isalnum() else "-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "x"


def _cell(value):
    """One table cell's text, escaped so it cannot break the row.

    A pipe would end the cell and a newline would end the row, so both are
    neutralised. Nothing is shortened: a long value stays long, and the
    renderer wraps it rather than cutting it.
    """
    text = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, default=str)
    return text.replace("|", "\\|").replace("\n", "<br>")


def _code_span(text):
    """A code span that survives its own content.

    A value can hold backticks, so the delimiter is one longer than the
    longest run inside it, and a value that starts or ends with a backtick is
    padded, as the CommonMark rules require.
    """
    text = _cell(text)
    longest = run = 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    ticks = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{ticks}{pad}{text}{pad}{ticks}"


def _code_cell(value):
    if value is None:
        return "*(none)*"
    return _code_span(str(value))


def _value_cell(value):
    """A field value in a table cell, JSON-encoded so its type is visible: a
    string reads quoted, an absent value reads `null`, a number reads bare."""
    return _code_span(json.dumps(value, ensure_ascii=False, default=str))


def _table(headers, rows):
    if not rows:
        return ""
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _kv_table(headers, pairs):
    """A two-column table, skipping nothing: an absent value is rendered as
    absent rather than dropped, so a reader never has to wonder whether a row
    was omitted or never existed."""
    return _table(list(headers), [[k, v] for k, v in pairs])


# Role name -> the run-record key holding the model that role ran. Pricing is
# per role, so a rate card is only readable next to the model it priced.
_ROLE_MODEL_KEY = {
    "extractor": "extractor_model",
    "checker": "checker_model",
    "review": "review_model",
}

# The four counters a role's per-role usage block carries. Any of them non-zero
# means the role made a call, which is what separates "nothing priced this" from
# "this never ran".
_TOKEN_COUNTERS = ("input_tokens", "output_tokens",
                   "cache_read_tokens", "cache_write_tokens")


def _money(value):
    """A dollar figure, or a statement that there is none.

    A run without a rate card records tokens and no cost, and this says so in
    words. It never renders that as `$0.000000`, which would read as a call or a
    run that was free rather than one nothing priced.
    """
    if value is None:
        return "*(not priced)*"
    if not isinstance(value, (int, float)):
        return "*(not recorded)*"
    return f"${value:.6f}"


def _run_cost_cell(meta):
    """The run's total, and how much of the run it covers.

    A call whose charge could not be read is counted in tokens and not in
    dollars, which leaves the sum covering less than the run. Printing it bare
    would understate the run by however many calls came back without a
    receipt, so the figure carries the count of them, in the wording every
    other cost cell here uses.
    """
    cost = meta.get("cost_usd")
    if not meta.get("cost_incomplete"):
        return _money(cost)
    return cost_with_coverage(cost, _money(cost),
                              meta.get("unreceipted_calls"))


def _role_cost_cell(block):
    """One role's cost, distinguishing "nothing priced this" from "this never
    ran".

    Both states hold a null cost, and they mean opposite things: a role that
    made calls nobody could price has spend the record cannot state, while a
    role that made no calls has none to state. Reading the counters is what
    tells them apart, and a run that stopped before the reviewer's turn puts
    exactly that case in front of a reader. A latched coverage flag is read
    the same way: it is set by a call, so it is itself proof the role ran.

    The coverage rides on the ROLE's own figure as well as on the run's,
    because these are the rows a reader adds up when they want one stage's
    bill: a floor that reached this table bare would be summed as a total.
    """
    cost = block.get("cost_usd")
    incomplete = block.get("cost_incomplete")
    if cost is None and not incomplete and not any(
            block.get(counter) for counter in _TOKEN_COUNTERS):
        return "*(no calls)*"
    if not incomplete:
        return _money(cost)
    return cost_with_coverage(cost, _money(cost),
                              block.get("unreceipted_calls"))


def _checker_cost_cell(aggregate):
    """What the checker cost, and how much of the checking it covers.

    The rule the run's own total follows, applied to the checker's share: a
    check whose charge never arrived is billed and counted in tokens, so this
    sum prices fewer calls than the checker made and is stated as the floor it
    is.
    """
    cost = aggregate.get("checker_cost_usd")
    if not aggregate.get("checker_cost_incomplete"):
        return _money(cost)
    return cost_with_coverage(cost, _money(cost),
                              aggregate.get("checker_unreceipted_responses"))


def _present(value, absent="*(not recorded)*"):
    if value is None:
        return absent
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str) and not value.strip():
        return absent
    return str(value)


def _indent_continuation(text, indent="  "):
    """Keep a multi-line message inside its list item."""
    lines = str(text).split("\n")
    return "\n".join([lines[0]] + [indent + line for line in lines[1:]])


def _plural(count, singular, plural=None):
    """`1 field` / `2 fields`, so the document reads as prose rather than as
    a template with an `(s)` in it."""
    word = singular if count == 1 else (plural or singular + "s")
    return f"{count} {word}"


# A one-line lede must say something. A first sentence shorter than this is
# a lead-in rather than a summary (`Last resort.`), so the next sentence is
# taken too.
_LEDE_FLOOR = 40


def _lede(text):
    """The opening sentence or two of a description, for an index row.

    Whole sentences only, and verbatim: this abbreviates by stopping early,
    never by rewriting, so what a reader sees in the index is the opening of
    what the model was told. The full text is always one link away.
    """
    rest = " ".join(str(text or "").split())
    out = ""
    while rest and len(out) < _LEDE_FLOOR:
        stop = rest.find(". ")
        if stop == -1:
            out = (out + " " + rest).strip()
            break
        out = (out + " " + rest[:stop + 1]).strip()
        rest = rest[stop + 2:]
    return out


# ---------------------------------------------------------------------------
# The renderer
# ---------------------------------------------------------------------------

class _Renderer:

    def __init__(self, session):
        self.s = session
        self.out = []
        # Global counters, so every call and every check has an anchor a link
        # elsewhere in the document can name.
        self._call_n = 0
        self._check_n = 0
        # field path -> list of (label, anchor) in the order things happened.
        # Built while the turns are rendered, so the field index at the end is
        # the same walk rather than a second reading that could disagree.
        self._trail = {}
        self._checker_messages = session.checker_messages()
        # Per-field check ordinal, counted as the verdicts are rendered. The
        # run counts the same way (`check_index` on the wire-log entry is this
        # field's 1-based check number), so counting here is what pairs a
        # verdict with the message that produced it, including across a resume
        # where the count is rebuilt from the log rather than carried.
        self._check_ordinal = {}

    # -- emit helpers -----------------------------------------------------

    def _p(self, text=""):
        self.out.append(text)

    def _block(self, text):
        if text:
            self.out.append(text)
            self.out.append("")

    def _ask(self, segments):
        """One checker call's text, fenced, in the order the call carried it.

        A re-ask that replayed the reply it corrects carries three segments,
        two of them things the checker was sent and one of them the checker's
        own words. Those two are labelled where they appear, so a reader is
        never left to infer which of three fences the model wrote. A call with
        one segment is the field's message and nothing else, and it is
        introduced by the line above it rather than labelled again.
        """
        for i, (role, text) in enumerate(segments):
            if role == "assistant":
                self._p("the reply being corrected:")
                self._p()
            elif i:
                self._p("the correction, as a new user turn:")
                self._p()
            self._block(_fence(text, "text"))

    def _note(self, path, label, anchor):
        self._trail.setdefault(path, []).append((label, anchor))

    # -- top level --------------------------------------------------------

    def render(self):
        meta = self.s.meta
        study = _present(meta.get("study_id"), "(unknown study)")
        self._p(f"# Transcript: {study}")
        self._p()
        self._p(
            "One extraction session, from the instrument that was sent to the "
            "extraction that came out, in the order it happened. It is meant "
            "to be read start to finish: each turn shows what the model said, "
            "what it called, and what came back, so the document broadly "
            "mirrors what the model had in front of it at that point."
        )
        self._p()
        self._p(
            "*meltiro* renders this from the session directory alone. Nothing "
            "in it is inferred, reconstructed from the config bundle, or "
            "recomputed: every line is read off a file this run wrote."
        )
        self._p()
        self._render_contents()
        self._render_run()
        self._render_instrument()
        self._render_stages()
        self._render_output()
        self._render_field_history()
        self._render_tool_definitions()
        self._p("")
        text = "\n".join(self.out)
        # One trailing newline, and no accidental runs of blank lines from the
        # section builders.
        while "\n\n\n\n" in text:
            text = text.replace("\n\n\n\n", "\n\n\n")
        return text.rstrip("\n") + "\n"

    def _render_contents(self):
        self._p("## Contents")
        self._p()
        for label, anchor in (
            ("1. The run", "sec-run"),
            ("2. The instrument", "sec-instrument"),
            ("3. The extraction, turn by turn", "sec-extraction"),
            ("4. The review", "sec-review"),
            ("5. The extraction output", "sec-output"),
            ("6. What happened to each field", "sec-field-history"),
            ("7. The tool definitions in full", "sec-tool-definitions"),
        ):
            self._p(f"- [{label}](#{anchor})")
        self._p()
        self._p(
            "Sections 1 to 6 are the run, in the order it happened, and read "
            "start to finish. Section 7 is reference material: the tool "
            "schemas are long enough to bury the run, so they sit at the end "
            "and [section 2.5](#instrument-tools) indexes them."
        )
        self._p()
        self._p(
            "Turns are numbered with the run's own turn ids, so a turn here "
            "and a `turn_id` in `diagnostics/tool_calls.jsonl` are the same "
            "turn. Tool calls and checker calls are numbered across the whole "
            "document, and every one of them carries an anchor, so section 6 "
            "can link a field to every write and every check it received."
        )
        self._p()

    # -- 1. the run -------------------------------------------------------

    def _render_run(self):
        meta = self.s.meta
        self._p(_anchor("sec-run"))
        self._p("## 1. The run")
        self._p()

        self._p("### Identity and outcome")
        self._p()
        pairs = [
            ("Study id", _code_cell(meta.get("study_id"))),
            ("Session id", _code_cell(meta.get("session_id"))),
            ("Session directory", _code_cell(str(self.s.dir))),
            ("Status", _code_cell(meta.get("status"))),
            ("Phase", _code_cell(meta.get("current_phase"))),
            ("Started", _present(meta.get("started_at"))),
            ("Last written", _present(meta.get("updated_at"))),
        ]
        if meta.get("pause_reason"):
            pairs.append(("Paused because",
                          _code_cell(meta.get("pause_reason"))))
        if meta.get("failure_reason"):
            pairs.append(("Failure reason",
                          _code_cell(meta.get("failure_reason"))))
        if meta.get("error_message"):
            # The composed message of the failure that ended the run, beside
            # the status it produced. `error` names a category; this is the
            # sentence, and having it here means the outcome table answers
            # "what went wrong" without a search through the turns below.
            pairs.append(("Error", _cell(meta.get("error_message"))))
        if meta.get("failed_validation_reason"):
            pairs.append(("Stated surrender reason",
                          _cell(meta.get("failed_validation_reason"))))
        self._block(_kv_table(["Property", "Value"], pairs))

        self._p("### The engine that ran it")
        self._p()
        self._p(
            "No config fingerprint covers *meltiro*'s own prose (the framing "
            "around the config's prompts, and every tool result the "
            "dispatcher returns to a model). `engine_fp` does: it hashes both "
            "engine packages' versions together with a digest of each one's "
            "source, so it names the code that asked the question. The commit "
            "beside it names the checkout that code was read from."
        )
        self._p()
        dirty = meta.get("git_dirty")
        tree = {True: "dirty at session start",
                False: "clean at session start"}.get(
                    dirty, "*(git unavailable)*")
        self._block(_kv_table(["Property", "Value"], [
            ("*meltiro* version", _code_cell(meta.get("meltiro_version"))),
            ("`engine_fp`", _code_cell(meta.get("engine_fp"))),
            ("git commit", _code_cell(meta.get("git_commit"))),
            ("git working tree", tree),
        ]))

        self._render_models()
        self._render_fingerprints()
        self._render_structure()
        self._render_spend()
        self._render_warnings()
        self._render_capabilities()
        self._render_receipts()

    def _render_models(self):
        meta = self.s.meta
        decoding = meta.get("decoding_params") or {}
        rows = []
        for role, label, configured_key, resolved_key, decoding_key in (
            ("extractor", "Extractor", "extractor_model",
             "extractor_model_resolved", "extractor"),
            ("checker", "Checker", "checker_model",
             "checker_model_resolved", "checker"),
            ("review", "Reviewer", "review_model",
             "review_model_resolved", "review"),
        ):
            configured = meta.get(configured_key)
            if configured is None:
                rows.append([label, "*(stage off)*", "*(stage off)*",
                             "*(stage off)*"])
                continue
            params = decoding.get(decoding_key)
            rows.append([
                label,
                _code_cell(configured),
                _code_cell(meta.get(resolved_key)) if meta.get(resolved_key)
                else "*(not recorded)*",
                _code_cell(json.dumps(params, ensure_ascii=False,
                                      sort_keys=True))
                if params else "*(not recorded)*",
            ])
        self._p("### Models")
        self._p()
        self._p(
            "The configured id is the alias the config named. The reported id "
            "is what the provider said it served, read off the first call "
            "that role made. The decoding parameters are the exact dict the "
            "adapter sent, after any provider quirk was applied; what the "
            "config asked for is recorded beside it in `run.json` as "
            "`decoding_specified`, because a model that refuses a control is "
            "sent none of it and the wire alone cannot say whether a value "
            "was written at all."
        )
        self._p()
        self._block(_table(
            ["Role", "Configured", "Reported by the provider", "Decoding"],
            rows))
        omitted = meta.get("images_omitted") or {}
        if omitted:
            roles = ", ".join(f"`{r}`" for r in sorted(omitted))
            self._p(
                f"The paper's figures were withheld from {roles}: that role's "
                "model is text-only, so it was sent no image parts and its "
                "image-label list rendered as the none-available state."
            )
            self._p()
        transport = meta.get("transport")
        served = meta.get("served_providers") or []
        if transport == "direct":
            self._p(
                "Every call went straight to its provider. A direct call has "
                "no gateway routing receipt, so there is none to record."
            )
            self._p()
        elif transport:
            self._block(_kv_table(["Property", "Value"], [
                ("Transport", _code_cell(transport)),
                ("Upstream providers that served the routed calls",
                 ", ".join(f"`{s}`" for s in served)
                 if served else "*(none recorded)*"),
                ("Routing receipts",
                 str(len(meta.get("generation_ids") or []))),
            ]))

    def _render_fingerprints(self):
        meta = self.s.meta
        self._p("### Fingerprints")
        self._p()
        self._p(
            "At one *meltiro* version, the same stage fingerprint plus the "
            "same input means the model was asked the same question. A "
            "disabled stage records no fingerprint."
        )
        self._p()
        self._block(_kv_table(["Fingerprint", "Value"], [
            ("`run_fp`", _code_cell(meta.get("run_fp"))),
            ("`config_fp` (extractor)", _code_cell(meta.get("config_fp"))),
            ("`checker_fp`", _code_cell(meta.get("checker_fp"))
             if meta.get("checker_fp") else "*(stage off)*"),
            ("`review_fp`", _code_cell(meta.get("review_fp"))
             if meta.get("review_fp") else "*(stage off)*"),
            ("`template_hash`", _code_cell(meta.get("template_hash"))),
            ("`prompt_hash`", _code_cell(meta.get("prompt_hash"))),
            ("`tool_set_hash`", _code_cell(meta.get("tool_set_hash"))),
        ]))

    def _render_structure(self):
        meta = self.s.meta
        structure = meta.get("structure") or {}
        caps = meta.get("caps") or {}
        self._p("### Pipeline shape and budgets")
        self._p()
        self._block(_kv_table(["Setting", "Value"], [
            ("Checker", "on" if structure.get("checker") else "off"),
            ("Checks allowed per field",
             _present(structure.get("max_checks_per_field"))),
            ("Paper context per quote (characters each side)",
             _present(meta.get("checker_context_chars"))
             if structure.get("checker") else "*(stage off)*"),
            ("Reviewer", "on" if structure.get("review") else "off"),
            ("Reviewer's own writes checked",
             "yes" if structure.get("check_reviewer_edits") else "no"),
            ("Extractor tool-call cap", _present(caps.get("max_tool_calls"))),
            ("Reviewer tool-call cap",
             _present(caps.get("max_review_tool_calls"))),
            ("Extractor tool calls dispatched",
             _present(meta.get("tool_call_count"))),
            # CHECKS, not calls: a check whose first reply recorded no verdict
            # is re-asked, and that check made two provider calls. The wire log
            # is where the calls are counted.
            ("Checks run", _present(meta.get("checker_calls_run"))),
        ]))
        self._p(
            "The caps are the budget the CURRENT segment honoured. They ride "
            "in no fingerprint, so a resume may raise them; the `resumed` "
            "notes in section 3 carry each segment's own values."
        )
        self._p()

    def _render_spend(self):
        meta = self.s.meta
        self._p("### Spend")
        self._p()
        self._block(_kv_table(["Meter", "Total"], [
            ("Cost", _run_cost_cell(meta)),
            ("Input tokens (charged in full)",
             _present(meta.get("input_tokens"))),
            ("Output tokens", _present(meta.get("output_tokens"))),
            ("Prompt-cache writes",
             _present(meta.get("cache_creation_tokens"))),
            ("Prompt-cache reads", _present(meta.get("cache_read_tokens"))),
        ]))
        self._p(
            "Totals across every call the session made: the extractor's "
            "turns, every per-field checker call, and the reviewer's turns."
        )
        self._p()
        self._render_spend_by_role()
        self._render_rate_card()

    def _render_spend_by_role(self):
        """The same meters again, split by the role that ran them up, so a
        reader can check any one figure: a role's counters and that role's
        card (in the table below) multiply back to the role's cost here."""
        by_role = self.s.meta.get("usage_by_role")
        if not isinstance(by_role, dict) or not by_role:
            return
        self._block(_table(
            ["Role", "Cost", "Input", "Output", "Cache writes", "Cache reads"],
            [[role,
              _role_cost_cell(block),
              _present(block.get("input_tokens")),
              _present(block.get("output_tokens")),
              _present(block.get("cache_write_tokens")),
              _present(block.get("cache_read_tokens"))]
             for role, block in by_role.items()]))
        self._p(
            "A role states a cost or states none. `*(not priced)*` against a "
            "role that made calls means nothing said what its model charges: "
            "its tokens are the whole of its record, and the run's total is "
            "withheld with it, because a sum over the priced roles alone would "
            "read as the whole run."
        )
        self._p()

    def _render_rate_card(self):
        """The rates each role was priced at, and where those rates came from.

        Printing each role's card beside its own counters keeps the
        arithmetic reproducible from this document alone, however far the
        provider's prices have since moved (see rates.py). The last two
        columns say which source supplied the card and, for the price table,
        the day the vendor's page behind it was read.
        """
        meta = self.s.meta
        cards = meta.get("cost_rates")
        cards = cards if isinstance(cards, dict) else {}
        usage = meta.get("usage_by_role")
        usage = usage if isinstance(usage, dict) else {}
        if not cards:
            self._p(
                "This run records no rates, so it states what it spent in "
                "tokens and no dollar figure at all. Unit prices come from the "
                "operator (`rates:` in `pipeline.yaml`) or from *direktoro*'s "
                "price table; a cost computed against prices nobody wrote down "
                "could not be checked afterwards, and a zero would read as a "
                "free run."
            )
            self._p()
            return
        rows = []
        routed = False
        unpriced = False
        for role, card in cards.items():
            model = _code_cell(meta.get(_ROLE_MODEL_KEY.get(role)))
            meters = usage.get(role) or {}
            if card:
                rows.append([
                    role, model,
                    _present(card.get("input_per_1m")),
                    _present(card.get("output_per_1m")),
                    _present(card.get("cache_read_per_1m")),
                    _present(card.get("cache_write_per_1m")),
                    _present(card.get("source")),
                    _present(card.get("as_of"), absent="*(n/a)*"),
                ])
            elif meters.get("cost_usd") is not None:
                routed = True
                rows.append([role, model] + ["*(per call)*"] * 4
                            + ["gateway charge", "*(n/a)*"])
            elif any(meters.get(counter) for counter in _TOKEN_COUNTERS):
                unpriced = True
                rows.append([role, model] + ["*(not priced)*"] * 4
                            + ["*(none)*", "*(n/a)*"])
            else:
                # No card, no charge, and no tokens: this role never got as far
                # as a call, so nothing is known about what it would have cost.
                rows.append([role, model] + ["*(no calls)*"] * 4
                            + ["*(none)*", "*(n/a)*"])
        self._block(_table(
            ["Role", "Model", "Input", "Output", "Cache read", "Cache write",
             "Rates from", "Read on"], rows))
        self._p(
            "Rates are USD per million tokens. `operator` is a card written "
            "under `rates:` in `pipeline.yaml`; `table` is *direktoro*'s dated "
            "price table, and **Read on** is the day the vendor's page behind "
            "that entry was read."
        )
        if routed:
            self._p()
            self._p(
                "A role priced from the `gateway charge` runs a routed model: "
                "each call costs what the gateway reported charging for it, "
                "which is a fact about what was billed rather than a figure "
                "anybody computed, so it needs no rate card behind it."
            )
        if unpriced:
            self._p()
            self._p(
                "A role marked `*(not priced)*` had neither an operator card "
                "nor a price-table entry for its model. It records its token "
                "counters and no dollar figure, and the run states no total."
            )
        self._p()

    def _render_warnings(self):
        warnings = self.s.meta.get("warnings") or []
        self._p("### Warnings the run recorded")
        self._p()
        if not warnings:
            self._p("None. The run recorded no non-fatal degradation.")
            self._p()
            return
        for warning in warnings:
            self._p("- " + _indent_continuation(warning))
        self._p()

    def _capability_notes(self):
        """What this document cannot show at the level this run recorded.

        Keyed off the RECORDED level, not off which files happen to be on
        disk, so the answer is "the run chose not to keep this" rather than a
        guess. A file the level promised but that is absent is reported
        separately, where it would have been rendered.
        """
        level = self.s.level
        notes = []
        if not captures_instrument(level):
            notes.append(
                "`minimal` keeps no `instrument/`, so section 2 has no "
                "captured prompts, tool definitions, or figure labels to "
                "render. They were never written down. The fingerprints above "
                "still identify the instrument, they just cannot reproduce "
                "it, and re-rendering it from the config bundle now would be "
                "a reconstruction rather than a record."
            )
        if not captures_api_calls(level):
            notes.append(
                "Only `full` keeps `api_calls.jsonl`, the verbatim wire log. "
                "Each checker call's rendered user message lives nowhere "
                "else, so the checks in this document show what the checker "
                "scored and what it decided, but not the exact message it was "
                "sent."
            )
        return notes

    def _render_capabilities(self):
        self._p("### What this document can show")
        self._p()
        self._p(
            f"This run kept its diagnostics at `{self.s.level}`. The level is "
            "operational: it chooses which records of the run survive on disk "
            "and changes nothing about what any model was asked, so it rides "
            "in none of the fingerprints above."
        )
        self._p()
        notes = self._capability_notes()
        if not notes:
            self._p(
                "`full` is the top level, so every record a run can keep is "
                "here and nothing below is missing for want of one."
            )
            self._p()
        else:
            for note in notes:
                self._p("- " + note)
            self._p()
        self._p(
            "Within what the level kept, this document truncates nothing. "
            "Prompts, paper text, tool arguments, tool results, checker "
            "messages, and the extraction output are all printed in full. The "
            "one thing not rendered is the request and response bodies in "
            "`api_calls.jsonl`, which repeat the instrument and the turns "
            "already shown here; read that file directly for the exact bytes."
        )
        self._p()
        if self.s.field_history_derived:
            self._p(
                "`diagnostics/field_history.json` is absent, which means this "
                "session was killed mid-run rather than stopped: the file is "
                "written at every stop. Section 6 is derived here from the "
                "event log by the same function that writes it, so it "
                "describes the log as it stands."
            )
            self._p()

    def _render_receipts(self):
        ids = self.s.meta.get("generation_ids") or []
        if not ids:
            return
        self._p("### Routing receipts")
        self._p()
        self._p(
            "One gateway generation id per routed call, in call order. A "
            "directly served call has no equivalent receipt."
        )
        self._p()
        self._block(_fence("\n".join(str(i) for i in ids), "text"))

    # -- 2. the instrument ------------------------------------------------

    def _render_instrument(self):
        self._p(_anchor("sec-instrument"))
        self._p("## 2. The instrument")
        self._p()
        self._p(
            "Everything the config bundle put in front of a model, captured "
            "as it was sent, at session creation. Each piece is printed once, "
            "here, and referred back to from the turns below. The checker's "
            "system prompt in particular is one string for the whole run: it "
            "is printed here and never repeated, however many checks the run "
            "made."
        )
        self._p()
        self._p(
            "The tool definitions are the exception: 2.5 indexes them and "
            "their full text is [section 7](#sec-tool-definitions), at the "
            "end. They carry the whole field schema, and printing them here "
            "would put most of the document between a reader and the run."
        )
        self._p()
        if not captures_instrument(self.s.level):
            self._p(
                f"**This session kept its diagnostics at `{self.s.level}`, "
                "which captures no instrument.** There is nothing to print "
                "here. The prompts, the tool definitions, and the figure "
                "labels the models saw were never written to disk, and the "
                "instrument is captured once, at session creation, so a "
                "session started at `minimal` never gains one. Re-rendering "
                "them from the config bundle now would be a reconstruction "
                "dressed as a capture, and would be valid only if nothing in "
                "the bundle had changed since; *meltiro* does not do it. What "
                "survives is the identity: `config_fp`, `checker_fp`, "
                "`review_fp`, `prompt_hash`, and `tool_set_hash` in section 1 "
                "still say which instrument this was."
            )
            self._p()
            return

        self._render_prompt(
            "instrument-extractor-system",
            "2.1 The extractor's system prompt",
            self.s.instrument["extractor_system"],
            "The rendered system message the extractor was given, with "
            "`{include:...}` partials expanded and `{reference:...}` lists "
            "substituted.",
        )
        self._render_prompt(
            "instrument-extractor-user",
            "2.2 The extractor's first user message",
            self.s.instrument["extractor_user"],
            "The text portion of the opening user message: the paper text and "
            "the image-label notice. Any figures went alongside it as image "
            "parts, and are listed in 2.6.",
        )
        self._render_prompt(
            "instrument-review-system",
            "2.3 The reviewer's system prompt",
            self.s.instrument["review_system"],
            "The reviewer runs on a fresh context and is shown the assembled "
            "extraction output rather than the extractor's conversation. This "
            "is the system message it was given.",
            off_note=(
                "The reviewer stage was off for this run, so it has no "
                "system prompt."
                if not (self.s.meta.get("structure") or {}).get("review")
                else None),
        )
        self._render_prompt(
            "instrument-checker-system",
            "2.4 The checker's system prompt",
            self.s.instrument["checker_system"],
            "One string for the whole run, cached across every per-field "
            "check. Each check in sections 3 and 4 links back here rather "
            "than reprinting it.",
            off_note=(
                "The checker stage was off for this run "
                "(`max_checks_per_field` is 0), so it has no system prompt "
                "and no check appears below."
                if not (self.s.meta.get("structure") or {}).get("checker")
                else None),
        )
        self._render_tool_index()
        self._render_figures()

    def _render_prompt(self, anchor, heading, text, blurb, off_note=None):
        self._p(_anchor(anchor))
        self._p("### " + heading)
        self._p()
        if off_note is not None:
            self._p(off_note)
            self._p()
            return
        self._p(blurb)
        self._p()
        if text is None:
            self._p(
                "*This file is absent from `diagnostics/instrument/`. The "
                "session's level does capture the instrument, so it was not "
                "supplied at session creation rather than dropped by the "
                "level.*"
            )
            self._p()
            return
        self._block(_fence(text, "text"))

    def _render_tool_index(self):
        """The tool catalogue as an index, one row per tool.

        The full definitions carry the whole field catalogue and would bury
        the run, so they live in section 7 under the same per-tool anchors;
        nothing is dropped, only moved.
        """
        tools = self.s.instrument["tools"]
        self._p(_anchor("instrument-tools"))
        self._p("### 2.5 The tool catalogue")
        self._p()
        self._p(
            "The extractor's tools as passed to the API, in the order they "
            "were sent; their descriptions are engine and config text the "
            "model reads. The reviewer's catalogue is this one minus "
            "`record_initial_check`, which reports on the inputs before "
            "extraction begins and so has no honest moment in the review "
            "stage. `mark_complete` is described to each role in its own "
            "terms; the rest are identical."
        )
        self._p()
        if not self._tool_list():
            self._render_missing_tools()
            return
        rows = [[f"[`{t.get('name')}`](#tool-{_slug(t.get('name'))})",
                 _cell(_lede(t.get("description", "")))]
                for t in self._tool_list()]
        self._block(_table(["Tool", "What the model is told it does"], rows))
        self._p(
            "Each row opens with as much of the tool's description as makes a "
            "sentence. The full description and the complete input schema for "
            "every tool are printed verbatim in "
            "[section 7](#sec-tool-definitions), at the end of the document. "
            "Nothing is left out of this document, only moved out of the way "
            "of the run."
        )
        self._p()

    def _tool_list(self):
        tools = self.s.instrument["tools"]
        if not isinstance(tools, list):
            return []
        return [t for t in tools if isinstance(t, dict)]

    def _render_missing_tools(self):
        tools = self.s.instrument["tools"]
        if tools is None:
            self._p(
                "*`instrument/tool_definitions.json` is absent from this "
                "session.*")
        elif not isinstance(tools, list):
            self._p(
                "*`instrument/tool_definitions.json` is not a JSON array.*")
        else:
            self._p("*`instrument/tool_definitions.json` is empty.*")
        self._p()

    def _render_tool_definitions(self):
        """Section 7: every tool's description and input schema, in full.

        Deliberately last: reference material a reader jumps to from a call
        in section 3 or from the index in 2.5 (see `_render_tool_index`).
        """
        self._p(_anchor("sec-tool-definitions"))
        self._p("## 7. The tool definitions in full")
        self._p()
        if not captures_instrument(self.s.level):
            self._p(
                f"This session kept its diagnostics at `{self.s.level}`, "
                "which captures no instrument, so there are no tool "
                "definitions to print. [Section 2](#sec-instrument) says what "
                "follows from that."
            )
            self._p()
            return
        self._p(
            "Reference material, printed last because it is what a reader "
            "jumps to rather than reads through: every tool's description and "
            "its complete input schema, exactly as passed to the API and in "
            "the order they were sent. [Section 2.5](#instrument-tools) is "
            "the index."
        )
        self._p()
        if not self._tool_list():
            self._render_missing_tools()
            return
        for tool in self._tool_list():
            name = tool.get("name", "(unnamed)")
            self._p(_anchor(f"tool-{_slug(name)}"))
            self._p(f"### `{name}`")
            self._p()
            self._block(_fence(tool.get("description", ""), "text"))
            self._p("Input schema:")
            self._p()
            self._block(_json_fence(tool.get("input_schema")))

    def _render_figures(self):
        labels = self.s.instrument["image_labels"]
        hashes = self.s.meta.get("image_hashes") or {}
        self._p(_anchor("instrument-figures"))
        self._p("### 2.6 The figures")
        self._p()
        if labels is None:
            self._p("*`instrument/image_labels.json` is absent from this "
                    "session.*")
            self._p()
            return
        if not labels:
            self._p(
                "No figures were attached to the extractor's prompt, either "
                "because the paper bundle has none or because the extractor's "
                "model is text-only (section 1 says which)."
            )
            self._p()
        else:
            self._p(
                "The cropped figures attached to the extractor's prompt as "
                "image parts. A field whose evidence cites one of these "
                "labels had the image itself sent to the checker as its "
                "evidence."
            )
            self._p()
            rows = [[f"`{label}`",
                     _code_cell((hashes.get(label) or {}).get("sha256")),
                     _present((hashes.get(label) or {}).get("byte_length"))]
                    for label in labels]
            self._block(_table(
                ["Label", "SHA-256 of the cropped PNG", "Bytes"], rows))
        extra = sorted(set(hashes) - set(labels or []))
        if extra:
            self._p(
                "The paper bundle also carries these figures, hashed at "
                "session start but not attached to the extractor's prompt: "
                + ", ".join(f"`{label}`" for label in extra) + "."
            )
            self._p()

    # -- 3 and 4. the turns -----------------------------------------------

    def _render_stages(self):
        extraction, review = _split_stages(self.s.events)
        self._p(_anchor("sec-extraction"))
        self._p("## 3. The extraction, turn by turn")
        self._p()
        self._p(
            "Each turn is one call to the extractor's model and everything "
            "that came back from it: the prose it wrote, the tools it called, "
            "and the result each call returned. A result shows what the "
            "dispatcher applied, what it rejected and why, and the checker's "
            "verdict on each field it judged, next to that field."
        )
        self._p()
        if (self.s.meta.get("structure") or {}).get("checker"):
            self._p(
                "Every check below was made by a separate call of its own "
                "under [the checker's system prompt]"
                "(#instrument-checker-system), which is printed once in "
                "section 2 and never repeated here. What changes per check is "
                "the user message and the verdict, and those are shown in "
                "full every time."
            )
            self._p()
        self._p(
            "The conversation is rebuilt from the event log. Each turn's "
            "verbatim assistant content is in `tool_calls.jsonl` under "
            "`assistant_message`; what is shown here is that turn's prose and "
            "its tool calls, which is the same content in a readable order."
        )
        self._p()
        if not extraction:
            self._p("*The event log holds no extractor turns.*")
            self._p()
        else:
            self._render_events(extraction, stage="extractor")

        self._p(_anchor("sec-review"))
        self._p("## 4. The review")
        self._p()
        if not (self.s.meta.get("structure") or {}).get("review"):
            self._p(
                "The reviewer stage was off for this run, so there is nothing "
                "in this section. The extraction above is what shipped."
            )
            self._p()
            return
        self._p(
            "**A separate stage on a fresh context.** The reviewer is not "
            "continuing the conversation above: it was given its own system "
            "prompt ([section 2.3](#instrument-review-system)), the paper, "
            "and the assembled extraction output as it stood at the end of "
            "section 3. It never saw the extractor's turns, its tool results, "
            "or any checker verdict raised against them. Its turn numbers "
            "continue the run's single turn counter, which is why they carry "
            "on from the extractor's rather than restarting."
        )
        self._p()
        if not review:
            self._p(
                "*The reviewer stage was enabled but the event log holds no "
                "review turns: the run stopped before the review began.*")
            self._p()
            return
        self._render_events(review, stage="review")

    def _render_events(self, events, *, stage):
        for group in _group_turns(events):
            if group["turn_id"] is None:
                for event in group["events"]:
                    self._render_run_event(event)
                continue
            self._render_turn(group, stage=stage)

    def _render_turn(self, group, *, stage):
        turn_id = group["turn_id"]
        events = group["events"]
        self._p(_anchor(f"turn-{turn_id}"))
        label = "Reviewer turn" if stage == "review" else "Turn"
        self._p(f"### {label} {turn_id}")
        self._p()

        text = _turn_text(events)
        if text:
            self._p("The model wrote:")
            self._p()
            self._block(_quote(text))
        else:
            self._p(
                "*The model wrote no prose in this turn; it answered with "
                "tool calls only.*")
            self._p()

        stop_note = _describe_stop_reason(_turn_stop_reason(events))
        if stop_note:
            self._p("*How the turn ended: " + stop_note + "*")
            self._p()

        for event in events:
            name = event.get("event")
            if name in _TOOL_CALL_EVENTS:
                self._render_tool_call(event, turn_id=turn_id)
            elif name in _TURN_CONTENT_EVENTS or name == "value_canonicalised":
                # A `value_canonicalised` event restates what its tool call's
                # result already carries under `_canonicalisations`, and the
                # call renders it there, next to the field it changed.
                continue
            else:
                self._render_run_event(event)

    def _render_tool_call(self, event, *, turn_id):
        self._call_n += 1
        n = self._call_n
        anchor = f"call-{n}"
        result = event.get("result") or {}
        status = result.get("status", "(no status)")
        tool = event.get("tool", "(unnamed tool)")
        self._p(_anchor(anchor))
        self._p(f"#### Call {n}. `{tool}`")
        self._p()
        if "view" in result:
            # A read-only tool wrote nothing, so neither the applied/partial/
            # failed vocabulary nor the per-field gloss describes it.
            self._p(f"Dispatched in turn {turn_id}. Status `{status}`. A "
                    "read-only call: it wrote nothing and was answered with a "
                    "view of the extraction as it then stood.")
        else:
            # Keyed off the STATUS, not off the event name: the extractor
            # names its events per status (`tool_call_applied` /
            # `_partial` / `_failed`) but the reviewer logs every dispatch as
            # one `review_tool_call`, so the status is the signal both stages
            # share.
            outcome = {
                "ok": "applied",
                "partial": "applied in part",
            }.get(status, "rejected")
            line = (f"Dispatched in turn {turn_id}. Status `{status}`, "
                    f"{outcome}.")
            gloss = _STATUS_GLOSS.get(status)
            if gloss:
                line += f" ({gloss.capitalize()}.)"
            self._p(line)
        self._p()

        self._p("What the model asked for:")
        self._p()
        self._block(_json_fence(event.get("args")))

        self._render_applied(result, anchor)
        self._render_rejected(result, anchor)
        self._render_call_level_errors(result)
        self._render_warnings_block(result)
        self._render_canonicalisations(result)
        self._render_weak_quote_matches(result)
        self._render_view(result)
        self._render_applied_changes(result)
        self._render_summary_line(result)
        self._render_verdicts(result)

    def _render_applied(self, result, call_anchor):
        diffs = result.get("_field_diffs") or {}
        if not diffs:
            return
        self._p(f"**Applied: {_plural(len(diffs), 'field')}.** Each row is "
                "the value either side of this call.")
        self._p()
        rows = []
        for path, diff in diffs.items():
            before = (diff or {}).get("before")
            after = (diff or {}).get("after")
            changed = "no" if before == after else "yes"
            rows.append([f"`{path}`", _value_cell(before), _value_cell(after),
                         changed])
            self._note(path, f"call {self._call_n} applied", call_anchor)
        self._block(_table(["Field", "Before", "After", "Changed"], rows))

    def _render_rejected(self, result, call_anchor):
        failed = result.get("failed_fields") or {}
        if not failed:
            return
        self._p(f"**Rejected: {_plural(len(failed), 'field')}.** Validation "
                "refused what follows, and the model was told so in this "
                "tool result. The proposed value is not recorded: what the "
                "dispatcher keeps is its reason for refusing it.")
        self._p()
        for path, errors in failed.items():
            self._p(f"- `{path}`")
            for error in errors or []:
                code = (error or {}).get("code", "(no code)")
                message = (error or {}).get("message", "")
                self._p("  - `" + str(code) + "`: "
                        + _indent_continuation(message, "    "))
            self._note(path, f"call {self._call_n} rejected", call_anchor)
        self._p()

    def _render_call_level_errors(self, result):
        """Errors that name no field: a malformed argument, an unknown record
        id. They belong to the call, so they are reported apart from the
        per-field rejections above rather than mixed into them."""
        # The dispatcher's `errors` list flattens every per-field error into
        # itself alongside the call-level ones, so what is left after removing
        # one copy of each per-field error is exactly the call-level set. A
        # multiset difference (rather than a set one) keeps a genuine
        # call-level error that happens to read identically to a field's.
        loose = list(result.get("errors") or [])
        for errors in (result.get("failed_fields") or {}).values():
            for error in errors or []:
                if error in loose:
                    loose.remove(error)
        if not loose:
            return
        self._p("**Call-level errors.** These name no single field, so they "
                "belong to the call rather than to anything in it.")
        self._p()
        for error in loose:
            if isinstance(error, dict):
                code = error.get("code", "(no code)")
                path = error.get("path")
                where = f" at `{path}`" if path else ""
                self._p(f"- `{code}`{where}: "
                        + _indent_continuation(error.get("message", ""), "  "))
            else:
                self._p("- " + _indent_continuation(str(error), "  "))
        self._p()

    def _render_warnings_block(self, result):
        warnings = result.get("warnings") or []
        by_field = result.get("warnings_by_field") or {}
        if not warnings and not by_field:
            return
        self._p("**Warnings returned to the model.** These did not stop "
                "anything applying.")
        self._p()
        for warning in warnings:
            if isinstance(warning, dict):
                self._p("- " + _indent_continuation(
                    json.dumps(warning, ensure_ascii=False), "  "))
            else:
                self._p("- " + _indent_continuation(str(warning), "  "))
        for path, items in by_field.items():
            for item in items or []:
                text = item.get("message") if isinstance(item, dict) \
                    else str(item)
                self._p(f"- `{path}`: " + _indent_continuation(text, "  "))
        self._p()

    def _render_canonicalisations(self, result):
        records = result.get("_canonicalisations") or []
        if not records:
            return
        self._p("**Canonicalised.** The value the model entered matched a "
                "reference list alias rather than a canonical name, so the "
                "canonical name was stored and the model was told.")
        self._p()
        for record in records:
            self._p(
                f"- `{record.get('path')}`: entered "
                f"{_value_cell(record.get('entered'))}, stored "
                f"{_value_cell(record.get('stored'))}")
        self._p()

    def _render_weak_quote_matches(self, result):
        """Name the evidence quotes that passed only once case was folded.

        A case-folded match passes and is not an error, so it appears here
        rather than under `Rejected` — but it is the one tier where the
        difference is the model's own rather than a PDF converter's, and the
        tier is recorded nowhere else in the document.
        """
        records = result.get("_weak_quote_matches") or []
        if not records:
            return
        self._p("**Matched on case only.** The quote is not in the paper as "
                "written: it matched once both sides were lowercased. The "
                "check accepts that, and the value was stored, but the "
                "evidence rests on a weaker match than a quote reproduced "
                "exactly. Nothing was said to the model.")
        self._p()
        for record in records:
            self._p(
                f"- `{record.get('path')}`: "
                f"{_value_cell(record.get('quote'))} "
                f"(tier `{record.get('tier')}`)")
        self._p()

    def _render_view(self, result):
        view = result.get("view")
        if view is None:
            return
        self._p("The view it was answered with:")
        self._p()
        self._block(_json_fence(view))

    def _render_applied_changes(self, result):
        changes = result.get("applied_changes")
        if not changes:
            return
        self._p("The dispatcher's own report of what it wrote:")
        self._p()
        self._block(_json_fence(changes))

    def _render_summary_line(self, result):
        summary = result.get("extraction_output_summary")
        if not isinstance(summary, dict):
            return
        parts = ", ".join(f"{k} {v}" for k, v in summary.items())
        self._p(f"State of the extraction after this call, as the model was "
                f"told it: {parts}.")
        self._p()

    def _render_verdicts(self, result):
        verdicts = result.get("_checker_verdicts") or {}
        if not verdicts:
            return
        self._p(f"**The checker looked at "
                f"{_plural(len(verdicts), 'field')} written by this call.**")
        self._p()
        # What this call actually put to the model, read off the event rather
        # than assumed from the verdict: whether a failed check's text reached
        # the extractor is a fact about the run being rendered, and a document
        # that asserted the engine's current rule would misdescribe any
        # session recorded under a different one.
        challenged = set(result.get("checker_challenges") or {})
        for path, verdict in verdicts.items():
            self._render_verdict(path, verdict or {},
                                 shown_to_extractor=path in challenged)

    def _render_verdict(self, path, verdict, *, shown_to_extractor=False):
        self._check_n += 1
        n = self._check_n
        anchor = f"check-{n}"
        kind = verdict.get("verdict", "(no verdict)")
        error_origin = bool(verdict.get("error_origin"))
        if error_origin:
            # No cause in the headline: a check ends without a verdict from
            # exhausted retries, a reply that called no tool, a cut-off reply,
            # a verdict outside the vocabulary, or a fault in the plumbing.
            # Naming one of them here would be a claim about this check that
            # the rationale below is the actual record of.
            headline = ("challenge, but from a failed check — not a checker "
                        "judgement")
            label = "check error"
        elif kind == "challenge":
            headline = "challenge"
            label = "challenge"
        elif kind == "ok":
            headline = "ok"
            label = "ok"
        else:
            headline = str(kind)
            label = str(kind)
        ordinal = self._check_ordinal.get(path, 0) + 1
        self._check_ordinal[path] = ordinal
        reprompted = verdict.get("reprompted")
        reprompted = int(reprompted) if isinstance(reprompted, int) else 0
        self._p(_anchor(anchor))
        suffix = "" if ordinal == 1 else f" (check {ordinal} of this field)"
        if reprompted:
            times = "once" if reprompted == 1 else _plural(reprompted, "time")
            suffix += f" (re-asked {times})"
        self._p(f"##### Check {n}. `{path}`: {headline}{suffix}")
        self._p()
        self._note(path, f"check {n} {label}", anchor)

        if error_origin:
            reached = ("its text WAS put to the extractor in this call's "
                       "tool result" if shown_to_extractor else
                       "its text was not put to the extractor")
            self._p(
                "This is not a judgement. The check failed, and the failure "
                "was degraded to a challenge so that the other fields in its "
                "batch would not fail with it. It is an absence of "
                "information rather than an objection to the value: it spent "
                f"one of the field's check slots, {reached}, and it cost "
                # The same cell the table below prints, so a failed check's
                # figure states its coverage in prose too: this is the check
                # most likely to have lost a receipt, its call having gone
                # wrong.
                f"{self._verdict_cost_cell(verdict)} — the calls behind a "
                "failed check may well have completed and been billed before "
                "their answers turned out to be unusable. What was sent is "
                "printed below whenever the session kept a wire log of it."
            )
            self._p()

        messages = self._checker_messages.get((path, ordinal)) or []
        if len(messages) > 1:
            # Every ask was sent and billed. The first came back without a
            # verdict, which is the whole reason there is a second, and the
            # second corrected it: the same field message, the reply where the
            # checker could replay one, then the correction. Each ask is
            # printed whole, in the order it carried, so the field appears in
            # both and the correction only in the last.
            self._p(
                f"This check took {_plural(len(messages), 'ask')}, each sent "
                "and billed. All of them in full:")
            self._p()
            for i, segments in enumerate(messages, start=1):
                which = ("the first ask" if i == 1 else
                         "re-asked, correcting the reply that recorded no "
                         "verdict")
                self._p(f"Ask {i} of {len(messages)}, {which}:")
                self._p()
                self._ask(segments)
        elif messages:
            self._p("The message this check was sent, in full:")
            self._p()
            self._ask(messages[0])
        elif not captures_api_calls(self.s.level):
            self._p(
                "*The rendered user message for this check is not in the "
                "session: it is written only to `api_calls.jsonl`, which "
                f"`{self.s.level}` does not keep. What the checker was "
                "looking at is below.*"
            )
            self._p()
        elif not error_origin:
            self._p(
                "*No wire-log entry matches this check, so its rendered user "
                "message could not be recovered.*")
            self._p()

        rows = [
            ["System prompt",
             "[printed once in section 2.4](#instrument-checker-system)"],
            ["Value it scored", _value_cell(verdict.get("value_checked"))],
            ["Evidence it scored",
             _value_cell(verdict.get("evidence_checked"))],
            ["Extractor's note, as it saw it",
             _value_cell(verdict.get("note_checked"))],
            ["Verdict", f"`{kind}`"],
            # How many times the checker had to be re-asked before it recorded
            # a verdict at all. It says something about the CHECKER MODEL
            # rather than about this field, and it is beside the outcome
            # because it is also what says how many calls the cost below
            # covers.
            ["Re-asks before a verdict", _present(reprompted)],
            ["Raised by", _code_cell(verdict.get("stage"))],
            ["Cost", self._verdict_cost_cell(verdict)],
            ["Tokens",
             f"{_present(verdict.get('input_tokens'))} in, "
             f"{_present(verdict.get('output_tokens'))} out"],
        ]
        self._block(_table(["Property", "Value"], rows))
        self._p("Its rationale:")
        self._p()
        self._block(_quote(_present(verdict.get("rationale"),
                                    "(no rationale recorded)")))
        if verdict.get("notes"):
            self._p("Its own note on the verdict:")
            self._p()
            self._block(_quote(verdict["notes"]))

    @staticmethod
    def _verdict_cost_cell(verdict):
        """One check's cost, and how much of the check it covers.

        A gateway-served call states its own charge, and a response that came
        back without one leaves a figure that prices the rest. Saying so where
        the number is printed is the difference between a small cost and an
        understated one. One check can make more than one call — a re-ask is a
        second — so the count is of this check's own calls.
        """
        cost = verdict.get("cost_usd")
        if not verdict.get("cost_incomplete"):
            return _money(cost)
        return cost_with_coverage(cost, _money(cost),
                                  verdict.get("unreceipted_responses"))

    def _render_run_event(self, event):
        text = _describe_run_event(event)
        if text is None:
            return
        self._p("*Run note: " + text + "*")
        self._p()

    # -- 5. the output ----------------------------------------------------

    def _render_output(self):
        self._p(_anchor("sec-output"))
        self._p("## 5. The extraction output")
        self._p()
        self._p(
            "`extraction_output.json` as it stands, in full. This is the only "
            "file in the session a model wrote, and it is what a downstream "
            "consumer reads. Everything else in this document is the "
            "deterministic record of how it came to be."
        )
        self._p()
        self._block(_json_fence(self.s.output))

    # -- 6. field history -------------------------------------------------

    def _render_field_history(self):
        history = self.s.field_history or {}
        aggregate = history.get("aggregate") or {}
        fields = history.get("fields") or {}
        self._p(_anchor("sec-field-history"))
        self._p("## 6. What happened to each field")
        self._p()
        self._p(
            "The aggregate below is `diagnostics/field_history.json`'s own, "
            "which is derived from the same event log this document is. The "
            "`fields_*` counts count FIELDS; the rest count events."
        )
        self._p()
        self._block(_table(["Measure", "Value"], [
            ["Fields written", _present(aggregate.get("fields_written"))],
            ["Fields checked", _present(aggregate.get("fields_checked"))],
            ["Fields the reviewer touched",
             _present(aggregate.get("fields_reviewer_touched"))],
            ["Fields still challenged at the end",
             _present(aggregate.get("fields_with_unresolved_challenge"))],
            ["Challenges raised",
             _present(aggregate.get("challenges_raised"))],
            ["Challenges answered by changing the value",
             _present(aggregate.get("challenges_revised"))],
            ["Challenges answered by standing by the value",
             _present(aggregate.get("challenges_overruled"))],
            # Every way a check can end without a verdict — retries exhausted,
            # a reply that recorded none, a cut-off reply, an answer outside
            # the vocabulary — counts here. Naming one of them would read as a
            # claim that the others did not happen.
            ["Checks that ended with no verdict",
             _present(aggregate.get("check_errors"))],
            ["What the checker cost", _checker_cost_cell(aggregate)],
        ]))
        self._p(
            "No arithmetic identity holds between these: one field can be "
            "challenged more than once, and a challenge nobody answered is "
            "neither revised nor overruled."
        )
        self._p()

        self._p("### Every field, and where to find it")
        self._p()
        if not fields:
            self._p("*No field was ever written or proposed in this run.*")
            self._p()
            return
        self._p(
            "One entry per field path, in the order the run first touched "
            "each. The trail links to every call that wrote it and every "
            "check it received, so a field can be followed without leaving "
            "this document."
        )
        self._p()
        for path, entry in fields.items():
            self._render_field_entry(path, entry or {})
        untracked = [p for p in self._trail if p not in fields]
        if untracked:
            self._p(
                "These field paths appear in the turns above but not in the "
                "field history: "
                + ", ".join(f"`{p}`" for p in sorted(untracked)) + "."
            )
            self._p()

    def _render_field_entry(self, path, entry):
        final = entry.get("final") or {}
        self._p(_anchor("field-" + _slug(path)))
        self._p(f"#### `{path}`")
        self._p()
        rows = [
            ["Still in the output",
             "yes" if final.get("present") else "no"],
            ["Final value", _value_cell(final.get("value"))],
            ["Writes that landed", _present(final.get("writes"))],
            ["Proposals rejected", _present(final.get("rejections"))],
            ["Last written by", _code_cell(final.get("last_write_stage"))],
            ["Checks received", _present(final.get("checks"))],
            ["Last verdict", _code_cell(final.get("last_verdict"))],
            ["Still challenged",
             "yes" if final.get("unresolved_challenge") else "no"],
            ["Reviewer touched it",
             "yes" if final.get("reviewer_touched") else "no"],
        ]
        self._block(_table(["Property", "Value"], rows))
        trail = self._trail.get(path) or []
        if trail:
            self._p("Trail: " + ", ".join(
                f"[{label}](#{anchor})" for label, anchor in trail) + ".")
            self._p()
        if final.get("unresolved_challenge"):
            self._p(
                "The checker's last word on this field was an objection and "
                "nothing answered it. That does not change the run's status: "
                "a challenge is advisory. It is flagged so a human who wants "
                "to look at this cell can find it."
            )
            self._p()


# ---------------------------------------------------------------------------
# Event walking
# ---------------------------------------------------------------------------

def _is_review_event(event):
    return (event.get("stage") == "review"
            or event.get("event") in _REVIEW_EVENTS)


def _split_stages(events):
    """Split the log into the extractor's part and the reviewer's.

    The reviewer runs after the extractor, so the split is positional: every
    event from the first review-classified one onward belongs to the review,
    which keeps run-level notes (and the closing `terminate`) in the stage
    they actually happened in.
    """
    for i, event in enumerate(events):
        if _is_review_event(event):
            return events[:i], events[i:]
    return list(events), []


def _group_turns(events):
    """Group a stage's events into contiguous runs sharing a `turn_id`.

    A turn's events are appended sequentially by the loop, so contiguity is
    the run's own grouping rather than an assumption imposed here. Events with
    no turn id (session notes, resumes, the terminate) come through as their
    own groups, in place, so the narrative keeps its order.
    """
    groups = []
    for event in events:
        turn_id = event.get("turn_id")
        if (groups and groups[-1]["turn_id"] == turn_id
                and turn_id is not None):
            groups[-1]["events"].append(event)
            continue
        groups.append({"turn_id": turn_id, "events": [event]})
    return groups


def _turn_text(events):
    """The prose a turn's model wrote.

    Preferring the `assistant_text` event: it is the human-transcript record
    the loop writes for exactly this purpose, and it is the concatenation of
    the turn's text blocks. Falling back to the text blocks of the verbatim
    `assistant_message` covers a turn that logged one but no separate text
    event.
    """
    for event in events:
        if event.get("event") == "assistant_text" and event.get("text"):
            return event["text"]
    for event in events:
        if event.get("event") != "assistant_message":
            continue
        parts = [b.get("text", "") for b in (event.get("content") or [])
                 if isinstance(b, dict) and b.get("type") == "text"]
        joined = "\n".join(p for p in parts if p).strip()
        if joined:
            return joined
    for event in events:
        if event.get("event") == "final_review_response" \
                and event.get("assistant_text"):
            return event["assistant_text"]
    return ""


def _turn_stop_reason(events):
    """How the provider stopped this turn, or None.

    Every model turn logs exactly one `assistant_message`, and it carries the
    reason. A turn the loop ended as a refusal records that reason a second
    time, on its own `extractor_refused` event, and a session whose
    `assistant_message` carries no `stop_reason` field is read from there: the
    gloss below is the only place the document states the MECHANISM behind a
    refusal, so it rests on either record rather than on one of them. With
    neither, this is None and prints nothing — the same as a turn that ended
    in the ordinary way.
    """
    for event in events:
        if event.get("event") == "assistant_message" \
                and event.get("stop_reason") is not None:
            return event["stop_reason"]
    for event in events:
        if event.get("event") == "extractor_refused":
            return event.get("stop_reason")
    return None


# How a turn's stop reason reads in prose, for the reasons that CHANGE how the
# turn should be read: a refusal means the reply above was blocked rather than
# given, and a truncation means it was cut off mid-sentence rather than
# finished. `end_turn` and `tool_use` are the turn doing what a turn does, and
# a note saying so on every turn would bury the two that matter.
_STOP_REASON_GLOSS = {
    "refusal": (
        "the endpoint refused it (`stop_reason` `refusal`) — a host content "
        "filter blocked the reply, or the model declined the request. "
        "Whatever the turn carried above, it was declined"),
    "max_tokens": (
        "it hit the output cap (`stop_reason` `max_tokens`), so the reply "
        "above is cut off rather than finished"),
}


def _describe_stop_reason(stop_reason):
    """The sentence for a turn's stop reason, or None when it needs none."""
    return _STOP_REASON_GLOSS.get(stop_reason)


def _describe_run_event(event):
    """One sentence for an event that is not a turn's content.

    Returns None for events whose content is rendered elsewhere, so nothing is
    said twice.
    """
    name = event.get("event")
    if name in _TURN_CONTENT_EVENTS or name == "value_canonicalised":
        return None
    if name == "session_started":
        return "the session was created."
    if name == "resumed":
        # The decoding sentence appears only on a segment that changed the
        # blocks: the event carries the pair exactly then, and a control the
        # model refuses moves no fingerprint, so this note is the only place
        # such an edit is visible at all.
        decoding = ""
        if "decoding_specified" in event:
            decoding = (
                " The decoding this segment states is not the previous "
                "segment's: "
                + json.dumps(event.get("previous_decoding_specified"),
                             ensure_ascii=False, sort_keys=True)
                + " to "
                + json.dumps(event.get("decoding_specified"),
                             ensure_ascii=False, sort_keys=True) + ".")
        return (
            "the run was resumed. Extractor tool-call cap "
            f"{event.get('previous')} to {event.get('max_tool_calls')}, "
            "reviewer tool-call cap "
            f"{event.get('previous_max_review_tool_calls')} to "
            f"{event.get('max_review_tool_calls')}, diagnostics "
            f"{event.get('previous_diagnostics')} to "
            f"{event.get('diagnostics')}. The code tree was "
            + ("dirty" if event.get("git_dirty") else "clean")
            + " at the start of this segment." + decoding
        )
    if name == "extractor_reprompt":
        return ("the turn called no tool, so *meltiro* sent this back: "
                + json.dumps(event.get("text", ""), ensure_ascii=False))
    if name == "review_reprompt":
        return ("the reviewer called no tool, so *meltiro* sent this back: "
                + json.dumps(event.get("text", ""), ensure_ascii=False))
    if name == "text_only_stall":
        return (
            f"{event.get('consecutive_text_only_turns')} consecutive turns "
            "called no tool, so the run stopped. Raising a cap does not help "
            "a model that never calls a tool, so this is terminal rather than "
            "a pause.")
    if name == "review_text_only_stall":
        return (
            f"the reviewer went {event.get('consecutive_text_only_turns')} "
            "consecutive turns without calling a tool, so the run stopped.")
    if name in ("repeated_failure_stall", "review_repeated_failure_stall"):
        codes = ", ".join(f"`{c}`" for c in (event.get("error_codes") or []))
        who = "the reviewer" if name.startswith("review") else "the extractor"
        return (
            f"{who} submitted `{event.get('tool')}` "
            f"{event.get('consecutive_identical_failures')} times running and "
            f"failed identically each time ({codes}), so the run stopped "
            "rather than keep paying for it. First error: "
            + json.dumps(event.get("error_message") or "", ensure_ascii=False))
    if name == "extractor_paused":
        return (
            f"the run paused ({event.get('pause_reason')}). The session is "
            "still in progress and can be resumed into the same conversation.")
    if name == "extractor_refused":
        # What the refusal above cost the run. The mechanism is the turn's own
        # stop note, immediately above this line — this event carries the
        # `stop_reason` that note falls back to, so the two sit together
        # whichever record the reason is read from — and the remedy is the
        # error note immediately below it.
        return (
            "the run ends here. Nothing about the paper or the template was "
            "judged, so there is no extraction to call valid or invalid: the "
            "record holds the turns above and this refusal, and no re-prompt "
            "followed, since it would carry the request that was refused.")
    if name == "extractor_abandoned":
        return ("the extractor called `abandon_extraction` and gave up: "
                + json.dumps(event.get("reason") or "", ensure_ascii=False))
    if name == "review_abandoned":
        return ("the reviewer called `abandon_extraction` and gave up: "
                + json.dumps(event.get("reason") or "", ensure_ascii=False))
    if name == "review_cap_hit":
        return (
            f"the reviewer's tool-call cap fired after "
            f"{event.get('review_tool_calls')} calls. Unlike the extractor's "
            "cap this terminates the run: the review conversation is "
            "fresh-context and is never replayed, so there is nothing for a "
            "resume to continue.")
    if name == "final_review_no_response":
        return ("a reviewer turn carried neither prose nor a tool call. That "
                "is an infrastructure failure rather than a judgement.")
    if name == "final_review_edits_none_applied":
        return (f"the reviewer attempted {event.get('attempted')} edit(s) and "
                "none of them landed, so the extraction output is unchanged "
                "from the end of section 3.")
    if name == "review_error":
        return ("a reviewer turn raised an unrecoverable provider error: "
                + json.dumps(event.get("message") or "", ensure_ascii=False))
    if name == "error":
        return ("the run raised: "
                + json.dumps(event.get("message") or "", ensure_ascii=False))
    if name == "provider_retry":
        return (
            f"the {event.get('stage')} stage hit a transient provider failure "
            f"and retried (attempt {event.get('attempt')}, after "
            f"{event.get('delay_seconds')}s): "
            + json.dumps(event.get("error") or "", ensure_ascii=False))
    if name == "summary_mismatch_advisory":
        return ("the extracted study summary diverges from the paper "
                "bundle's manifest summary: "
                + json.dumps(event.get("message") or event,
                              ensure_ascii=False))
    if name == "torn_line_dropped":
        return (
            f"a torn final line ({event.get('byte_length')} bytes) was "
            f"dropped from the event log on resume, at line "
            f"{event.get('line_number')}: a hard kill left the last append "
            "incomplete.")
    if name == "terminate":
        return f"the run finished with status `{event.get('status')}`."
    # An event with no sentence above is shown raw rather than dropped: an
    # event the transcript cannot describe is still an event the run recorded,
    # and a new event type reaching a reader as its own JSON is a legible
    # prompt to give it words here.
    return "unrecognised event " + json.dumps(event, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def render_transcript(session_dir):
    """Render one session directory to a Markdown document, returned as a
    string.

    Strict: a missing session directory, a missing or unparseable `run.json`,
    a missing or corrupt event log, and a missing extraction output are each a
    `SessionError` with a message naming the file. There is no partial
    document.
    """
    return _Renderer(_Session(session_dir)).render()


def write_transcript(session_dir, out_path=None):
    """Render `session_dir` and write it, returning the path written.

    Defaults to `{session_dir}/diagnostics/transcript.md`, which is what a run
    writes at every stop. `meltiro transcript --out FILE` passes an explicit
    path so a session can be re-rendered anywhere without disturbing the copy
    in the session.
    """
    session_dir = Path(session_dir)
    document = render_transcript(session_dir)
    if out_path is None:
        out_path = session_dir / "diagnostics" / TRANSCRIPT_FILENAME
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(document, encoding="utf-8")
    return out_path
