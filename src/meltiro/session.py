"""On-disk session state for a single study's agentic extraction.

A session directory (`{run_root}/{study_id}/sessions/{timestamp}_{shortfp}/`)
separates the two things a run produces. The EXTRACTION OUTPUT, the validated
LLM-generated result, sits alone at the top. Everything else is DIAGNOSTICS,
the deterministic record of how the process went, and lives under
`diagnostics/`:

  - `extraction_output.json`: the canonical extraction state, atomically
    rewritten after every applied tool call. The only file at the top level.
  - `diagnostics/run.json`: small mutable summary: status, counters,
    fingerprints, started_at / updated_at, current_phase, and the operational
    settings the run honoured (`caps`, `diagnostics`).
  - `diagnostics/field_history.json`: the per-field history, derived at every
    stop from the event log below and regenerable from it alone (see
    meltiro.field_history).
  - `diagnostics/tool_calls.jsonl`: append-only audit log: every tool call
    (applied or failed) with the checker verdicts it carried, the verbatim
    ordered assistant message per turn (`assistant_message`, the
    byte-identical replay source), the human-transcript assistant-text
    fragment, and every termination event. The `assistant_message` event
    deliberately duplicates each tool_use's name and input from the turn's
    tool_call events: replay needs the exact ordered content, so verbatim
    beats reconstruction.
  - `diagnostics/api_calls.jsonl`: the verbatim wire log, image bytes reduced
    to hashes. Written only at `--diagnostics full`.
  - `diagnostics/instrument/`: the instrument as sent (`system_prompt.txt`,
    `user_prompt.txt`, `review_system_prompt.txt`,
    `checker_system_prompt.txt`, `checker_user_scaffold.txt`,
    `tool_definitions.json`, `image_labels.json`), captured once at session
    creation at `--diagnostics standard` and above.
  - `diagnostics/transcript.md`: the whole session as one readable document,
    rendered from the files above at every stop (see meltiro.transcript).

The three diagnostics levels and what each keeps are in meltiro.diagnostics.
`tool_calls.jsonl` is in every one of them because it is the run's memory:
replay rebuilds the conversation from it, so a session without it cannot be
resumed.

The run root is always supplied by the caller (`runs_dir=`); there is no
CWD-relative default. A Session resolves its directory to an absolute path at
construction, so `session_dir` and the paths derived from it are absolute
whatever the caller passed in (see `Session.__init__`); run_log.json records
them and must stay resolvable without knowing the invocation cwd.

Resume reads `diagnostics/run.json` first (refuses if status != in_progress or
if any supplied stage fingerprint, config_fp / checker_fp / review_fp, does not
match), repairs a torn final jsonl line if a hard kill left one, then
replays `tool_calls.jsonl` to rebuild the conversation byte-identically (the
assistant side from each turn's `assistant_message` event, the tool_result
side from the same serialisation the live loop sent).
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from meltiro import __version__
from meltiro.diagnostics import (
    DEFAULT_DIAGNOSTICS, captures_api_calls, captures_instrument,
    validate_diagnostics)
from meltiro.errors import ResumeRefused, SessionError
from meltiro.extraction_record import ExtractionRecord
from meltiro.field_history import build_field_history
from meltiro.fingerprint import (
    bundle_fingerprint, figure_hashes, run_fingerprint)
from meltiro.run_log import (
    alteksto_version, current_engine_fp, direktoro_version, git_state)
from meltiro.statuses import TERMINAL_STATUSES


# The paper's own axes, as `capture_bundle_fingerprint` records them and
# `Session.resume` re-reads them. The composite `bundle_fp` is deliberately not
# among them: it is a digest OF these five, so comparing it would refuse a
# changed paper without being able to say which part of it changed, and
# comparing both would report the same drift twice.
#
# `tables_fp` and `supplements_fp` are axes for the reason the others are:
# both are shown to the model, so re-transcribing a cell or editing a
# supplement's prose between a pause and a resume changes what the run was
# reading as surely as re-cropping an exhibit does. A supplement is a document
# the run quotes from in its `notes` and cites exhibits out of, so it is on
# this list for the same reason `text_fp` is.
BUNDLE_AXES = ("text_fp", "figures_fp", "manifest_fp", "tables_fp",
               "supplements_fp")


def _engine_label(meltiro_v, direktoro_v):
    """One phrase naming both halves of an engine identity."""
    return (f"meltiro {meltiro_v or '(unrecorded)'} + direktoro "
            f"{direktoro_v or '(absent)'}")


def _drift_axis(meta):
    """Which axis moved, for the message a refused resume carries.

    A stage fingerprint folds in engine-owned material as well as the config
    bundle's — the tool schema, the framing this package writes around the
    bundle's prompts — so upgrading meltiro or direktoro, or editing either
    one's source, moves it with the bundle untouched. Told only that the
    config drifted, an operator goes looking for an edit nobody made.

    The comparison is on `engine_fp`, the run's own engine axis, which is a
    digest of both packages' SOURCE as well as their versions
    (`fingerprint.engine_fingerprint`). Versions alone cannot answer this: an
    edit to this package between two runs of one release changes what a stage
    fingerprint covers and leaves every version string identical, which is the
    ordinary state of a tree under development. The stored fingerprint and the
    one the running code would record are therefore what decide, and the
    versions are printed beside them because they are what a human reads.
    """
    stored = _engine_label(meta.get("meltiro_version"),
                           meta.get("direktoro_version"))
    current = _engine_label(__version__, direktoro_version())
    recorded_fp = meta.get("engine_fp")
    if recorded_fp is None:
        # A session recorded without the key. The axis is undetermined rather
        # than either answer: the versions can still show an engine that
        # moved, but versions that match rule nothing out, and reading an
        # unwritten key as agreement would blame the config for an edit it
        # cannot see.
        if stored != current:
            return (f"This session records no engine_fp, so the axis cannot "
                    f"be settled by content, but the ENGINE did move: it was "
                    f"started by {stored} and is being resumed by {current}. "
                    f"A stage fingerprint folds in engine-owned material, so "
                    f"it drifts on an engine change alone, with the config "
                    f"bundle unedited.")
        return (f"Which axis moved cannot be determined: this session records "
                f"no engine_fp, and the engine is identified here by content "
                f"rather than by version. It records the same versions as the "
                f"engine running now ({current}), which rules nothing out — "
                f"an edit to either package under an unchanged version moves "
                f"a stage fingerprint with the config bundle untouched. Check "
                f"both the engine and the config.")
    current_fp = current_engine_fp()
    if recorded_fp == current_fp:
        return (f"The engine is the one that started this session, by content "
                f"and not merely by version ({current}, {recorded_fp}), so "
                f"what moved is the config — the bundle, or a command-line "
                f"override that feeds the fingerprints (models, "
                f"--max-checks-per-field, --final-review).")
    # What the comparison establishes and no more: the engine moved. Whether
    # the config moved WITH it is a question this comparison does not ask, so
    # the message sends the operator back to the config once the engine is
    # restored rather than telling them the bundle is untouched.
    return (f"The ENGINE moved under this session: it was started by {stored} "
            f"under {recorded_fp} and is being resumed by {current} under "
            f"{current_fp}. A stage fingerprint folds in engine-owned "
            f"material, so an engine change alone is enough to move it. This "
            f"says nothing about the config bundle, which may have been "
            f"edited too: restore the engine, and if the resume is still "
            f"refused, the config is what moved.")


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _utc_now_compact():
    # Microsecond precision so multiple sessions started in the same
    # second still get unique directories.
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _atomic_write_json(path, data):
    """Write `data` to `path` via tmp + rename. Prevents truncated files
    if the process dies mid-write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _atomic_write_text(path, text):
    """Write `text` to `path` via tmp + rename (same durability guarantee as
    _atomic_write_json, for the raw jsonl the torn-line repair rewrites)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def result_to_model_text(result):
    """Serialise a tool result dict to the exact string the model receives in
    its `tool_result.content` field, live and on replay.

    Strips any underscore-prefixed key: those are UI-only telemetry (for
    example `_field_diffs`, `_canonicalisations`) captured for the transcript
    renderer, and they must never enter conversation history. The session
    event log retains the full unstripped dict as the transcript record; only
    the model-visible payload loses the underscore keys.

    This is the single canonical serialisation shared by the live extractor
    loop (orchestrator) and `replay_messages`, so a resumed run rebuilds
    byte-identical tool_result content instead of feeding the model telemetry
    it never saw live.
    """
    public = {k: v for k, v in result.items() if not k.startswith("_")}
    return json.dumps(public, ensure_ascii=False)


class Session:
    """One agentic-extraction session for one study.

    Use `Session.create(...)` to start a fresh session and
    `Session.resume(...)` to reattach to an in-progress one.
    """

    # The checker is not a phase: it runs inside the extractor's (and
    # optionally the reviewer's) tool calls, so a run is only ever extracting,
    # reviewing, or done.
    PHASES = {"extracting", "final_review", "done"}

    def __init__(self, session_dir, meta):
        # Resolved to an absolute path here, the one place that owns the
        # session's on-disk identity. `session_dir` and
        # `extraction_record_path` are recorded into run_log.json, the
        # cross-run index; a relative path there is resolvable only against
        # the invocation cwd, which no artefact records, so resolution
        # happens at construction while that cwd is still current.
        #
        # `resolve()` over `os.path.abspath` deliberately: it canonicalises,
        # following symlinks, so the recorded path names the directory the
        # run ACTUALLY wrote to and two spellings of one destination record
        # the same path. The cost is that a recorded path need not be
        # string-prefixed by the `--out` the operator typed (on macOS
        # `--out /tmp/x` records `/private/tmp/x/...`), so a consumer must
        # resolve its own side before comparing rather than matching
        # prefixes.
        self.session_dir = Path(session_dir).resolve()
        self.meta = meta
        # The extraction output sits alone at the top of the session
        # directory; everything else is the record of how it was produced
        # and lives under `diagnostics/`.
        self.extraction_record_path = self.session_dir / "extraction_output.json"
        self.diagnostics_dir = self.session_dir / "diagnostics"
        self.meta_path = self.diagnostics_dir / "run.json"
        self.field_history_path = self.diagnostics_dir / "field_history.json"
        self.tool_calls_path = self.diagnostics_dir / "tool_calls.jsonl"
        self.api_calls_path = self.diagnostics_dir / "api_calls.jsonl"
        self.instrument_dir = self.diagnostics_dir / "instrument"
        # Thread-safe writer for parallel checker fan-out.
        from meltiro.api_logger import ApiCallWriter
        self._api_writer = ApiCallWriter(self.api_calls_path)

    @property
    def diagnostics(self):
        """The diagnostics level governing THIS segment of the run.

        Recorded in run.json (`diagnostics`) rather than held only in memory,
        so a finished session says which artefacts it was asked to keep and a
        reader knows why a file is absent. It sits next to `caps` because it
        is the same kind of thing: an operational setting that rides in no
        fingerprint and that a resume may legitimately change.
        """
        return validate_diagnostics(self.meta.get("diagnostics"))

    # ----------------------------------------------------------------------
    # API call audit log (one line per call: extractor / checker / review)
    # ----------------------------------------------------------------------

    def log_api_call(self, call_type, request_kwargs, response, **extra):
        """Capture one API call verbatim. Image bytes in request messages
        are redacted to {image_ref, sha256, byte_length} stubs; the
        canonical bytes live in the paper bundle's figures. Their hashes
        are captured separately on the session at start time (see
        capture_image_hashes), so the transcript can flag drift.

        The wire log is the one artefact only `--diagnostics full` keeps. At
        the lower levels this returns without writing, so `api_calls.jsonl`
        is never created at all rather than created empty: an empty file
        would read as a run that made no calls.
        """
        self.write_api_call_entry(
            self.api_call_entry(call_type, request_kwargs, response, **extra))

    def api_call_entry(self, call_type, request_kwargs, response, **extra):
        """The entry this call earns, or None at a level that keeps no wire
        log.

        Composition touches no disk and fails only over its inputs: a
        diagnostics level this session cannot read, a request or response
        shape `make_entry` cannot compose. It is separable from the write for
        exactly that reason — a caller that must survive an IO fault
        (`Orchestrator._log_api_call_guarded`) can still let those raise.
        """
        if not captures_api_calls(self.diagnostics):
            return None
        from meltiro.api_logger import make_entry
        return make_entry(call_type, request_kwargs, response, **extra)

    def write_api_call_entry(self, entry):
        """Append one composed entry to `api_calls.jsonl`, under the writer's
        lock. A None entry is a level that keeps no wire log, and there is
        nothing to append."""
        if entry is None:
            return
        self._api_writer.write(entry)

    def capture_image_hashes(self, figure_paths):
        """At session start, sha256 every cropped figure used by this
        study so subsequent re-cropping is detectable. `figure_paths`
        is an iterable of pathlib.Path. Stored in meta as
        `image_hashes: {label: {sha256, byte_length}}`.

        The recipe is `fingerprint.figure_hashes`, the same one `figures_fp`
        is built from, so this per-image record and the bundle fingerprint
        beside it are the same numbers.
        """
        self.meta["image_hashes"] = figure_hashes(figure_paths)
        self.write_meta()

    def capture_bundle_fingerprint(self, bundle):
        """At session start, record the PAPER's own fingerprint in meta.

        Six values (`text_fp`, `figures_fp`, `manifest_fp`, `tables_fp`,
        `supplements_fp`, `bundle_fp`; see
        `fingerprint.bundle_fingerprint`) naming the input this run was given.
        They are folded into no other fingerprint: the config axes describe the
        question and this describes what the question was asked of, so a reader
        holding a run record has both halves and can compare either on its own.

        Written here, at session start, from the bundle as loaded, next to the
        per-image hashes and on the same terms: it is a fact about the input
        the run began with, and a session records it once.
        """
        self.meta.update(bundle_fingerprint(bundle))
        self.write_meta()

    # ----------------------------------------------------------------------
    # Creation / Resume
    # ----------------------------------------------------------------------

    @classmethod
    def create(cls, study_id, *, config_fp, checker_fp, review_fp,
               instrument_fp, extractor_call_fp, checker_call_fp,
               review_call_fp, engine_fp,
               extractor_model, checker_model, review_model,
               tool_set_hash, template_hash, prompt_hash,
               runs_dir, tool_definitions=None, system_prompt=None,
               user_prompt=None, image_labels=None,
               review_system_prompt=None, checker_system_prompt=None,
               checker_user_scaffold=None,
               caps=None, structure=None,
               checker_context_chars=None, decoding_specified=None,
               diagnostics=DEFAULT_DIAGNOSTICS):
        """Start a fresh session and write the initial run.json + empty
        extraction output.

        The instrument as sent is captured into `diagnostics/instrument/` so
        the transcript view never needs to reach back to local config / paper
        text / figure directories:

        - `tool_definitions.json`: the tool schema as sent
        - `system_prompt.txt`: the rendered EXTRACTOR system message text
        - `user_prompt.txt`: the rendered user-message text portion
          (paper text, then each attached exhibit under its label and
          caption, or the statement that none accompany the study)
        - `review_system_prompt.txt`: the rendered reviewer system message,
          absent when the reviewer stage is off
        - `checker_system_prompt.txt`: the rendered checker system message,
          one string for the whole run, absent when the checker is off
        - `checker_user_scaffold.txt`: the per-field scaffold every check of
          this run was rendered from, slot tokens and all, absent when the
          checker is off
        - `image_labels.json`: the exhibits attached, in the order the
          message attached them, each as `{"label", "caption"}` — the label
          an `<img>` citation names, and the caption the paper prints beside
          the crop (null where the manifest declared none)

        All three prompts render deterministically from the config bundle,
        the template, and the role's model, so all three are captured HERE,
        at creation: capturing a stage's prompt when it first runs would
        leave a run that never reached the stage with no record of what it
        would have asked.

        Any not supplied is simply not written, and nothing downstream fills
        the gap from the current on-disk config: the transcript reports the
        file as absent instead, because re-rendering it later would be valid
        only if nothing in the config bundle had changed since, which no
        artefact records.

        `--diagnostics minimal` skips the instrument capture entirely. It is
        captured ONCE, here, so a resume at a higher level does not backfill
        it: re-rendering later would be a reconstruction dressed as a
        capture.
        """
        diagnostics = validate_diagnostics(diagnostics)
        runs_dir = Path(runs_dir)
        ts = _utc_now_compact()
        short_fp = config_fp.split(":", 1)[-1][:6]
        session_id = f"{ts}_{short_fp}"
        session_dir = runs_dir / str(study_id) / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=False)

        # Origin anchor for provenance: the short git commit the running
        # meltiro package came from (not the operator's cwd) at session
        # start, plus whether the tree carrying it had uncommitted changes.
        # It says where the running code was read from; `engine_fp` below
        # identifies the code itself. The dirty flag alone is None for an
        # installed copy, which has no working tree; both are None when
        # nothing on disk records the origin (see run_log.git_state). This
        # reading is taken at session start; the run-log entry takes its own
        # at append time, so the two can legitimately differ if the code
        # changes mid-run (see run_log.git_state).
        git_commit, git_dirty = git_state()
        # The other half of the engine: direktoro builds the provider-call
        # identity block leading every stage fingerprint and resolves what is
        # actually sent. None when not installed
        # (see run_log.direktoro_version).
        direktoro_ver = direktoro_version()
        # And what admitted the input: alteksto decides whether the directory
        # this run was handed is a bundle, and which of its files are crops.
        # It moves no fingerprint (see run_log.alteksto_version).
        alteksto_ver = alteksto_version()

        meta = {
            "session_id": session_id,
            "study_id": str(study_id),
            "status": "in_progress",
            "current_phase": "extracting",
            "started_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            # The engine's identity, human-readable half: the versions say
            # which releases asked the question, the commit says where
            # they came from — a checkout, or an installed copy's own record
            # — and `engine_fp` below identifies the code itself. `meltiro_version` is always present, including in
            # an installed copy with no repository around it;
            # `direktoro_version` and `alteksto_version` are null only when
            # those packages are absent.
            "meltiro_version": __version__,
            "direktoro_version": direktoro_ver,
            "alteksto_version": alteksto_ver,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "config_fp": config_fp,
            "checker_fp": checker_fp,
            "review_fp": review_fp,
            # The three orthogonal axes, recorded beside the stage fingerprints
            # above. Each stage fingerprint blends what the config author wrote
            # with which model was called, which is the right identity for
            # refusing a drifted resume and the wrong one for comparing runs.
            # These separate the two, plus the engine, so the questions a
            # consumer actually asks are one-field diffs:
            #   same instrument, different model -> instrument_fp equal, a
            #     *_call_fp differs (config_fp/checker_fp/review_fp all move
            #     together here, so they cannot answer it).
            #   same everything, different meltiro -> instrument_fp and every
            #     *_call_fp equal, engine_fp differs (no stage fingerprint
            #     moves at all here, so they cannot answer this either).
            # A disabled stage records a null call fingerprint, mirroring its
            # null stage fingerprint. See fingerprint's module docstring.
            "instrument_fp": instrument_fp,
            "extractor_call_fp": extractor_call_fp,
            "checker_call_fp": checker_call_fp,
            "review_call_fp": review_call_fp,
            # The engine axis: hash(meltiro_version | meltiro source digest |
            # direktoro_version | direktoro source digest). The loose fields
            # above stay, because a human reads those; this is the value a
            # consumer compares on, and it identifies the code by content, so
            # two runs share it exactly when the same source ran (see
            # fingerprint.engine_fingerprint).
            "engine_fp": engine_fp,
            # The whole-run fingerprint: hash(config_fp | checker_fp |
            # review_fp | engine_fp) with a documented sentinel for a disabled
            # stage (see fingerprint.run_fingerprint). It identifies the full
            # run-producing configuration, so downstream consumers key a run on
            # this rather than on config_fp alone (which collapses mixed
            # checker/reviewer arms that share an extractor config), building
            # `llm:<run_fp>` producer strings. Derived HERE from the values
            # this session is created with, rather than passed in, so
            # meta.run_fp can never disagree with the fingerprints recorded
            # alongside it. The dry-run report computes it from the same
            # function on the same inputs, so the two never drift. Folding
            # engine_fp in means run_fp equality does not survive a meltiro
            # release, which is the honest reading and is why the axes above
            # are recorded separately.
            "run_fp": run_fingerprint(
                config_fp, checker_fp, review_fp, engine_fp),
            "extractor_model": extractor_model,
            "checker_model": checker_model,
            "review_model": review_model,
            # Characters of surrounding paper text the checker was shown on
            # each side of each matched quote. Null when the checker is off,
            # like `checker_model` and `checker_fp` beside it. It is config
            # identity, not an operational budget: it rides in `checker_fp`,
            # so a resume that changes it is refused by the drift gate, and it
            # is recorded here so a finished session says how wide the
            # checker's view of the paper was without re-reading the bundle.
            "checker_context_chars": checker_context_chars,
            "tool_set_hash": tool_set_hash,
            "template_hash": template_hash,
            "prompt_hash": prompt_hash,
            "tool_call_count": 0,
            # Total per-field CHECKS made in this session, across every tool
            # call that triggered the fan-out. Not a count of rounds: there are
            # none. Not a count of provider calls either: a check re-asked
            # after a reply that recorded no verdict made two of those and is
            # one check here.
            "checker_calls_run": 0,
            # Per-entity next record-id index, keyed by entity name. Session
            # bookkeeping, not part of the consumer-facing extraction output:
            # it is persisted here (never in extraction_output.json) and
            # threaded back into the ExtractionRecord on resume so a record id
            # is never reissued after a removal. Empty at creation (no records
            # minted yet); write_extraction_record syncs it after every write.
            "record_id_counters": {},
            # Whether the extractor has recorded its initial check. Session
            # bookkeeping on the same footing as the id counters: it gates
            # every extractor mutation, so it must survive a pause or a
            # resumed extractor would be told to make a pre-extraction report
            # after the fact. It cannot be derived from the extraction output,
            # because a template that declares no initial-check fields leaves
            # the block legitimately empty after a successful call.
            "initial_check_recorded": False,
            # Non-fatal degradations recorded during the run (e.g. the
            # checker falling back to minimal identity context, or a
            # manifest-summary vs extracted-summary mismatch). Empty by
            # default; appended to via `add_warning`.
            "warnings": [],
            # The bounds the orchestrator ACTUALLY honoured, recorded with the
            # run so the transcript view shows them even if the config YAML is
            # edited later. Written here at creation, but NOT frozen: the
            # extractor's tool-call cap and the reviewer's are operational
            # budgets that ride in no fingerprint, so a resume may raise them
            # (the documented recovery from a cap-hit pause) and rewrites these
            # to the values then in force. The `resumed` event carries each
            # bound's new and previous value, and is the per-segment history
            # this snapshot cannot hold.
            #
            # `max_checks_per_field` is NOT one of these and is not here: it is
            # substituted into the prompts and folded into `structure_hash`, so
            # it rides in `config_fp`, `checker_fp`, `review_fp` and
            # `instrument_fp`, and changing it refuses a resume at the drift
            # gate. It is recorded under `structure` below, with the rest of
            # what the fingerprints cover.
            "caps": caps or {},
            # How much of the deterministic record this segment kept (see
            # meltiro.diagnostics). Operational, not methodology: it rides in
            # no fingerprint, and it lives here for the same reason `caps`
            # does, so a reader of a finished session knows why a file is
            # absent. Like the caps it reports the CURRENT segment's setting
            # and is rewritten by every resume; the `resumed` events in
            # tool_calls.jsonl are the per-segment record.
            "diagnostics": diagnostics,
            # Pipeline structure for this run: which optional stages are on.
            # {"checker": bool, "review": bool, "max_checks_per_field": int,
            # "check_reviewer_edits": bool}. A stage that is off records a null
            # stage fingerprint (checker_fp / review_fp) above, and its model
            # may be null too.
            "structure": structure or {},
            # The operator's decoding block per role, verbatim, as the config
            # bundle wrote it; a role that wrote none is absent. Its
            # counterpart `decoding_params` is written per role as each role's
            # first response comes back, and holds what the WIRE carried. The
            # two differ whenever a model refuses a control it was given: the
            # value is dropped, silently and by design, and moves no
            # fingerprint. Recording only the wire side would make "wrote a
            # value the model refused" and "wrote nothing" the same artefact.
            #
            # A key written with a null value is recorded as written, though
            # the resolver reads a null as "unspecified" and sends nothing for
            # it. The two documents answer different questions: this one says
            # what the bundle states, and `decoding_params` says what the wire
            # carried. Normalising the null away here would leave a bundle
            # that says `temperature: null` indistinguishable in the artefact
            # from one that never names the control, which is a difference an
            # operator wrote down on purpose.
            #
            # Rewritten by every resume to the CURRENT segment's blocks (see
            # `Orchestrator.resume_session`), like `caps` and `diagnostics`:
            # a refused control moves no fingerprint, so an edit to one is
            # admitted by the drift gate and has to be visible in the record.
            "decoding_specified": decoding_specified or {},
        }
        s = cls(session_dir, meta)
        s.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        s.write_meta()
        # Empty extraction output on disk.
        _atomic_write_json(s.extraction_record_path, ExtractionRecord().to_dict())
        if captures_instrument(diagnostics):
            s.instrument_dir.mkdir(parents=True, exist_ok=True)
            if tool_definitions is not None:
                _atomic_write_json(
                    s.instrument_dir / "tool_definitions.json",
                    tool_definitions)
            if system_prompt is not None:
                (s.instrument_dir / "system_prompt.txt").write_text(
                    system_prompt, encoding="utf-8")
            if user_prompt is not None:
                (s.instrument_dir / "user_prompt.txt").write_text(
                    user_prompt, encoding="utf-8")
            if review_system_prompt is not None:
                (s.instrument_dir / "review_system_prompt.txt").write_text(
                    review_system_prompt, encoding="utf-8")
            if checker_system_prompt is not None:
                (s.instrument_dir / "checker_system_prompt.txt").write_text(
                    checker_system_prompt, encoding="utf-8")
            if checker_user_scaffold is not None:
                (s.instrument_dir / "checker_user_scaffold.txt").write_text(
                    checker_user_scaffold, encoding="utf-8")
            if image_labels is not None:
                _atomic_write_json(
                    s.instrument_dir / "image_labels.json", list(image_labels))
        # Initialise tool_calls.jsonl (touch).
        s.tool_calls_path.touch()
        s.append_event({"event": "session_started", "ts": _utc_now_iso()})
        return s

    @classmethod
    def resume(cls, session_dir, *, expected_config_fp=None,
               expected_checker_fp=None, expected_review_fp=None,
               expected_bundle=None):
        """Resume an in-progress session. Refuses on status mismatch, on drift
        in ANY supplied stage fingerprint (config_fp / checker_fp / review_fp),
        and on a changed PAPER (`expected_bundle`).

        Each expected_* argument is checked only when supplied (non-None), so
        callers that care about a subset can pass just those. The orchestrator
        passes all four so a changed extractor, checker, OR review config —
        or a changed paper — blocks the resume rather than silently continuing
        under new inputs.

        `expected_bundle` is the `PaperBundle` this resume was handed, and it
        is checked against the five axes `capture_bundle_fingerprint` recorded
        at session start, through the same `bundle_fingerprint` recipe. The
        stage fingerprints cannot stand in for it: the paper is folded into
        none of them by design (the config axes say what was asked, the bundle
        axes say what it was asked of), so an edited `text.md`, a re-cropped
        figure or a rewritten manifest moves nothing they cover. A resumed
        session replays a conversation whose earlier turns quoted the OLD text
        and whose evidence was verified against it, so continuing under new
        material would produce one extraction citing two different papers.

        The session continues at the diagnostics level run.json records. The
        level is operational and rides in no fingerprint, so a resume MAY
        change it; that is the orchestrator's job, on the same terms as the
        tool-call caps and recorded on the same `resumed` event.

        What this gate does NOT check is the ENGINE. `meltiro_version`,
        `direktoro_version`, `alteksto_version`, `git_commit`, `git_dirty`,
        `engine_fp` and `run_fp` are all read once, at creation, and none of
        them can refuse a resume, so a resume across a new commit or an
        upgraded package is admitted. That is deliberate: `engine_fp` moves on
        any edit to either package's source, and refusing on it would refuse
        the documented cap-hit recovery (pause, raise the cap, resume) to
        anyone whose tree moved in between, which working on the engine
        guarantees. What changes with the engine is the
        framing the pipeline writes around the config's prompts; what changes
        with a drifted stage fingerprint is the question itself, which is why
        that one refuses. The engine change is recorded instead: the
        orchestrator writes each segment's own version, commit, dirtiness and
        engine fingerprint onto the `resumed` event it appends, and warns when
        the identity has moved (see `Orchestrator.resume_session` and
        `_warn_engine_drift`). `run.json`'s copy therefore reads as the engine
        at session START, which is what `fingerprint.run_fingerprint`
        documents `run_fp` to mean.

        The engine is read for one purpose here: a stage fingerprint folds in
        engine-owned material, so it drifts on an engine change with the config
        bundle unedited. Every refusal below therefore names the axis that
        moved, or says that the session does not record enough to tell
        (`_drift_axis`), rather than sending an operator to look for an edit
        nobody made.
        """
        session_dir = Path(session_dir)
        meta_path = Session.meta_path_for(session_dir)
        if not meta_path.exists():
            raise ResumeRefused(
                f"diagnostics/run.json missing at {meta_path}; can't resume.")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # Loud rather than defaulted: a session directory whose run.json does
        # not say what it kept is not one this version wrote, and guessing a
        # level would silently decide which artefacts the rest of the run
        # produces.
        validate_diagnostics(meta.get("diagnostics"))
        if meta.get("status") != "in_progress":
            raise ResumeRefused(
                f"Session status is {meta.get('status')!r}, not "
                f"'in_progress'. Cannot resume."
            )
        for label, expected, key in (
            ("Config", expected_config_fp, "config_fp"),
            ("Checker", expected_checker_fp, "checker_fp"),
            ("Review", expected_review_fp, "review_fp"),
        ):
            if expected is not None and meta.get(key) != expected:
                raise ResumeRefused(
                    f"{label} fingerprint drift: session was started with "
                    f"{meta.get(key)}, current config is {expected}. Refusing "
                    f"to resume; start a fresh session. {_drift_axis(meta)}"
                )
        if expected_bundle is not None:
            current = bundle_fingerprint(expected_bundle)
            # An axis the session never recorded is UNDETERMINED, not moved.
            # A session started before an axis existed has no value to
            # compare, and reading the absent key as a mismatch would report
            # a paper that changed and prescribe a fix that cannot work —
            # no bundle hashes to None, so re-pointing `--paper` at the
            # original directory returns the same refusal. The engine axis is
            # read the same way (see `_drift_axis`).
            unrecorded = [axis for axis in BUNDLE_AXES
                          if meta.get(axis) is None]
            moved = [axis for axis in BUNDLE_AXES
                     if meta.get(axis) is not None
                     and meta.get(axis) != current[axis]]
            if moved:
                detail = "; ".join(
                    f"{axis}: session started with {meta.get(axis)}, the "
                    f"bundle now supplied is {current[axis]}"
                    for axis in moved)
                raise ResumeRefused(
                    f"the paper bundle changed ({detail}). Refusing to "
                    f"resume: the conversation being replayed quotes the text "
                    f"and figures this session started with, and its recorded "
                    f"evidence was verified against them. Point --paper at "
                    f"the original bundle, or start a fresh session against "
                    f"the new one."
                )
            if unrecorded:
                named = ", ".join(unrecorded)
                raise ResumeRefused(
                    f"this session records no {named}, so whether the paper "
                    f"is the one it started against cannot be settled: those "
                    f"axes were not written when it began, and the axes it "
                    f"does record all match. The paper may be unchanged — "
                    f"re-pointing --paper at the original bundle will not "
                    f"clear this, because no bundle hashes to a missing "
                    f"value. Start a fresh session against the bundle you "
                    f"mean to extract, under this engine."
                )
        s = cls(session_dir, meta)
        # A hard kill (power loss) mid-append can leave the last line of the
        # append-only log truncated to invalid JSON. Repair that single torn
        # tail before anything reads the events, so an otherwise recoverable
        # session is resumable. Any malformed line earlier in the log is real
        # corruption and is left for read_events to reject loudly.
        s._repair_torn_final_line()
        return s

    @staticmethod
    def meta_path_for(session_dir):
        """Where run.json lives inside a session directory.

        The one place outside `__init__` that spells the path, so a caller
        that has a directory but not yet a Session (resume, auto-resume) can
        find the file without repeating the layout.
        """
        return Path(session_dir) / "diagnostics" / "run.json"

    @classmethod
    def in_progress_sessions(cls, study_id, *, runs_dir):
        """Every in-progress session for a study, as `(session_dir, meta)`.

        The unfiltered population `find_in_progress` chooses from. Exposed on
        its own because a caller that is about to start a FRESH run needs to
        know whether it is passing paid work by: "nothing to resume" and
        "sessions to resume, none of them under this configuration" are
        different facts about a run's money, and only one of them is worth
        saying out loud.

        A directory with no readable `run.json` is skipped rather than raised
        on: this is a discovery scan over whatever is on disk, and one
        unreadable neighbour must not stop a resumable session being found.
        """
        sessions_root = Path(runs_dir) / str(study_id) / "sessions"
        if not sessions_root.is_dir():
            return []
        found = []
        for d in sorted(sessions_root.iterdir()):
            meta_path = cls.meta_path_for(d)
            if not meta_path.exists():
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if m.get("status") != "in_progress":
                continue
            found.append((d, m))
        return found

    @staticmethod
    def newest_resumable(sessions, *, expected_config_fp=None):
        """The most-recent session of an already-scanned population, or None.

        `sessions` is `[(session_dir, meta)]` as `in_progress_sessions`
        returns. Split from the scan so a caller that needs BOTH answers — the
        one session to resume, and how many it is about to spend past — gets
        them from a single pass over the directory rather than from two, which
        could disagree if a concurrent run finished a session between them.

        Filters on config_fp when supplied, so config drift does not surface a
        stale session.
        """
        candidates = [
            (m.get("updated_at", ""), d)
            for d, m in sessions
            if expected_config_fp is None
            or m.get("config_fp") == expected_config_fp
        ]
        if not candidates:
            return None
        candidates.sort()
        return candidates[-1][1]

    @classmethod
    def find_in_progress(cls, study_id, *, runs_dir,
                         expected_config_fp=None):
        """Return the most-recent in-progress session for a study, or None.

        The scan and the choice in one call, for a caller that wants only the
        answer.
        """
        return cls.newest_resumable(
            cls.in_progress_sessions(study_id, runs_dir=runs_dir),
            expected_config_fp=expected_config_fp)

    # ----------------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------------

    def write_meta(self):
        self.meta["updated_at"] = _utc_now_iso()
        _atomic_write_json(self.meta_path, self.meta)

    def write_field_history(self):
        """Derive `diagnostics/field_history.json` from the event log and
        write it. Returns the document.

        Called whenever the run stops, at a pause as well as at finalisation,
        so a session a reader opens always carries a field history current to
        the last thing that happened. It is rewritten wholesale each time
        rather than appended to: it is derived, so there is nothing to
        preserve, and rebuilding it from the log is the only way it can be
        guaranteed to agree with the log.
        """
        history = build_field_history(self.read_events())
        _atomic_write_json(self.field_history_path, history)
        return history

    def write_transcript(self):
        """Render `diagnostics/transcript.md` from this session on disk and
        write it. Returns the path.

        Called whenever the run stops, at a pause as well as at finalisation,
        and LAST, after run.json, the extraction output, and the field history
        have all been written, because the document is rendered from those
        files rather than from this object's memory. That ordering is what
        makes the copy a run writes byte-identical to what
        `meltiro transcript SESSION_DIR` produces from the same session
        afterwards: both read the same finished files through the same code.
        """
        from meltiro.transcript import write_transcript
        return write_transcript(self.session_dir)

    def add_warning(self, message):
        """Append a non-fatal warning to `meta.warnings` and persist, skipping
        an exact duplicate so a resumed run never re-persists a warning it
        already recorded.

        Warnings record degradations that did NOT stop the run (e.g. the
        checker falling back to minimal identity context) or a property of
        the finished artefact worth flagging. The in-memory latches gating
        some of these reset on resume, so a run resumed N times would
        otherwise append N identical copies; the strings are fully keyed
        (study and specific degradation), so exact-match dedup suffices.

        The key is ALWAYS present: `Session.create` writes `"warnings": []`, so
        a session that recorded none carries an empty list rather than no key
        and a reader never has to tell "nothing to report" apart from "written
        by something that did not record this". The `setdefault` below is a
        guard for a meta dict assembled some other way, not the ordinary path.
        """
        warnings = self.meta.setdefault("warnings", [])
        if message in warnings:
            return
        warnings.append(message)
        self.write_meta()

    def write_extraction_record(self, extraction_record):
        # Persist the per-entity record-id counter (run.json) BEFORE the
        # extraction output. The two go to separate files, so a crash between
        # the writes leaves one ahead of the other. Counter-first means the
        # persisted counter is always at or ahead of the persisted records:
        # the worst case on resume is a skipped number (a gap, explicitly
        # fine), never a reissued id that would collide with a surviving
        # record. The counter is session bookkeeping and lives only in
        # run.json (never in the consumer-facing extraction_output.json).
        self.meta["record_id_counters"] = extraction_record.record_id_counters()
        self.write_meta()
        _atomic_write_json(self.extraction_record_path, extraction_record.to_dict())
        # The initial-check gate flag goes out AFTER the output, which is the
        # opposite ordering to the counter above and for the opposite reason.
        # The counter is safe when it runs ahead (worst case a skipped id);
        # this flag is safe when it lags. A crash between the two writes then
        # leaves the gate SHUT over an output that already holds the initial
        # check, costing one repeated call that simply revises it. Flag-first
        # would fail the other way: gate open, initial check absent, and an
        # extraction resuming as though a report on the inputs had been made.
        # Written only when it changes, so this costs one extra meta write per
        # run rather than one per tool call.
        if self.meta.get("initial_check_recorded") != \
                extraction_record.initial_check_recorded:
            self.meta["initial_check_recorded"] = \
                extraction_record.initial_check_recorded
            self.write_meta()

    def append_event(self, event):
        """Append one JSON line to tool_calls.jsonl. `event` is a dict; a
        `ts` field is added if absent."""
        if "ts" not in event:
            event = {"ts": _utc_now_iso(), **event}
        with open(self.tool_calls_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def read_events(self):
        """Return the list of all logged events. Skips blank lines and
        raises SessionError on a malformed jsonl entry.

        A torn FINAL line (a hard kill mid-append truncates only the last
        line) is not this method's concern: Session.resume repairs it before
        any read. So a malformed line reaching here is genuine corruption
        earlier in the log, and it stays a loud failure.
        """
        if not self.tool_calls_path.exists():
            return []
        out = []
        with open(self.tool_calls_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise SessionError(
                        f"Malformed JSON on line {i} of "
                        f"{self.tool_calls_path}: {e}. This is not the "
                        "auto-recoverable torn final line (a hard kill "
                        "mid-append truncates only the LAST line, which "
                        "resume repairs); a malformed line earlier in the log "
                        "is corruption that needs manual inspection before "
                        "this session can be resumed."
                    )
        return out

    def _repair_torn_final_line(self):
        """Drop a torn final line from tool_calls.jsonl, if present.

        A power loss mid-append can leave the last line of the append-only
        log truncated to invalid JSON. That single torn TAIL is recoverable:
        the extraction record is persisted AFTER the events of its batch (see
        write_extraction_record and the extractor loop's dispatch/write
        ordering), so the persisted record can never lead the event log, and
        dropping the unparseable tail cannot desync the record from the
        replayed conversation. At worst it drops a tail event the record had
        not yet incorporated, which is the ordinary mid-batch-crash situation
        resume already tolerates.

        Only the final content line is repaired. A malformed line with valid
        content after it is not a torn append; it is left in place for
        read_events to reject loudly. The torn line is physically removed (an
        atomic rewrite) so the resumed run can keep appending without turning
        the torn tail into a mid-file malformed line on the next read. What
        was dropped is recorded in meta.warnings and a torn_line_dropped
        event.

        A softer variant of the same torn write leaves the final event's JSON
        intact but drops only its trailing newline. Nothing is lost there, so
        the newline is simply restored (no warning) to stop the next append
        concatenating onto the unterminated line.
        """
        if not self.tool_calls_path.exists():
            return
        raw = self.tool_calls_path.read_text(encoding="utf-8")
        if not raw:
            return
        # A well-formed append-only log ends every event with "\n", so the
        # element after the final newline is normally empty. split keeps the
        # exact per-line bytes so a rewrite of the kept region is byte-for-byte
        # identical to the original.
        lines = raw.split("\n")
        last_idx = None
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip():
                last_idx = i
                break
        if last_idx is None:
            return  # only blank lines: nothing to repair
        try:
            json.loads(lines[last_idx])
        except json.JSONDecodeError:
            pass  # torn tail: dropped below
        else:
            # The last content line is intact JSON. If the file ends with a
            # newline it is clean; nothing to do. If it does not, a torn write
            # dropped only the record separator: the event survived, so
            # restore the newline (no data lost, no warning) rather than let
            # the next append concatenate onto it and corrupt the line.
            if not raw.endswith("\n"):
                _atomic_write_text(self.tool_calls_path, raw + "\n")
            return
        torn = lines[last_idx]
        # Rewrite the log without the torn tail (and any trailing blanks),
        # preserving every kept line's bytes.
        kept = lines[:last_idx]
        new_text = "\n".join(kept)
        if kept:
            new_text += "\n"
        _atomic_write_text(self.tool_calls_path, new_text)
        message = (
            f"Dropped a torn final line ({len(torn)} bytes) from "
            f"tool_calls.jsonl on resume (line {last_idx + 1}): a hard kill "
            "left the last append incomplete. The dropped bytes are recorded "
            "in the torn_line_dropped event."
        )
        self.add_warning(message)
        self.append_event({
            "event": "torn_line_dropped",
            "line_number": last_idx + 1,
            "byte_length": len(torn),
            "raw": torn,
        })

    def load_extraction_record(self):
        """Load the canonical extraction output from disk into an ExtractionRecord instance.

        The per-entity record-id counter is not stored in the extraction
        output; it is threaded in from run.json so a resumed session never
        reissues a removed record's id.
        """
        if not self.extraction_record_path.exists():
            record = ExtractionRecord()
        else:
            with open(self.extraction_record_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            record = ExtractionRecord.from_dict(d)
        record.set_record_id_counters(self.meta.get("record_id_counters", {}))
        record.set_initial_check_recorded(
            self.meta.get("initial_check_recorded", False))
        # The completion claim is deliberately NOT restored. It is a
        # content-staleness bit — cleared by every write, including the
        # reviewer's — not a record of which stage the run reached, and a
        # resume needs the latter. `meta.current_phase` carries that, and
        # `Orchestrator._run_to_stop` routes on it. Restoring the claim would
        # let a resumed extractor treat a record the reviewer had since edited
        # as its own finished work.
        return record

    def max_turn_id(self):
        """Largest `turn_id` present in the event log, or 0 when none.

        Used on resume to seed the orchestrator's session-global turn
        counter so replayed turns never reuse an id an earlier loop wrote
        (which would merge two turns into one message on the next replay).
        """
        top = 0
        for e in self.read_events():
            tid = e.get("turn_id")
            if isinstance(tid, int) and tid > top:
                top = tid
        return top

    # ----------------------------------------------------------------------
    # Conversation replay (used by --resume)
    # ----------------------------------------------------------------------

    def replay_messages(self):
        """Rebuild the `messages` list, in direktoro's canonical conversation
        format, from the saved tool_calls jsonl, byte-identical to what the
        live run sent.

        Each turn contributes an assistant message and (usually) a paired
        user message. The assistant message is the verbatim ordered content
        the orchestrator logged for that turn in its `assistant_message`
        event, so the provider's original block order (text interleaved with
        tool_use) is preserved exactly rather than forced text-first. The
        user side is rebuilt from the same tool_result serialisation the live
        loop used (result_to_model_text, which strips UI-only telemetry).

        Events that aren't turn traffic (session_started, resumed,
        terminate, ...) are ignored here; they are contextual or carry their
        own message construction in the orchestrator.

        The rebuilt conversation always ends on a USER message. A trailing turn
        that logged its assistant side and never its user side is a kill in the
        window between those two appends, and it is dropped rather than
        replayed — see the guard below for why an assistant-final conversation
        is not merely untidy but unsendable.

        Only the EXTRACTOR's turns are rebuilt. The final reviewer runs its own
        tool loop, whose conversation is deliberately fresh-context and is not
        `self.messages`; replaying it here would splice reviewer turns into the
        extractor's conversation. A resume after a crash mid-review re-runs the
        review from scratch instead (see `_review_loop`). Its dispatches are
        named apart already (`review_tool_call`, `review_reprompt`), but its
        `assistant_message` and `assistant_text` events share these names, so
        they are marked `stage: "review"` and excluded here. Extractor events
        carry no `stage` key, so the filter admits `None`, and admits an
        explicit "extractor" too; anything else is excluded, which fails closed
        for any stage added later.

        Returns the list of message dicts in the order the canonical
        conversation format expects.
        """
        events = self.read_events()
        # Group by turn_id (insertion order). Each turn collects the verbatim
        # assistant content (assistant_message, the authoritative assistant
        # side; a turn missing it is a crash artefact and replay refuses, see
        # _replay_assistant_content) and the user-side blocks.
        turns = {}
        order = []
        for e in events:
            ev = e.get("event")
            if ev not in ("assistant_message", "tool_call_applied",
                          "tool_call_failed", "tool_call_partial",
                          "assistant_text", "extractor_reprompt"):
                continue
            if e.get("stage") not in (None, "extractor"):
                continue
            turn_id = e.get("turn_id")
            if turn_id is None:
                continue
            if turn_id not in turns:
                turns[turn_id] = {"assistant": None,
                                  "has_text_event": False, "user": []}
                order.append(turn_id)
            t = turns[turn_id]
            if ev == "assistant_message":
                # The exact ordered content list appended to self.messages
                # live; the authoritative assistant side for this turn.
                t["assistant"] = e["content"]
            elif ev == "assistant_text":
                # The human-transcript prose record; only a marker here (the
                # verbatim text already lives in the assistant_message event).
                t["has_text_event"] = True
            elif ev == "extractor_reprompt":
                # The tool-free re-prompt the orchestrator sent back after a
                # text-only turn. It is the user side of that turn.
                t["user"].append({
                    "type": "text", "text": e.get("text", ""),
                })
            elif ev in ("tool_call_applied", "tool_call_failed",
                        "tool_call_partial"):
                t["user"].append({
                    "type": "tool_result",
                    "tool_use_id": e["tool_use_id"],
                    "content": result_to_model_text(e["result"]),
                })

        # A LAST turn with an assistant side and no user side is dropped, and
        # only the last one. Every completed turn has both — a tool-calling
        # turn's tool results, a text-only turn's re-prompt — so this shape is
        # a kill in the window between the two appends. Replaying it would end
        # the conversation on an assistant message, which the next call sends
        # as a PREFILL: the model is asked to continue its own narration
        # instead of answering, and a prefill ending in whitespace is refused
        # by the API outright, so the resume fails on its first call. Dropping
        # the turn re-sends it cleanly — the turn made no tool call and changed
        # no extraction record, so nothing is lost but the model's own words
        # about it, which the event log still holds. A dangling turn EARLIER in
        # the log is not this: it would mean the run continued past a turn with
        # no user side, which nothing can produce, so it is left in place for
        # `_replay_assistant_content` to meet on its own terms.
        if order:
            last = turns[order[-1]]
            if last["assistant"] is not None and not last["user"]:
                order = order[:-1]

        messages = []
        for tid in order:
            t = turns[tid]
            assistant = self._replay_assistant_content(tid, t)
            if assistant:
                messages.append({"role": "assistant", "content": assistant})
            if t["user"]:
                messages.append({"role": "user", "content": t["user"]})
        return messages

    def _replay_assistant_content(self, turn_id, t):
        """Return the assistant message content for one replayed turn.

        The byte-identical source is the turn's `assistant_message` event,
        the verbatim ordered content sent live. Every tool-calling turn logs
        one (the terminal repeated-failure stall included), and a turn's
        events are appended sequentially with the assistant_message last, so
        a surviving assistant_message implies every one of its preceding
        tool_call events survived: replay can never build a tool_use whose
        tool_result was lost to a torn tail while the assistant_message is
        present.

        A turn with events but no assistant_message can therefore only be a
        crash artefact (a hard kill between the turn's appends, or a torn
        assistant_message line dropped by the torn-tail repair). There is no
        faithful way to know what the model emitted in that turn, so per the
        no-silent-fallback rule the resume is refused rather than rebuilt
        into a divergent conversation.

        A turn carrying an `assistant_text` event but no `assistant_message`
        gets the same refusal for a different reason. Both are written on
        every turn and `assistant_message` is always appended FIRST, so a
        truncated tail that loses the message loses the text with it: this
        combination is not one a torn log can produce. It means the file was
        edited, or written by something that is not this engine, and its block
        order cannot be trusted to rebuild.
        """
        if t["assistant"] is not None:
            return t["assistant"]
        if t["has_text_event"]:
            raise SessionError(
                f"Turn {turn_id} in {self.tool_calls_path} logged assistant "
                "text but no assistant_message event. This engine always "
                "writes the assistant_message first, so no truncation can "
                "leave the text without it: the log has been edited or was "
                "not written by meltiro, and the turn cannot be replayed "
                "faithfully. Start a fresh session."
            )
        raise SessionError(
            f"Turn {turn_id} in {self.tool_calls_path} has turn events but "
            "no assistant_message event. Every completed turn logs one, so "
            "this is a crash artefact: the run was killed between logging "
            "the turn's tool calls and its assistant message, or the "
            "assistant_message line was torn and dropped on a prior resume. "
            "The conversation tail cannot be rebuilt verbatim; the "
            "extraction record itself is intact. Start a fresh session."
        )

    # ----------------------------------------------------------------------
    # Status transitions
    # ----------------------------------------------------------------------

    def set_phase(self, phase):
        if phase not in self.PHASES:
            raise SessionError(f"Unknown phase: {phase}")
        self.meta["current_phase"] = phase
        self.write_meta()

    def increment_tool_call_count(self):
        self.meta["tool_call_count"] = \
            int(self.meta.get("tool_call_count", 0)) + 1
        self.write_meta()

    def record_checker_calls(self, count):
        """Add `count` to the session's tally of CHECKS run, and flush.

        One per field checked, not one per provider call: a check whose first
        reply recorded no verdict is re-asked, and both asks belong to the one
        check this counts. The wire log is where the calls themselves are
        counted.

        Called once per tool call whose fan-out ran, so the flush cadence
        matches the tool-call cadence: a crash loses at most the current call's
        tally rather than the whole run's.
        """
        self.meta["checker_calls_run"] = \
            int(self.meta.get("checker_calls_run", 0)) + int(count)
        self.write_meta()

    def finalise(self, status):
        if status not in TERMINAL_STATUSES:
            raise SessionError(f"Unknown terminal status: {status}")
        self.meta["status"] = status
        self.meta["current_phase"] = "done"
        self.append_event({"event": "terminate", "status": status,
                           "ts": _utc_now_iso()})
        self.write_meta()
