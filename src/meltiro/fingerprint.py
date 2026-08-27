"""Fingerprints for the agentic pipeline.

This module keeps the contracts and exact preimages.

Stage fingerprints, one per LLM stage, so an edit to any stage's inputs
moves that stage's fingerprint and only that one:

  - `config_fingerprint(...)` -> `config_fp`: the extractor.
  - `checker_config_fingerprint(...)` -> `checker_fp`: the per-field checker.
  - `review_config_fingerprint(...)` -> `review_fp`: the final reviewer.

Each takes as its first component direktoro's provider-call identity block
(model id, provider, base_url, Route, and the RESOLVED decoding params keyed
under the wire's own parameter name), built by the caller with
`direktoro.call_identity_fields(...)` and serialised with
`direktoro.canonical_json`. The block is always handed in as a string: this
module imports no direktoro symbol, so `import meltiro` keeps working with
direktoro absent. Decoding params reach each stage fingerprint once, through
the block — `structure_hash` must never carry them too.

Orthogonal axes recorded beside the stage fingerprints, for comparing runs:

  - `instrument_fingerprint(...)` -> `instrument_fp`: MODEL-FREE, covering the
    config author's instrument plus the engine's tool contract (`tool_set_hash`
    folds in the tool definitions, whose descriptions are engine prose).
  - `call_fingerprint(...)` -> `extractor_call_fp` / `checker_call_fp` /
    `review_call_fp`: per role, the model and how it is reached.
  - `engine_fingerprint(...)` -> `engine_fp`: meltiro's and direktoro's
    versions and source-content hashes.

`run_fingerprint(...)` -> `run_fp` folds the three stage fingerprints and
`engine_fp` into the whole-run identity. `bundle_fingerprint(...)` ->
`bundle_fp` names the paper and is folded into nothing: the pair
(`run_fp`, `bundle_fp`) is what was asked of what.

Supporting hashes: `tool_set_hash`, `field_catalogue_hash`,
`reference_lists_hash`.
"""

import hashlib
import json
from pathlib import Path

from meltiro.bundle import read_transcription


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(payload):
    """The one JSON serialisation every hash in this package is taken over.

    Keys sorted and separators tight, so a dict's insertion order and a
    formatter's whitespace reach no digest and two callers hashing the same
    content land on the same bytes. List order is left alone: everywhere this
    is used a list's order is content, and reordering one is a real edit.

    IT DIVERGES FROM `direktoro.canonical_json` IN ONE BYTE-VISIBLE WAY, and
    the two must not be substituted for each other. direktoro's passes
    `ensure_ascii=False`, so a non-ASCII character is serialised as itself;
    this one leaves the `json` default in place, so the same character is
    serialised as a `\\uXXXX` escape. The preimages therefore differ for any
    payload carrying non-ASCII text — which the reference lists, the template
    and the prompts routinely do.

    The divergence is recorded rather than removed. Every fingerprint this
    module produces is a PUBLISHED number: a run record carries `config_fp`,
    `instrument_fp` and the rest so a reader can recompute them and check that
    a reported extraction came from a stated bundle. Aligning the escaping
    would move `prompts_hash`, `tool_set_hash`, `instrument_fp` and all three
    stage fingerprints for every bundle with a non-ASCII byte in it, and every
    already-published value would stop verifying — a real cost, paid for a
    cosmetic agreement between two functions that never hash the same payload.
    direktoro's serialises ITS call-identity block, which arrives here already
    a string and is folded in verbatim; nothing crosses the boundary as a
    structure both would serialise.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def figure_hashes(figure_paths):
    """`{label: {"sha256", "byte_length"}}` over a paper bundle's cropped
    figures.

    The label is the file's stem — what the manifest declares and what the
    extractor cites as `<img>label</img>`. One recipe shared by
    `Session.capture_image_hashes` and `bundle_fingerprint`, so the digests a
    run reports and the digests it fingerprints are the same numbers.

    An unreadable path is skipped: a record of the readable crops beats none.
    """
    hashes = {}
    for path in figure_paths:
        path = Path(path)
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        hashes[path.stem] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "byte_length": len(raw),
        }
    return hashes


def tool_set_hash(tools):
    """Canonical-JSON SHA-256 of the tool definitions.

    `tools` is a tool catalogue: `tools.all_tool_definitions()`'s {role:
    [tool, ...]} mapping for the extractor and reviewer, or
    `tools.checker_tool_definitions()`'s plain list for the checker.
    `sort_keys=True` makes dict-key reordering irrelevant; each list's order
    is significant. The two catalogues are hashed separately and land in
    different fingerprints, so this says how to hash one, never which.
    """
    return _sha256(canonical_json(tools))


def field_catalogue_hash(template):
    """SHA-256 of the template's flattened field catalogue.

    An input to `checker_config_fingerprint`, so editing a field's
    checker-visible spec in extraction_template.yaml moves `checker_fp`. The
    relevant subset is hashed rather than the file, whose hash also moves on
    irrelevant comment edits.

    The hashed subset covers every field attribute that alters the checker's
    rendered per-field message (see checker_prompts.build_checker_user_message)
    or the field's validation surface: variable, description,
    extraction_instruction, field_type, allow_other, evidence, options,
    canonical_reference, required, and role.

    `role` is hashed because it decides WHICH value the checker is shown, not
    merely how a field is rendered: the `role: summary` field is the fallback
    study-identity context handed to every checker call (see
    `Orchestrator._study_identity_context`), so excluding it would let an
    edit change every checker call while `checker_fp` sat still. Excluded:
    `label` and `soft_canonicalisation` (the engine acts on neither) and a
    section's `qa:` flag (only fields are hashed, by the attribute list
    above).
    """
    from meltiro.template import iter_fields

    rows = []
    for block_key in [
        "study_fields", "record_fields",
        "initial_check_fields", "quality_check_fields",
    ]:
        for f in iter_fields(template[block_key]):
            rows.append({
                "block": block_key,
                "variable": f["variable"],
                "description": f["description"],
                "extraction_instruction": f.get("extraction_instruction"),
                "field_type": f.get("field_type"),
                "allow_other": f.get("allow_other"),
                "evidence": f.get("evidence"),
                "options": f.get("options"),
                "canonical_reference": f.get("canonical_reference"),
                "required": f.get("required"),
                "role": f.get("role"),
            })
    return _sha256(canonical_json(rows))


def reference_lists_hash(reference_lists):
    """SHA-256 of the reference lists' CONTENT (canonical names + aliases).

    Aliases are rendered into no prompt (only canonical names are, via
    render_reference_block), so an alias edit moves no prompt hash; this
    component is what carries it into all three stage fingerprints.
    `search_terms` are excluded: neither rendered nor used for
    canonicalisation.

    `canonical_json` sorts the payload's keys, so iteration order here cannot
    reach the digest. Entry order within a list and alias order within an
    entry ARE content and are preserved: reordering either is a real edit and
    must move the hash.
    """
    from meltiro.reference_lists import entry_canonical_name, entry_aliases

    payload = {}
    for name in reference_lists or {}:
        payload[name] = [
            {"name": entry_canonical_name(e), "aliases": list(entry_aliases(e))}
            for e in (reference_lists[name] or [])
        ]
    return _sha256(canonical_json(payload))


def checker_config_fingerprint(call_identity,
                               system_prompt_text,
                               user_prompt_template_text,
                               tool_set_hash="none",
                               structure_hash="default",
                               field_catalogue_hash_str="none",
                               reference_hash="none",
                               checker_context_fields=None,
                               checker_context_chars=0):
    """Fingerprint a checker config.

    The two prompt components are the CONFIG's half of each prompt the checker
    sends: what the bundle wrote for the checker, and its overrides of the
    checker's engine prompts (`checker_prompts.build_checker_config_text` and
    `checker_user_config_text`). Editing either, or changing the
    model/decoding/catalogue, produces a new fingerprint, so rerunning under a
    new prompt won't overwrite old verdicts. `call_identity` is the checker
    model's
    provider-call identity block (see the module docstring), so a resume that
    switches the checker's provider is refused. `structure_hash` folds in
    only the image-capability toggle. `reference_hash` carries the
    reference-list content, so an alias edit, invisible to the rendered
    prompts, still moves the fingerprint.

    `checker_context_fields` is the template's ordered list of record fields
    rendered into the `{identity_context}` slot of every per-record checker
    call (see checker_prompts.build_record_context); a template-level list,
    so it rides here as its own ordered component rather than inside
    `field_catalogue_hash`.

    `tool_set_hash` covers the checker's own tool catalogue — the schema its
    verdict must fit. It is the checker's alone, hashed apart from the
    extractor's and reviewer's catalogues (see `meltiro.tools`), so editing
    the verdict schema moves this fingerprint and no other, and editing the
    extraction template's tools moves theirs and not this one.

    `checker_context_chars` is how many characters of surrounding PAPER text
    the checker is shown on each side of a matched quote: it changes what the
    checker is asked, so it is config identity, while the text it selects is
    the run's input and stays out. It is a plain integer here — 0 is a width,
    never a stand-in for an absent checker, because this fingerprint is
    computed only for a run whose checker RUNS (an off checker records a null
    `checker_fp`). `instrument_fingerprint`, computed for every run, is the
    one that must tell a zero-width window from no window at all.
    """
    sp = _sha256(system_prompt_text)
    up = _sha256(user_prompt_template_text)
    # Ordered, so a reordering (a genuine edit to the label) moves the hash.
    sig = json.dumps(list(checker_context_fields or []), separators=(",", ":"))
    key = (f"{call_identity}|{sp}|{up}|{tool_set_hash}"
           f"|{structure_hash}|{field_catalogue_hash_str}"
           f"|{reference_hash}|{sig}|ctx{int(checker_context_chars)}")
    return f"checker_fp:{_sha256(key)}"


def review_config_fingerprint(call_identity,
                              system_prompt_text,
                              tool_set_hash="none",
                              structure_hash="default",
                              reference_hash="none"):
    """Fingerprint the final-review stage config.

    The review model and review prompt appear in no other fingerprint, so
    editing the model, prompt, tool set, or review decoding params moves this
    one alone. `call_identity` is the review model's provider-call identity
    block (see the module docstring). `system_prompt_text` is the CONFIG's
    half of the review system prompt (`prompt_builder.build_config_prompt_text`):
    reference placeholders already substituted, so two papers under one config
    share the fingerprint, and the engine's own reviewer prompts outside it.
    `reference_hash` is here because the reviewer drives the same
    `ToolDispatcher` the extractor does: an alias edit — rendered into no
    prompt — changes which values its tool calls may write.
    """
    sp = _sha256(system_prompt_text)
    key = (f"{call_identity}|{sp}|{tool_set_hash}|{structure_hash}"
           f"|{reference_hash}")
    return f"review_fp:{_sha256(key)}"


def structure_hash(max_checks_per_field, final_review=True,
                   supports_images=True, check_reviewer_edits=False):
    """Compact stringification of a stage's STRUCTURE toggles — what the
    pipeline does around the call, not how the call is dialled.

    Decoding params do NOT ride here: they reach each stage fingerprint once,
    through the call-identity block, and carrying them here too would
    double-count them. The tool-call cap (`max_tool_calls`) is deliberately
    not hashed either: an operational budget, recorded per segment in
    run.json, and hashing it would refuse the documented raise-the-cap
    resume.

    `max_checks_per_field` is the checker's on-switch (on exactly when > 0)
    and budget in one. `final_review` and `check_reviewer_edits` fold in the
    reviewer toggle and whether the checker also gates the REVIEWER's tool
    calls; setting `check_reviewer_edits` with either stage off still records
    a distinct structure. `supports_images` is the stage model's declared
    image capability — meltiro behaviour (no image parts sent, the label list
    rendered as none-available), not a wire-decoding parameter, so it stays
    here rather than in the identity block.
    """
    base = f"checks{max_checks_per_field}"
    if not final_review:
        base += "_noreview"
    if check_reviewer_edits:
        base += "_checkreview"
    if not supports_images:
        base += "_noimages"
    return base


def config_fingerprint(call_identity, prompt_hash, template_hash,
                       tool_set_hash="none", structure_hash="default",
                       reference_hash="none"):
    """Fingerprint an extraction config.

    `call_identity` is the extractor model's provider-call identity block
    (see the module docstring): the same prompts/tools on two providers get
    two fingerprints, so their session dirs never collide and a resume that
    changes the extractor's provider/endpoint/route is refused.

    prompt_hash and template_hash are passed through unchanged (already
    SHA-256 digests in run_log.json). tool_set_hash and structure_hash
    default to sentinel values so a single-shot run (no tools, default
    structure) gets a stable fingerprint when backfilled. `reference_hash` is
    what moves config_fp on an alias edit: reference names already ride in
    prompt_hash via the rendered prompt, aliases do not.
    """
    key = (f"{call_identity}|{prompt_hash}|{template_hash}"
           f"|{tool_set_hash}|{structure_hash}|{reference_hash}")
    return f"config_fp:{_sha256(key)}"


# Stable sentinel standing in for something that did not exist for this run:
# a stage that did not run (in the `run_fingerprint` preimage), a checker
# window that was never opened (in `instrument_fingerprint`), and a paper
# bundle that supplies no figures (in `bundle_fingerprint`). It matches the
# `="none"` sentinel the stage fingerprints above use for an absent component,
# so the whole module speaks one absence word. A present stage fingerprint is
# always prefixed (`checker_fp:` / `review_fp:`), so this bare token can never
# collide with a real one, and its fixed position in the preimage (checker
# second, review third) keeps the two absences distinct even though they share
# the token. Where it stands in for a number, it can never collide with one
# either: an integer never renders as `none`. Where it stands in for a
# canonical-JSON payload it cannot collide either: that payload is always a
# bracketed list.
ABSENT_STAGE = "none"


def run_fingerprint(config_fp, checker_fp, review_fp, engine_fp):
    """Fingerprint a whole run's producing configuration: the three stage
    fingerprints and the engine identity folded into one.

    `config_fp` alone identifies only the EXTRACTOR stage; `run_fp` keeps
    apart runs that share an extractor config but assign different
    checker/reviewer models or prompts. Downstream builds one producer
    string (`llm:<run_fp>`) per distinct (extractor, checker, reviewer)
    triple. Folding `engine_fp` in means `run_fp` equality does not survive
    a release of either package — the deliberate cost the separately
    recorded axes pay for.

    A `run_fp` is computed once, at session creation, and never recomputed:
    a session can outlive the tree it started against (pause, upgrade,
    resume), so what stands behind `run_fp` is the engine at session start;
    per-segment truth lives in the `resumed` events.

    Preimage (exact): the three stage fingerprints and the engine fingerprint
    joined by `|`, in the fixed order extractor, checker, reviewer, engine:

        run_fp:SHA256( config_fp | checker_fp_or_sentinel
                       | review_fp_or_sentinel | engine_fp )

    Each part is the fingerprint's full self-prefixed string (`config_fp:...`,
    `checker_fp:...`, `review_fp:...`, `engine_fp:...`), the same value written
    to run.json, hashed verbatim. `config_fp` and `engine_fp` are always
    present. A DISABLED stage records a null fingerprint (the orchestrator
    passes `None`), and that `None` is replaced by the `ABSENT_STAGE`
    sentinel BEFORE hashing, never left to Python's `str(None)`. The
    sentinel is distinct from every real fingerprint (those are prefixed)
    and sits in a fixed position, so every ablation is well-defined and
    mutually distinct:

        extractor + checker + reviewer -> config_fp:X|checker_fp:Y|review_fp:Z|engine_fp:E
        extractor + checker            -> config_fp:X|checker_fp:Y|none|engine_fp:E
        extractor + reviewer           -> config_fp:X|none|review_fp:Z|engine_fp:E
        extractor only                 -> config_fp:X|none|none|engine_fp:E

    The result is self-prefixed `run_fp:<sha256hex>`, a full digest with no
    truncation anywhere. The six-char shortening in a session directory name
    is not of this value: it is the first six characters of `config_fp`
    (`Session.create`), which is a different fingerprint answering a different
    question, so a directory name is no abbreviation of the run's identity.
    """
    checker = checker_fp if checker_fp is not None else ABSENT_STAGE
    review = review_fp if review_fp is not None else ABSENT_STAGE
    key = f"{config_fp}|{checker}|{review}|{engine_fp}"
    return f"run_fp:{_sha256(key)}"


def instrument_structure_hash(max_checks_per_field, final_review=True,
                              check_reviewer_edits=False):
    """The USER-AUTHORED subset of `structure_hash`.

    `structure_hash` also folds in `supports_images`, a property of the model
    a role is assigned; the instrument axis must stay model-free, so this
    variant covers only the three `pipeline.yaml` toggles. Same string shape
    as `structure_hash` minus the `_noimages` suffix, so the two read the
    same way side by side in a run record.
    """
    base = f"checks{max_checks_per_field}"
    if not final_review:
        base += "_noreview"
    if check_reviewer_edits:
        base += "_checkreview"
    return base


def instrument_fingerprint(prompts_hash, template_hash,
                           tool_set_hash="none",
                           structure_hash="default",
                           reference_hash="none",
                           checker_context_chars=None,
                           checker_context_fields=None):
    """Fingerprint the INSTRUMENT: everything the config author wrote, plus the
    engine's tool contract.

    One of the three orthogonal axes, and it is MODEL-FREE: every model,
    provider, endpoint, route and decoding parameter is deliberately absent
    (`call_fingerprint`). It is NOT engine-free, and the exception is
    `tool_set_hash`: the tool definitions carry the engine's own descriptions
    of what each tool does, so rewording one moves this axis. That is the
    right behaviour — an instrument is the question put to a model, and a tool
    description is part of the question — but it means `instrument_fp` is not
    a pure config identity, and two runs of one bundle under engine versions
    whose tool prose differs record different values. The engine's OTHER prose
    — the framing around the bundle's prompts, and every engine prompt the
    bundle leaves as the engine wrote it — is not here and rides in
    `engine_fingerprint`.

    `prompts_hash` covers the three prompt files with `{include:NAME}`
    partials expanded — the SOURCE text — plus every engine prompt the bundle
    overrides, that being text the config author wrote; reference-list
    content, aliases included, rides in `reference_hash`. `structure_hash`
    here is `instrument_structure_hash`, not `structure_hash`.

    `checker_context_chars` is the checker's quote-context width, or None
    when there is no checker and so no window at all: None renders as the
    module's `ABSENT_STAGE` word, a width of zero as the number zero, which
    an integer can never be confused with. The distinction must not lean on
    `structure_hash` beside it also carrying `checks0` — a component only
    distinct in company is one refactor away from not being distinct at all.

    Computable from a config directory alone
    (`config_bundle.load_config_bundle`). A run records its own EFFECTIVE
    value, and when a CLI flag overrode `pipeline.yaml` TWO components must
    follow the override: `structure_hash` carries the effective toggles, and
    `prompts_hash` must be recomputed against the effective
    `{include_if:...}` predicates rather than read from the bundle's
    load-time value (see `Instrument.fingerprint` and
    `ConfigBundle.prompts_hash_for`).
    """
    sig = json.dumps(list(checker_context_fields or []), separators=(",", ":"))
    ctx = (ABSENT_STAGE if checker_context_chars is None
           else int(checker_context_chars))
    key = (f"{prompts_hash}|{template_hash}|{tool_set_hash}"
           f"|{structure_hash}|{reference_hash}"
           f"|ctx{ctx}|{sig}")
    return f"instrument_fp:{_sha256(key)}"


def call_fingerprint(call_identity):
    """Fingerprint one role's PROVIDER CALL: the model and how it is reached.

    A named hash of direktoro's call-identity block, which already rides
    inside each stage fingerprint blended with that stage's content; hashed
    on its own it makes "same instrument, different API" a diff on one
    recorded field. Recorded per role (`extractor_call_fp`,
    `checker_call_fp`, `review_call_fp`) because a config may swap one role's
    model and leave another's alone. A disabled stage records None, matching
    its null stage fingerprint.
    """
    return f"call_fp:{_sha256(call_identity)}"


def engine_fingerprint(meltiro_version, meltiro_source_hash,
                       direktoro_version=None, direktoro_source_hash=None):
    """Fingerprint the ENGINE: the code that asked the question.

    Two packages, each identified by the version it declares and a content
    hash of its own source files (`run_log.source_hash` for meltiro,
    `direktoro.source_hash` for direktoro): the version names a release, the
    source hash names the bytes, so two runs share an `engine_fp` when and
    only when they ran the same source under the same declared versions.
    direktoro is half the engine — it builds the identity block leading every
    stage fingerprint and resolves what is actually sent.

    Both direktoro components are None when direktoro is not installed and
    each folds in as the fixed `nodirektoro` token, so an engine with no
    direktoro is distinct from every engine that has one.

    Preimage (exact), joined by `|` in this fixed order:

        engine_fp:SHA256( meltiro_version | meltiro_src
                          | direktoro_version | direktoro_src )

    The git commit and the working tree's state are recorded with every run
    but are not preimage components: they identify WHERE THE CODE CAME FROM,
    which for an installed copy is a commit with no working tree beside it;
    the source hash identifies the code itself, wherever it sits.
    """
    direktoro_ver = (direktoro_version if direktoro_version is not None
                     else "nodirektoro")
    direktoro_src = (direktoro_source_hash if direktoro_source_hash is not None
                     else "nodirektoro")
    key = (f"{meltiro_version}|{meltiro_source_hash}"
           f"|{direktoro_ver}|{direktoro_src}")
    return f"engine_fp:{_sha256(key)}"


def bundle_fingerprint(bundle):
    """Fingerprint the PAPER: which input this run was given.

    Returns the six self-prefixed values a run records, as a dict:

      - `text_fp`: SHA-256 of `text.md`'s bytes, the whole text the models
        were shown.
      - `figures_fp`: SHA-256 over the bundle's cropped figures as sorted
        `(label, sha256-of-bytes)` pairs, so a re-crop, a swapped image or a
        renamed label all move it. A bundle with no figures folds in the
        module's `ABSENT_STAGE` sentinel, making "this paper supplies no
        crops" a hashed fact rather than the digest of an empty payload.
      - `manifest_fp`: SHA-256 of `manifest.json`'s canonical JSON, so an
        edited title, id, summary or exhibit caption moves it while a
        reformat or a key reordering does not.
      - `tables_fp`: SHA-256 over the bundle's table transcriptions as sorted
        `(label, sha256-of-bytes)` pairs, on exactly `figures_fp`'s terms, so
        a re-transcribed cell, a transcription added to an exhibit that had
        none, or one withdrawn all move it. A bundle transcribing nothing
        folds in `ABSENT_STAGE`, making "this paper supplies no
        transcriptions" a hashed fact rather than the digest of an empty
        payload — and telling that apart from a bundle whose transcriptions
        happen to hash to nothing.
      - `supplements_fp`: SHA-256 over the supplementary material as sorted
        `(name, title, text-digest, crops, transcriptions)` entries, so a
        supplement arriving, a supplement withdrawn, a re-crop inside one, a
        re-transcription inside one, or an edit to its prose or its printed
        title all move it. A bundle carrying none folds in `ABSENT_STAGE`.
        It is a SEPARATE component rather than a contribution to the three
        above, and that is the whole point of the shape: `text_fp` and
        `manifest_fp` stay the article's, byte for byte, so a consumer that
        identifies a paper by them — the screening side does — is untouched
        by supplementary material landing later, while a consumer that reads
        the whole bundle sees the addition in `bundle_fp`.
      - `bundle_fp`: SHA-256 over the five above, joined by `|` in that fixed
        order, each hashed verbatim as its full self-prefixed string.

    This is folded into NOTHING — not `config_fp`, not `instrument_fp`, not
    `run_fp`. The paper is the run's INPUT; the pair (`run_fp`, `bundle_fp`)
    is the whole of what was asked of what.

    Computed at session start, from the loaded bundle, and written to
    `run.json` and the run-log row (see `Session.capture_bundle_fingerprint`).
    """
    root = Path(bundle.root)
    text_fp = "text_fp:" + hashlib.sha256(
        (root / "text.md").read_bytes()).hexdigest()

    digests = figure_hashes(bundle.figures.values())
    pairs = sorted((label, digests[label]["sha256"]) for label in digests)
    figures_fp = "figures_fp:" + _sha256(
        canonical_json(pairs) if pairs else ABSENT_STAGE)

    # The transcriptions, on `figures_fp`'s terms with one difference: a
    # transcription is digested as the TEXT a role is shown, through the
    # reader the message is built with, where a crop is digested as its bytes.
    # A crop's bytes are the crop; a transcription's file has surrounding
    # whitespace that no role can observe, so hashing it would move the
    # paper's identity — and refuse a resume — for a change to something
    # nobody read.
    table_pairs = _transcription_digests(bundle.tables)
    tables_fp = "tables_fp:" + _sha256(
        canonical_json(table_pairs) if table_pairs else ABSENT_STAGE)

    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8"))
    manifest_fp = "manifest_fp:" + _sha256(canonical_json(manifest))

    supplements_fp = "supplements_fp:" + _sha256(
        canonical_json(_supplement_payload(bundle))
        if bundle.supplements else ABSENT_STAGE)

    bundle_fp = "bundle_fp:" + _sha256(
        f"{text_fp}|{figures_fp}|{manifest_fp}|{tables_fp}"
        f"|{supplements_fp}")
    return {
        "text_fp": text_fp,
        "figures_fp": figures_fp,
        "manifest_fp": manifest_fp,
        "tables_fp": tables_fp,
        "supplements_fp": supplements_fp,
        "bundle_fp": bundle_fp,
    }


def _supplement_payload(bundle):
    """The hashable description of a bundle's supplementary material.

    One entry per supplement, in name order, carrying everything a run is
    shown out of it: the name and the printed title, a digest of the prose
    (a digest rather than the prose itself, so the preimage stays a fixed
    size whatever a supplement runs to), each exhibit's caption and printed
    footnote, and its crops and transcriptions as `figures_fp`'s own
    (label, digest) pairs.

    `None` for a supplement that prints no prose, which is a different fact
    from an empty one and hashes differently from it.

    The captions and footnotes are here because `supplements.json` is hashed
    NOWHERE ELSE. The article's are covered by `manifest_fp`, which digests
    `manifest.json` whole, and a supplement's declaration has no such
    wholesale cover — so free prose that every role is shown, and that says
    what an exhibit reports, would otherwise move no axis at all. A caption
    edited from "median" to "mean" tells three roles the exhibit reports
    something the crop does not, and this is the axis that has to notice.
    """
    entries = []
    for name in sorted(bundle.supplements):
        supplement = bundle.supplements[name]
        entries.append({
            "name": name,
            "title": supplement.title,
            "text": (_sha256(supplement.text)
                     if supplement.text is not None else None),
            "exhibits": sorted(supplement.exhibits.items()),
            "exhibit_notes": sorted(supplement.exhibit_notes.items()),
            "figures": _label_digests(supplement.figures),
            "tables": _transcription_digests(supplement.tables),
        })
    return entries


def _label_digests(paths):
    """Sorted `(label, sha256)` pairs for one map of label -> path."""
    digests = figure_hashes(paths.values())
    return sorted((label, digests[label]["sha256"]) for label in digests)


def _transcription_digests(paths):
    """Sorted `(label, sha256-of-the-text)` pairs for one map of
    label -> transcription path."""
    return sorted(
        (label, _sha256(read_transcription(path)))
        for label, path in paths.items())
