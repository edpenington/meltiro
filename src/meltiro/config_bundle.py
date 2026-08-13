"""Config bundle: the review-specific input contract.

Everything that makes the pipeline specific to one systematic review — field
schema, prompts, reference lists, loop caps — lives in a config bundle
directory; the pipeline code itself is generic. No worked example ships with
the package: `tests/fixtures/config_synthetic` is a synthetic test fixture,
not guidance. The layout and rules below are the specification.

Required layout:

    my-config/
      pipeline.yaml
      extraction_template.yaml
      reference/               (optional)  one *.yaml per reference list
        <list-name>.yaml
      prompts/
        extractor_system.md
        review_system.md
        checker_system.md
        partials/              (optional)  one *.md per {include:NAME} block
          <name>.md
          meltiro/             (optional)  overrides of engine sections
            <section>.md

`load_config_bundle(path)` returns a frozen `ConfigBundle` exposing every
file path, the parsed `pipeline.yaml` mapping, and the loaded reference
lists. Validation is loud and happens at load, before any API spend: a
`ConfigBundleError` lists EVERY missing required file; every
`canonical_reference:` field and `{reference:NAME}` placeholder must resolve
to a loaded list (`reference/` itself is optional); every `{include:NAME}`
must resolve to a `prompts/partials/NAME.md`, with no nesting; every file in
`prompts/partials/meltiro/` must be named for a section the engine ships;
`pipeline.yaml` keys are checked against `KNOWN_PIPELINE_KEYS`; tool-call cap
placeholders are banned from every prompt and override; and the checker's
prompt and overrides may cite only the placeholders the engine substitutes
into them.

Each role's system message is the engine's spine for that role followed by
the bundle's prompt file, appended. A prompt file therefore supplies the
review's own text and nothing about the engine; to supply different wording
for a section, a bundle ships `prompts/partials/meltiro/NAME.md`, and an
empty one leaves the section out (see `meltiro.prompt_partials`).
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from meltiro.errors import ConfigBundleError
from meltiro.fingerprint import (
    canonical_json,
    instrument_fingerprint,
    instrument_structure_hash,
    reference_lists_hash as _reference_lists_hash,
    tool_set_hash as _tool_set_hash,
)
from meltiro.prompt_partials import (
    CHECKER_SYSTEM,
    CHECKER_USER,
    ENGINE_NAMESPACE,
    EXPAND_ALL_BRANCHES,
    all_engine_override_pairs,
    engine_citation_message,
    engine_cited_names,
    engine_override_entries,
    engine_override_path,
    engine_section_names,
    included_names,
    spine_sections,
    stage_predicates,
    substitute_include_placeholders,
)
from meltiro.reference_lists import load_reference_lists, referenced_names
from meltiro.template import iter_fields, load_template
from meltiro.yaml_strict import strict_load


# The pipeline.yaml keys the CLI actually reads (see meltiro/cli.py
# `_build_orchestrator`). An unknown key is rejected loudly: silently
# ignoring a typo would change a decoding parameter without any trace in the
# provenance fingerprint.
KNOWN_PIPELINE_KEYS = frozenset({
    # Read by the CLI today.
    "extractor_model",
    "review_model",
    "checker_model",
    "checker_concurrency",
    "checker_context_chars",
    "checker_max_tokens",
    "extractor_max_tokens",
    "review_max_tokens",
    "max_tool_calls",
    "max_review_tool_calls",
    "max_checks_per_field",
    # One decoding block per role: an opaque mapping of decoding parameter
    # names to values, handed whole to direktoro's `split_decoding_config`,
    # which is what knows which key inside is a sampling control and which is
    # a thinking field. meltiro reads none of them. Optional per role; a role
    # with no block sends nothing and takes the model's own defaults. Decoding
    # params ride in their own role's stage fingerprint, never the instrument.
    "extractor_decoding",
    "review_decoding",
    "checker_decoding",
    # Pipeline-structure toggles. `final_review: false` disables the reviewer
    # stage; the checker is disabled by `max_checks_per_field: 0` (no separate
    # key). `check_reviewer_edits: true` extends the checker to the reviewer's
    # own tool calls. All three move the config fingerprint.
    "final_review",
    "check_reviewer_edits",
    # Optional per-role USD rate cards (`extractor`, `checker`, `review`);
    # a role the block leaves out takes direktoro's price table. Validated
    # where the CLI reads it; rationale in `meltiro.rates`. Rates reach no
    # fingerprint.
    "rates",
})

# The TOTAL number of checker calls one field may receive across the whole
# session: a check when first written, plus one re-check after a revision.
# 1 checks a field once and never re-checks; 0 disables the checker. Config
# overrides this fallback.
#
# Lives here, not in `meltiro.orchestrator`: this module must stay loadable
# without direktoro (a `--no-deps` wheel consumer reads bundles too), and the
# orchestrator imports direktoro at module scope. The orchestrator re-exports
# the name.
DEFAULT_MAX_CHECKS_PER_FIELD = 2


@dataclass(frozen=True)
class ConfigBundle:
    """A validated, loaded config bundle. Immutable.

    The four fingerprint fields are the bundle's content-only identity,
    computed at load so a consumer can pin a bundle without re-deriving the
    recipe. `template_hash` and `reference_lists_hash` are exactly the values
    the orchestrator folds into its run-time fingerprints; pinning that pair
    alone decides whether a stored value is still legal. `prompts_hash`
    covers the three prompt files with their partials expanded, plus every
    engine section this bundle overrides (see `_compute_prompts_hash`).
    `instrument_fp` is the MODEL-FREE composite of everything the config
    author wrote plus the engine's tool contract (the tool definitions carry
    the engine's own descriptions, so `instrument_fp` is not engine-free — see
    `fingerprint.instrument_fingerprint`); the run-time `config_fp`
    additionally folds in the extractor model and decoding params and so is
    not derivable here.
    """

    root: Path
    template_path: Path
    reference_dir: Path | None
    reference_lists: dict  # name -> list of entries (empty when no reference/)
    extractor_system_path: Path
    review_system_path: Path
    checker_system_path: Path
    partials_dir: Path
    pipeline: dict
    template_hash: str
    reference_lists_hash: str
    prompts_hash: str
    instrument_fp: str

    @property
    def prompt_paths(self):
        """The three prompt files, in the fixed order the hash is keyed by.

        Named once here so the load-time hash and any later recomputation
        cover the same set.
        """
        return [self.extractor_system_path, self.review_system_path,
                self.checker_system_path]

    def prompts_hash_for(self, predicates):
        """The prompt content hash as it renders under `predicates`.

        `prompts_hash` above is this same hash under `pipeline.yaml`'s own
        toggles. A run can differ: `--max-checks-per-field 0` and
        `--no-final-review` change which `{include_if:...}` branches reach a
        model and which spine sections compose, so a run computes this against
        the toggles it actually honoured (see `Instrument.fingerprint`); the
        two agree exactly when no flag overrode the file.
        """
        return _compute_prompts_hash(
            self.prompt_paths, self.partials_dir, predicates)


def load_config_bundle(path):
    """Load and return a `ConfigBundle`.

    Raises `ConfigBundleError`, and only that, for every defect in the bundle:
    a missing required file, a `pipeline.yaml` that does not parse or does not
    parse to a mapping, a malformed reference list, an extraction template that
    does not parse or violates the template model, a `canonical_reference` no
    reference list provides, an unresolvable prompt placeholder. That one type
    is the whole contract every caller is written against — the CLI catches it
    and exits 1 with the message on stderr — so a YAML syntax error or a
    template-model violation is carried inside it, underlying message and all,
    rather than reaching a caller as a traceback.
    """
    root = Path(path)
    problems = []
    if not root.exists():
        raise ConfigBundleError(
            [f"config bundle directory does not exist: {root}"], path=root)
    if not root.is_dir():
        raise ConfigBundleError(
            [f"config bundle path is not a directory: {root}"], path=root)

    template_path = root / "extraction_template.yaml"
    pipeline_path = root / "pipeline.yaml"
    reference_dir = root / "reference"
    prompts_dir = root / "prompts"
    partials_dir = prompts_dir / "partials"  # optional; only read on {include:}
    extractor_system_path = prompts_dir / "extractor_system.md"
    review_system_path = prompts_dir / "review_system.md"
    checker_system_path = prompts_dir / "checker_system.md"

    required = [
        pipeline_path,
        template_path,
        extractor_system_path,
        review_system_path,
        checker_system_path,
    ]
    for p in required:
        if not p.is_file():
            problems.append(f"missing required file: {p.relative_to(root)}")

    # The per-field checker message is rendered from the engine's own
    # `checker_user` section, so a file of that name in the bundle would be
    # read by nothing. Refused rather than ignored, because a bundle whose
    # author believes the checker reads it would run with wording nobody sent.
    stray_checker_user = prompts_dir / "checker_user_template.md"
    if stray_checker_user.is_file():
        problems.append(
            f"{stray_checker_user.relative_to(root)} is not part of a config "
            f"bundle: the engine renders the per-field checker message from "
            f"its own `{CHECKER_USER}` section. Delete the file, or move its "
            f"wording to prompts/partials/{ENGINE_NAMESPACE}/{CHECKER_USER}.md "
            f"to override that section.")

    if problems:
        raise ConfigBundleError(problems, path=root)

    # `load_config_bundle` raises ConfigBundleError and nothing else (see the
    # docstring, and every caller, which catches exactly that and exits 1). A
    # YAML syntax error is a defect in the bundle like any other, so pyyaml's
    # own diagnostic — which carries the file, line and column of the offending
    # token, the most useful part of the message — is carried through inside
    # the contract rather than escaping as a traceback.
    try:
        with open(pipeline_path, "r", encoding="utf-8") as f:
            pipeline = strict_load(f)
    except yaml.YAMLError as e:
        raise ConfigBundleError(
            [f"pipeline.yaml does not parse: {e}"], path=root) from e
    except OSError as e:
        raise ConfigBundleError(
            [f"pipeline.yaml could not be read: {e}"], path=root) from e
    if pipeline is None:
        pipeline = {}
    if not isinstance(pipeline, dict):
        raise ConfigBundleError(
            [f"pipeline.yaml must parse to a mapping, got "
             f"{type(pipeline).__name__}"],
            path=root,
        )

    # Reject unknown pipeline.yaml keys (see KNOWN_PIPELINE_KEYS).
    unknown_keys = sorted(k for k in pipeline if k not in KNOWN_PIPELINE_KEYS)
    if unknown_keys:
        raise ConfigBundleError(
            [f"pipeline.yaml has unknown key(s): "
             f"{', '.join(repr(k) for k in unknown_keys)}. Known keys: "
             f"{', '.join(sorted(KNOWN_PIPELINE_KEYS))}. Remove or correct "
             f"the offending key(s); an unrecognised key is silently ignored "
             f"otherwise and would not move the config fingerprint."],
            path=root,
        )

    # reference/ is optional. When present, load every *.yaml list.
    have_reference_dir = reference_dir.is_dir()
    reference_lists = (
        load_reference_lists(reference_dir) if have_reference_dir else {})

    # Loaded once: needed for the reference cross-validation and the content
    # fingerprint below. `load_template` raises ValueError for a template that
    # violates the model, a yaml.YAMLError for malformed YAML (a duplicate key
    # included, via strict_load) and OSError for a read fault. All three are
    # defects in this bundle, so they arrive at the caller as the one error
    # this function documents, carrying the underlying message intact.
    try:
        template = load_template(template_path)
    except ConfigBundleError:
        raise
    except (ValueError, yaml.YAMLError, OSError) as e:
        raise ConfigBundleError(
            [f"extraction_template.yaml is not loadable: {e}"],
            path=root) from e

    _cross_validate_references(template, reference_lists, root)

    prompt_paths = [extractor_system_path, review_system_path,
                    checker_system_path]

    # The override directory first: a file sitting there under a name no
    # section has overrides nothing, and every check below would pass a bundle
    # whose author believes it does.
    _validate_engine_overrides(partials_dir, root)

    # Partials next: the later checks expand includes, so every include must
    # already resolve without nesting.
    _validate_prompt_partials(prompt_paths, partials_dir, root)

    _validate_no_cap_placeholders(prompt_paths, partials_dir, root)

    _validate_prompt_references(
        prompt_paths, partials_dir, reference_lists, root)

    # After the cap guard, so a cited cap gets its targeted message rather
    # than the generic unknown-placeholder one.
    _validate_checker_placeholders(checker_system_path, partials_dir, root)

    # The bundle's own toggles, as pipeline.yaml states them. They decide
    # which `{include_if:...}` branch every render below takes, which spine
    # sections compose, and which stages run at all.
    predicates = stage_predicates(
        pipeline.get("max_checks_per_field", DEFAULT_MAX_CHECKS_PER_FIELD),
        pipeline.get("final_review", True))

    # Content fingerprint, computed here so the identity rides with the
    # loaded bundle.
    template_hash = template["template_hash"]
    ref_hash = _reference_lists_hash(reference_lists)
    prompts_hash = _compute_prompts_hash(
        prompt_paths, partials_dir, predicates)
    instrument_fp = _content_instrument_fingerprint(
        template, pipeline, prompts_hash, template_hash, ref_hash)

    return ConfigBundle(
        root=root,
        template_path=template_path,
        reference_dir=reference_dir if have_reference_dir else None,
        reference_lists=reference_lists,
        extractor_system_path=extractor_system_path,
        review_system_path=review_system_path,
        checker_system_path=checker_system_path,
        partials_dir=partials_dir,
        pipeline=pipeline,
        template_hash=template_hash,
        reference_lists_hash=ref_hash,
        prompts_hash=prompts_hash,
        instrument_fp=instrument_fp,
    )


def _compute_prompts_hash(prompt_paths, partials_dir, predicates):
    """SHA-256 over everything the config author wrote as prompt text.

    Two components. The three prompt files with `{include:NAME}` partials
    expanded, keyed by file stem: partials are expanded so a refactor that
    only moves shared text into a partial does not move the hash, while an
    edit to the text itself does. And every engine section this bundle
    OVERRIDES, as `[name, text]` pairs — including the empty overrides that
    leave a section out, because deciding what a model is not asked is as much
    a methodological choice as deciding what it is.

    `{reference:NAME}` placeholders are left literal: reference-list CONTENT
    is captured separately by `reference_lists_hash`, so folding it in here
    too would be redundant and would couple the prompts hash to reference
    edits. Canonically serialised, so a dict's insertion order reaches no
    digest.

    An un-overridden section contributes nothing. Its text is the engine's, so
    it moves `engine_fp` and no config fingerprint, and every bundle's
    `prompts_hash` holds across a release that rewords it. The consequence is
    intended: two bundles reading identically to a model, one on the engine's
    wording and one overriding it with byte-identical text, hash differently —
    one is pinned to the engine's copy and the other to its own.
    """
    payload = {"prompts": {}, "engine_overrides": all_engine_override_pairs(
        partials_dir, predicates=predicates)}
    for path in prompt_paths:
        text = path.read_text(encoding="utf-8")
        payload["prompts"][path.stem] = substitute_include_placeholders(
            text, partials_dir, predicates=predicates)
    canonical = canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _content_instrument_fingerprint(template, pipeline, prompts_hash,
                                    template_hash, reference_hash):
    """The bundle's instrument fingerprint: the MODEL-FREE identity of
    everything the config author wrote, together with the engine's tool
    contract (see `fingerprint.instrument_fingerprint` for why the tool
    definitions belong in it and why that makes it engine-dependent).

    Reuses `fingerprint.instrument_fingerprint`, the same function the
    orchestrator calls at run time, so a bundle's recorded identity and a
    run's are the same value from the same recipe.

    Structure toggles are read with the same defaults the CLI applies when a
    key is absent; the two are asserted equal in
    `tests/agentic_extraction/test_content_fingerprint.py`, since a drifted
    default would fingerprint a run that never happens. The checker's context
    width is read with the checker's on/off state, not on its own: a bundle
    that disables the checker has no window, whatever width the key beside it
    names.

    The call-identity block (model, provider, base_url, route, decoding) is
    omitted — that is what makes this the instrument rather than the run. The
    run-time `config_fp` folds the extractor model and decoding params in and
    so cannot be reproduced from a config directory.
    """
    from meltiro.tools import all_tool_definitions
    from meltiro.checker import DEFAULT_CONTEXT_CHARS

    # EVERY role's catalogue, matching what the orchestrator hashes at run
    # time; the extractor's list alone would print an instrument_fp no run
    # ever records.
    tools_hash = _tool_set_hash(all_tool_definitions(template))
    pipeline = pipeline or {}
    max_checks = int(pipeline.get(
        "max_checks_per_field", DEFAULT_MAX_CHECKS_PER_FIELD))
    if max_checks == 0:
        # Checker off: no quote-context window, so the fingerprint carries
        # None, matching `Instrument.checker_context_chars`.
        context_chars = None
    else:
        context_chars = pipeline.get("checker_context_chars")
        if context_chars is None:
            context_chars = DEFAULT_CONTEXT_CHARS
    return instrument_fingerprint(
        prompts_hash=prompts_hash,
        template_hash=template_hash,
        tool_set_hash=tools_hash,
        structure_hash=instrument_structure_hash(
            max_checks,
            final_review=bool(pipeline.get("final_review", True)),
            check_reviewer_edits=bool(
                pipeline.get("check_reviewer_edits", False)),
        ),
        reference_hash=reference_hash,
        checker_context_chars=context_chars,
        checker_context_fields=template.get("checker_context_fields"),
    )


def _cross_validate_references(template, reference_lists, root):
    """Fail loudly if the template names a canonical_reference the config
    bundle does not provide as a reference list.

    No separator scanning: list-valued reference fields are real
    `type: string_list` arrays validated element by element, so a canonical
    name may contain any character, commas and semicolons included.
    """
    named = set()
    for block_key in ("study_fields", "record_fields"):
        for f in iter_fields(template[block_key]):
            ref = f.get("canonical_reference")
            if ref:
                named.add(ref)
    missing = sorted(n for n in named if n not in reference_lists)
    if missing:
        available = sorted(reference_lists)
        raise ConfigBundleError(
            [f"template field(s) name canonical_reference '{name}' but no "
             f"reference/{name}.yaml is provided (available: {available})"
             for name in missing],
            path=root,
        )


def _validate_engine_overrides(partials_dir, root):
    """Fail loudly if `prompts/partials/meltiro/` holds anything that is not
    an override of a shipped engine section.

    An override is read from exactly one path, `<section>.md`, so a file named
    anything else — `recording_note.md`, `Recording_Notes.md`,
    `house_style.md`, `recording_notes.txt` — is inert: the section it was
    written to replace still renders the engine's own words, and the run
    behaves as though the file were not there. Enumerating the directory turns
    that silence into a load error naming the sections that exist.

    The names come from a directory LISTING and the comparison is
    case-sensitive (see `prompt_partials.engine_override_entries`), so a
    case-insensitive filesystem cannot make a bundle load on macOS and fail on
    Linux, or the reverse.
    """
    known = engine_section_names()
    expected = {f"{name}.md" for name in known}
    problems = []
    for entry in engine_override_entries(partials_dir):
        if entry in expected:
            continue
        problems.append(
            f"prompts/partials/{ENGINE_NAMESPACE}/{entry} overrides no engine "
            f"section. An override is read from '<section>.md' with the "
            f"section spelled exactly as the engine ships it, and the "
            f"sections are {list(known)}. Rename the file, or move it out of "
            f"{ENGINE_NAMESPACE}/ to make it a partial of your own.")
    if problems:
        raise ConfigBundleError(problems, path=root)


def _override_paths(partials_dir):
    """Every engine-section override this bundle ships, as `(name, path)`.

    Read from the shipped sections rather than from the directory listing, so
    the caller iterates over overrides that resolve;
    `_validate_engine_overrides` has already refused anything else in there.
    """
    return [(name, engine_override_path(name, partials_dir))
            for name in engine_section_names()
            if engine_override_path(name, partials_dir).is_file()]


def _validate_prompt_partials(prompt_paths, partials_dir, root):
    """Fail loudly if a prompt cites an `{include:NAME}` with no
    `prompts/partials/NAME.md`, cites an engine section, or a partial or an
    override nests another include.

    The render-time expansion in prompt_partials.py raises on all of these
    too, but only mid-run; this hoists them to config-load time, before any
    API spend, and collects every offender into one error. `partials/` itself
    is optional and touched only for includes actually cited.
    """
    problems = []
    checked = set()
    for path in prompt_paths:
        text = path.read_text(encoding="utf-8")
        for name in sorted(engine_cited_names(text)):
            problems.append(
                engine_citation_message(name, where=path.relative_to(root)))
        for name in sorted(included_names(text)):
            if name in checked:
                continue
            checked.add(name)
            partial_path = partials_dir / f"{name}.md"
            if not partial_path.is_file():
                problems.append(
                    f"prompt {path.relative_to(root)} cites unknown partial "
                    f"'{name}' via {{include:{name}}}; expected a file at "
                    f"prompts/partials/{name}.md. Add the partial or fix the "
                    f"placeholder.")
                continue
            content = partial_path.read_text(encoding="utf-8")
            nested = _nested_names(content)
            if nested:
                problems.append(
                    f"partial prompts/partials/{name}.md nests further "
                    f"include(s) {nested}; nesting is not supported. Inline "
                    f"the nested content or flatten the partials.")
    # Overrides are checked whether or not any prompt mentions them, because
    # nothing does: the engine composes a section by name, so an override's
    # only appearance in the bundle is the file itself.
    for name, override in _override_paths(partials_dir):
        nested = _nested_names(override.read_text(encoding="utf-8"))
        if nested:
            problems.append(
                f"engine-section override "
                f"prompts/partials/{ENGINE_NAMESPACE}/{name}.md nests further "
                f"include(s) {nested}; nesting is not supported. Inline the "
                f"nested content or flatten the partials.")
    if problems:
        raise ConfigBundleError(problems, path=root)


def _nested_names(content):
    """Every include placeholder inside a partial or an override, sorted."""
    return sorted(
        included_names(content)
        | {f"{ENGINE_NAMESPACE}:{n}" for n in engine_cited_names(content)})


# Tool-call cap placeholders, banned from every prompt. Neither is
# substituted, and interpolating a cap would couple it into prompt_hash (and
# thus config_fp), refusing the documented "hit the cap, resume with a raised
# cap" recovery. Any other unknown placeholder ships to the model verbatim;
# the checker's allowlists catch that for the two prompts the engine does
# substitute into.
_BANNED_PROMPT_PLACEHOLDERS = ("{max_tool_calls}",
                               "{max_review_tool_calls}")


def _validate_no_cap_placeholders(prompt_paths, partials_dir, root):
    """Fail loudly if any prompt or override interpolates a tool-call cap
    placeholder.

    A cap is an operational budget, kept out of the rendered prompt so
    raising it and resuming a cap-hit pause is not refused as config drift,
    and so no stage can read its own budget (see
    _BANNED_PROMPT_PLACEHOLDERS). Rejected at load, before any API spend.
    Includes are expanded first so a placeholder hiding in a partial is
    caught; every offender is collected into one error.
    """
    problems = []
    sources = [
        (path.relative_to(root),
         substitute_include_placeholders(
             path.read_text(encoding="utf-8"), partials_dir,
             predicates=EXPAND_ALL_BRANCHES))
        for path in prompt_paths
    ]
    sources += [
        (override.relative_to(root), override.read_text(encoding="utf-8"))
        for _, override in _override_paths(partials_dir)
    ]
    for where, text in sources:
        for token in _BANNED_PROMPT_PLACEHOLDERS:
            if token in text:
                problems.append(
                    f"prompt {where} contains the "
                    f"tool-call cap placeholder '{token}'. The cap is an "
                    f"operational budget, kept out of the rendered prompt (and "
                    f"so out of prompt_hash, config_fp, and review_fp) so "
                    f"raising it is not refused as config drift and no stage "
                    f"can read its own budget. Remove the placeholder.")
    if problems:
        raise ConfigBundleError(problems, path=root)


# Every placeholder `checker_prompts.build_checker_user_message` substitutes
# into the per-field scaffold — keep in step with that function.
# Substitution is a plain `str.replace`, so an unknown token would ship to
# the model as literal prompt text.
_CHECKER_USER_PLACEHOLDERS = frozenset({
    "field_path",
    "field_description",
    "extraction_instruction_block",
    "allowed_values_block",
    "identity_context",
    "evidence_block",
    "value",
    "notes_block",
})

# Every placeholder `checker_prompts.build_checker_system_text` substitutes
# into the checker SYSTEM prompt: the per-field check budget, and nothing
# else. The checker is sent no image labels, so `{image_labels_list}` in its
# prompt would reach the model as a literal token, and the check below names
# the variable instead. The extractor's and reviewer's prompts need no
# allowlist of their own: `prompt_builder` substitutes every slot either of
# them has.
_CHECKER_SYSTEM_PLACEHOLDERS = frozenset({"max_checks_per_field"})

# What counts as a placeholder for the checks above: a brace-wrapped lowercase
# identifier, nothing else, so the check reads a prompt the way the
# substitution does. `{include:NAME}` and `{reference:NAME}` carry a colon
# (handled and validated by their own passes); prose braces, JSON examples,
# and uppercase tokens do not match. The residual false positive — a bare
# lowercase identifier in braces meant as prose — has the same shape as a
# real slot, and rejecting it is the safe direction.
_PLACEHOLDER_TOKEN = re.compile(r"\{([a-z][a-z0-9_]*)\}")


def _validate_checker_placeholders(checker_system_path, partials_dir, root):
    """Fail loudly if the checker's config text cites a placeholder the engine
    does not substitute into it.

    Substitution is a plain `str.replace` per known slot, so an unknown
    placeholder is not a render-time error — it survives into the prompt as
    literal text, and the model reads `{field_pat}` where the field path
    should be. It is rejected here at load time instead.

    Three surfaces, because three pieces of text reach a checker call and the
    bundle can write any of them: the bundle's own checker prompt file
    (includes expanded, so a token arriving through a partial is caught), its
    overrides of the checker's system sections, and its override of the
    per-field scaffold. The two allowlists differ — the scaffold gets the
    per-field slots, the system prompt gets the check budget — so each surface
    is checked against its own. A slot a text OMITS is fine: the engine
    substitutes what is there.
    """
    checker_sections = {name for name, _ in spine_sections(CHECKER_SYSTEM)}
    surfaces = [(
        checker_system_path.relative_to(root),
        substitute_include_placeholders(
            checker_system_path.read_text(encoding="utf-8"), partials_dir,
            predicates=EXPAND_ALL_BRANCHES),
        _CHECKER_SYSTEM_PLACEHOLDERS,
    )]
    for name, override in _override_paths(partials_dir):
        if name in checker_sections:
            allowed = _CHECKER_SYSTEM_PLACEHOLDERS
        elif name == CHECKER_USER:
            allowed = _CHECKER_USER_PLACEHOLDERS
        else:
            continue
        surfaces.append((override.relative_to(root),
                         override.read_text(encoding="utf-8"), allowed))

    problems = []
    for where, text, allowed in surfaces:
        for name in sorted({n for n in _PLACEHOLDER_TOKEN.findall(text)
                            if n not in allowed}):
            problems.append(
                f"prompt {where} cites unknown placeholder '{{{name}}}'; the "
                f"engine substitutes only {sorted(allowed)} into it. An "
                f"unknown placeholder is not substituted, so it would be sent "
                f"to the checker as literal prompt text. Fix the spelling or "
                f"remove the placeholder.")
    if problems:
        raise ConfigBundleError(problems, path=root)


def _validate_prompt_references(prompt_paths, partials_dir, reference_lists,
                                root):
    """Fail loudly if any prompt or override cites a `{reference:NAME}`
    placeholder that the config bundle does not provide as a reference list.

    Includes are expanded first so a placeholder inside a partial is checked
    too (`_validate_prompt_partials` has already guaranteed they resolve). An
    override is checked on the same terms: reference substitution runs over
    the whole composed prompt, so an override may cite a list, and a name that
    resolves to nothing has to be found here. The render-time substitution in
    reference_lists.py raises on this too, but only mid-run; this hoists it to
    load time, before any API spend, collecting every offender into one error.
    """
    available = set(reference_lists)
    problems = []
    sources = [
        (path.relative_to(root),
         substitute_include_placeholders(
             path.read_text(encoding="utf-8"), partials_dir,
             predicates=EXPAND_ALL_BRANCHES))
        for path in prompt_paths
    ]
    sources += [
        (override.relative_to(root), override.read_text(encoding="utf-8"))
        for _, override in _override_paths(partials_dir)
    ]
    for where, text in sources:
        for name in sorted(referenced_names(text)):
            if name not in available:
                problems.append(
                    f"prompt {where} cites unknown "
                    f"reference list '{name}' via {{reference:{name}}}; "
                    f"available: {sorted(available)}. Add a "
                    f"reference/{name}.yaml or fix the placeholder.")
    if problems:
        raise ConfigBundleError(problems, path=root)
