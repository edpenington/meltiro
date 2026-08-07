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
        checker_user_template.md
        partials/              (optional)  one *.md per {include:NAME} block
          <name>.md

`load_config_bundle(path)` returns a frozen `ConfigBundle` exposing every
file path, the parsed `pipeline.yaml` mapping, and the loaded reference
lists. Validation is loud and happens at load, before any API spend: a
`ConfigBundleError` lists EVERY missing required file; every
`canonical_reference:` field and `{reference:NAME}` placeholder must resolve
to a loaded list (`reference/` itself is optional); every `{include:NAME}`
must resolve to a `prompts/partials/NAME.md`, with no nesting;
`pipeline.yaml` keys are checked against `KNOWN_PIPELINE_KEYS`; tool-call
cap placeholders are banned from all prompts; and the checker user template
may cite only the placeholders the engine substitutes.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from meltiro.errors import ConfigBundleError
from meltiro.fingerprint import (
    canonical_json,
    instrument_fingerprint,
    instrument_structure_hash,
    reference_lists_hash as _reference_lists_hash,
    tool_set_hash as _tool_set_hash,
)
from meltiro.prompt_partials import (
    EXPAND_ALL_BRANCHES,
    included_names,
    stage_predicates,
    substitute_include_placeholders,
)
from meltiro.reference_lists import load_reference_lists, referenced_names
from meltiro.template import iter_fields, load_template
from meltiro.thinking import PIPELINE_KEYS
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
    "checker_temperature",
    "review_temperature",
    "extractor_max_tokens",
    "review_max_tokens",
    "max_tool_calls",
    "max_review_tool_calls",
    "max_checks_per_field",
    "temperature",
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
    # Per-role thinking keys (`<role>_thinking_mode` / `_thinking_effort`),
    # spliced from `meltiro.thinking.PIPELINE_KEYS` so the allowlist cannot
    # drift from what the CLI reads. All optional. Decoding params: they ride
    # in their own role's stage fingerprint, never the instrument.
    *PIPELINE_KEYS,
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
    covers the four prompt files (partials expanded). `instrument_fp` is the
    model-free, engine-free composite of everything the config author wrote;
    the run-time `config_fp` additionally folds in the extractor model and
    decoding params and so is not derivable here.
    """

    root: Path
    template_path: Path
    reference_dir: Path | None
    reference_lists: dict  # name -> list of entries (empty when no reference/)
    extractor_system_path: Path
    review_system_path: Path
    checker_system_path: Path
    checker_user_template_path: Path
    partials_dir: Path
    pipeline: dict
    template_hash: str
    reference_lists_hash: str
    prompts_hash: str
    instrument_fp: str

    @property
    def prompt_paths(self):
        """The four prompt files, in the fixed order the hash is keyed by.

        Named once here so the load-time hash and any later recomputation
        cover the same set.
        """
        return [self.extractor_system_path, self.review_system_path,
                self.checker_system_path, self.checker_user_template_path]

    def prompts_hash_for(self, predicates):
        """The four-prompt content hash as it renders under `predicates`.

        `prompts_hash` above is this same hash under `pipeline.yaml`'s own
        toggles. A run can differ: `--max-checks-per-field 0` and
        `--no-final-review` change which `{include_if:...}` branches reach a
        model, so a run computes this against the toggles it actually
        honoured (see `Instrument.fingerprint`); the two agree exactly when
        no flag overrode the file.
        """
        return _compute_prompts_hash(
            self.prompt_paths, self.partials_dir, predicates)


def load_config_bundle(path):
    """Load and return a `ConfigBundle`. Raises `ConfigBundleError` if any
    required file is missing, `pipeline.yaml` doesn't parse to a mapping, a
    reference list is malformed, or the template names a `canonical_reference`
    that `reference/` does not provide.
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
    checker_user_template_path = prompts_dir / "checker_user_template.md"

    required = [
        pipeline_path,
        template_path,
        extractor_system_path,
        review_system_path,
        checker_system_path,
        checker_user_template_path,
    ]
    for p in required:
        if not p.is_file():
            problems.append(f"missing required file: {p.relative_to(root)}")

    if problems:
        raise ConfigBundleError(problems, path=root)

    with open(pipeline_path, "r", encoding="utf-8") as f:
        pipeline = strict_load(f)
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
    # fingerprint below.
    template = load_template(template_path)

    _cross_validate_references(template, reference_lists, root)

    prompt_paths = [extractor_system_path, review_system_path,
                    checker_system_path, checker_user_template_path]

    # Partials first: the later checks expand includes, so every include must
    # already resolve without nesting.
    _validate_prompt_partials(prompt_paths, partials_dir, root)

    _validate_no_cap_placeholders(prompt_paths, partials_dir, root)

    _validate_prompt_references(
        prompt_paths, partials_dir, reference_lists, root)

    # After the cap guard, so a cited cap gets its targeted message rather
    # than the generic unknown-placeholder one.
    _validate_checker_user_placeholders(
        checker_user_template_path, partials_dir, root)

    # Content fingerprint, computed here so the identity rides with the
    # loaded bundle.
    template_hash = template["template_hash"]
    ref_hash = _reference_lists_hash(reference_lists)
    prompts_hash = _compute_prompts_hash(
        prompt_paths, partials_dir,
        stage_predicates(
            pipeline.get("max_checks_per_field", DEFAULT_MAX_CHECKS_PER_FIELD),
            pipeline.get("final_review", True)))
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
        checker_user_template_path=checker_user_template_path,
        partials_dir=partials_dir,
        pipeline=pipeline,
        template_hash=template_hash,
        reference_lists_hash=ref_hash,
        prompts_hash=prompts_hash,
        instrument_fp=instrument_fp,
    )


def _compute_prompts_hash(prompt_paths, partials_dir, predicates):
    """SHA-256 of the four prompt files with `{include:NAME}` partials
    expanded.

    Partials are expanded so a prompt refactor that only moves shared text
    into a partial does not move the hash, while an edit to the text itself
    does. `{reference:NAME}` placeholders are left literal: reference-list
    CONTENT is captured separately by `reference_lists_hash`, so folding it
    in here too would be redundant and would couple the prompts hash to
    reference edits. Keyed by file stem and canonically serialised so the
    hash is order-independent and stable.
    """
    payload = {}
    for path in prompt_paths:
        text = path.read_text(encoding="utf-8")
        payload[path.stem] = substitute_include_placeholders(
            text, partials_dir, predicates=predicates)
    canonical = canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _content_instrument_fingerprint(template, pipeline, prompts_hash,
                                    template_hash, reference_hash):
    """The bundle's instrument fingerprint: the model-free, engine-free
    identity of everything the config author wrote.

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


def _validate_prompt_partials(prompt_paths, partials_dir, root):
    """Fail loudly if any prompt cites an `{include:NAME}` with no
    `prompts/partials/NAME.md`, or a partial nests another include.

    The render-time expansion in prompt_partials.py raises on both faults
    too, but only mid-run; this hoists them to config-load time, before any
    API spend, and collects every offender into one error. `partials/` itself
    is optional and touched only for includes actually cited.
    """
    problems = []
    checked = set()
    for path in prompt_paths:
        text = path.read_text(encoding="utf-8")
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
            nested = sorted(
                included_names(partial_path.read_text(encoding="utf-8")))
            if nested:
                problems.append(
                    f"partial prompts/partials/{name}.md nests further "
                    f"include(s) {nested}; nesting is not supported. Inline "
                    f"the nested content or flatten the partials.")
    if problems:
        raise ConfigBundleError(problems, path=root)


# Tool-call cap placeholders, banned from every prompt. Neither is
# substituted, and interpolating a cap would couple it into prompt_hash (and
# thus config_fp), refusing the documented "hit the cap, resume with a raised
# cap" recovery. Any other unknown placeholder ships to the model verbatim;
# the checker user template's allowlist catches that for the one prompt the
# engine does substitute into.
_BANNED_PROMPT_PLACEHOLDERS = ("{max_tool_calls}",
                               "{max_review_tool_calls}")


def _validate_no_cap_placeholders(prompt_paths, partials_dir, root):
    """Fail loudly if any prompt interpolates a tool-call cap placeholder.

    A cap is an operational budget, kept out of the rendered prompt so
    raising it and resuming a cap-hit pause is not refused as config drift,
    and so no stage can read its own budget (see
    _BANNED_PROMPT_PLACEHOLDERS). Rejected at load, before any API spend.
    Includes are expanded first so a placeholder hiding in a partial is
    caught; every offender is collected into one error.
    """
    problems = []
    for path in prompt_paths:
        text = path.read_text(encoding="utf-8")
        expanded = substitute_include_placeholders(
            text, partials_dir, predicates=EXPAND_ALL_BRANCHES)
        for token in _BANNED_PROMPT_PLACEHOLDERS:
            if token in expanded:
                problems.append(
                    f"prompt {path.relative_to(root)} contains the "
                    f"tool-call cap placeholder '{token}'. The cap is an "
                    f"operational budget, kept out of the rendered prompt (and "
                    f"so out of prompt_hash, config_fp, and review_fp) so "
                    f"raising it is not refused as config drift and no stage "
                    f"can read its own budget. Remove the placeholder.")
    if problems:
        raise ConfigBundleError(problems, path=root)


# Every placeholder `checker_prompts.build_checker_user_message` substitutes
# into the checker user template — keep in step with that function.
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

# What counts as a placeholder for the check above: a brace-wrapped lowercase
# identifier, nothing else, so the check reads a prompt the way the
# substitution does. `{include:NAME}` and `{reference:NAME}` carry a colon
# (handled and validated by their own passes); prose braces, JSON examples,
# and uppercase tokens do not match. The residual false positive — a bare
# lowercase identifier in braces meant as prose — has the same shape as a
# real slot, and rejecting it is the safe direction.
_PLACEHOLDER_TOKEN = re.compile(r"\{([a-z][a-z0-9_]*)\}")


def _validate_checker_user_placeholders(checker_user_template_path,
                                        partials_dir, root):
    """Fail loudly if the checker user template cites a placeholder the
    engine does not substitute.

    An unknown placeholder is not a render-time error — it survives into the
    prompt as literal text — so it is rejected here at load time instead.
    Includes are expanded first; every offending token is collected into one
    error. A slot the template OMITS is fine: the engine substitutes what is
    there.
    """
    text = checker_user_template_path.read_text(encoding="utf-8")
    expanded = substitute_include_placeholders(
        text, partials_dir, predicates=EXPAND_ALL_BRANCHES)
    unknown = sorted({
        name for name in _PLACEHOLDER_TOKEN.findall(expanded)
        if name not in _CHECKER_USER_PLACEHOLDERS
    })
    if unknown:
        raise ConfigBundleError(
            [f"prompt {checker_user_template_path.relative_to(root)} cites "
             f"unknown placeholder '{{{name}}}'; the engine substitutes only "
             f"{sorted(_CHECKER_USER_PLACEHOLDERS)}. An unknown placeholder is "
             f"not substituted, so it would be sent to the checker as literal "
             f"prompt text. Fix the spelling or remove the placeholder."
             for name in unknown],
            path=root,
        )


def _validate_prompt_references(prompt_paths, partials_dir, reference_lists,
                                root):
    """Fail loudly if any prompt cites a `{reference:NAME}` placeholder that
    the config bundle does not provide as a reference list.

    Includes are expanded first so a placeholder inside a partial is checked
    too (`_validate_prompt_partials` has already guaranteed they resolve).
    The render-time substitution in reference_lists.py raises on this too,
    but only mid-run; this hoists it to load time, before any API spend,
    collecting every offender into one error.
    """
    available = set(reference_lists)
    problems = []
    for path in prompt_paths:
        text = path.read_text(encoding="utf-8")
        expanded = substitute_include_placeholders(
            text, partials_dir, predicates=EXPAND_ALL_BRANCHES)
        for name in sorted(referenced_names(expanded)):
            if name not in available:
                problems.append(
                    f"prompt {path.relative_to(root)} cites unknown "
                    f"reference list '{name}' via {{reference:{name}}}; "
                    f"available: {sorted(available)}. Add a "
                    f"reference/{name}.yaml or fix the placeholder.")
    if problems:
        raise ConfigBundleError(problems, path=root)
