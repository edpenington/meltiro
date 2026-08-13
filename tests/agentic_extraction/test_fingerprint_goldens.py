"""Golden fingerprint values: the literals a published run is checked against.

A meltiro fingerprint is a published number. A run record carries `config_fp`,
`checker_fp`, `review_fp`, `run_fp`, `instrument_fp`, the three per-role
`*_call_fp` values and `engine_fp`, and a reader who wants to know whether a
reported extraction really came from a stated version of this code and a stated
config bundle recomputes those hashes and compares them with what was
published. That check is worth something only while the hashes are STABLE: a
fingerprint that quietly takes a new value for the same code and the same
config makes every previously published run unverifiable, because the reader
recomputes a number that no longer matches the printed one and has no way to
tell an accidental change from a deliberate one.

Every other fingerprint test in this directory asserts a RELATION: editing this
input moves that hash, editing that one leaves it alone. Relations survive a
preimage change. Reorder two components, drop one, change a separator, swap a
raw file hash for a rendered one, and every relation still holds while every
published value moves. Each assertion here instead compares a freshly computed
hash against a literal string written into this file, so a preimage change
surfaces as a failing test that names the fingerprint that moved.

The literals are computed against `tests/fixtures/config_synthetic/`, the
config bundle the engine is exercised against. It is repo content, so it fully
determines them: no clock, no session id, no network.

UPDATING A GOLDEN. A golden is correct to change only when the preimage was
MEANT to change. Changing one is a decision to break the reproducibility of
every fingerprint of that kind already published, so it belongs in a commit
that says in words what moved and why the break is worth it. Recomputing a
golden to turn a red test green, with no such reason recorded, silently
destroys the guarantee the fingerprints exist to give, and does it in the one
place nothing downstream can detect.

WHAT IS PINNED, AND WHAT CANNOT BE
----------------------------------
`instrument_fp` is pinned as a run records it: it is model-free and
engine-free by construction, so the config directory alone determines it.

The three stage fingerprints and `call_fp` each take a provider-call identity
block as their first component. That block is direktoro's (model, provider,
base_url, route, wire-keyed resolved decoding params), and its content is owned
by that package's registry rather than by anything in this repo. They are
therefore pinned twice, for two different reasons:

  - with `PINNED_CALL_IDENTITY`, a fixed placeholder standing in for the block,
    against the REAL prompts, template, tools and reference lists of the
    fixture bundle. This pins meltiro's own composition: which components ride
    in the preimage, in what order, under what separator, behind what prefix.
  - end to end for the extractor role, block included, so a change in the
    registry that would silently move every published `config_fp` is caught
    where it can be read as what it is.

`engine_fp` folds in both engine packages' versions and a content hash of each
one's source, none of which a test can pin: the source hash moves with every
edit to either package, and that is the point of it. What is pinned is the
function's preimage, from fixed versions and fixed stand-in source digests,
which is what fixes the `|` join, the component order, and the `nosource` and
`nodirektoro` tokens.

`run_fp` folds `engine_fp` in, so a run's `run_fp` is not pinnable either, and
deliberately so: `run_fp` equality is documented not to survive a release. What
is pinned is the composition, from the stage goldens above and a fixed engine
fingerprint, including the `ABSENT_STAGE` sentinel an ablated stage adds.
"""

import pytest

from direktoro import (
    call_identity_fields, canonical_json, model_info, resolved_decoding_params,
    split_decoding_config)
from meltiro.checker import CheckerConfig
from meltiro.checker_prompts import (
    build_checker_config_text, checker_user_config_text)
from meltiro.config_bundle import load_config_bundle
from meltiro.fingerprint import (
    call_fingerprint,
    checker_config_fingerprint,
    config_fingerprint,
    engine_fingerprint,
    field_catalogue_hash,
    reference_lists_hash,
    review_config_fingerprint,
    run_fingerprint,
    structure_hash,
    tool_set_hash,
)
from meltiro.prompt_builder import (
    build_config_prompt_text, compute_prompt_config_hash)
from meltiro.prompt_partials import REVIEW_SYSTEM, stage_predicates
from meltiro.template import load_template
from meltiro.tools import all_tool_definitions, checker_tool_definitions


# ---------------------------------------------------------------------------
# The goldens
# ---------------------------------------------------------------------------

# The placeholder standing in for direktoro's provider-call identity block.
# fingerprint.py treats that block as an opaque string it never parses, so a
# fixed value here exercises the composition exactly as a real block does while
# leaving the block's own content to the package that owns it.
PINNED_CALL_IDENTITY = "call-identity-golden-fixture"

# Fixed engine identity. A real one is two package versions and a sha256 over
# each package's own source files, all four read off the running install; these
# stand in so the preimage can be pinned. The source digests are visibly
# synthetic, because what is pinnable about them is their PLACE in the
# preimage, never their value: the real ones move with every edit to either
# package, which is what the axis exists to report.
PINNED_MELTIRO_VERSION = "1.2.3"
PINNED_DIREKTORO_VERSION = "4.5.6"
PINNED_MELTIRO_SOURCE = (
    "1111111111111111111111111111111111111111111111111111111111111111")
PINNED_DIREKTORO_SOURCE = (
    "2222222222222222222222222222222222222222222222222222222222222222")

# Component hashes over tests/fixtures/config_synthetic. Each is an input to
# one or more fingerprints below, pinned separately so a mismatch says WHICH
# input moved rather than only that some composite did.
GOLDEN_TEMPLATE_HASH = \
    "1821dbceb8ba6c4c8ac4835b155420d954a3c6c52ab1d1b42ee477e46c834ce5"
GOLDEN_PROMPTS_HASH = \
    "bc1d75f039c3c163ccd216054fa63d80ca81f96438d5a3d1a2a81b7f95f70ce1"
GOLDEN_REFERENCE_LISTS_HASH = \
    "51c7c7184b640abd363e95f5c169cb963760badb1792400a6ac5f017897fbcf6"
GOLDEN_TOOL_SET_HASH = \
    "bdf228c9cccfdfce1d06151fee8a72ee27b80a787dba21da7cac7b0bfee1f2a5"
GOLDEN_FIELD_CATALOGUE_HASH = \
    "aaf68cbd050c4f25e4981c0bb15e9255875c72ef34e033cc276f395bc2b60d94"
GOLDEN_EXTRACTOR_PROMPT_HASH = \
    "1efa5a75b25a6a14d3b40dd5fb2b5664ec51ab4b448ddf0cbf237062520f3a16"

# The instrument axis, as a run and `meltiro fingerprint` both record it.
GOLDEN_INSTRUMENT_FP = ("instrument_fp:"
                        "cf414ea2598141a1f6d3f7c453f9f9897345b4898ad93c511"
                        "8d805b1f4fede22")

# Stage fingerprints over the fixture bundle's real content, with
# PINNED_CALL_IDENTITY in place of the provider-call identity block.
GOLDEN_CONFIG_FP = ("config_fp:"
                    "0be3bf013967bed824b31e35ab7202451e487b3ecc8be9f13"
                    "f7dac308c732bd6")
GOLDEN_CHECKER_FP = ("checker_fp:"
                     "47b32127bf0baaf2ee167bffc81ce64f83f113bf5e1a8b010"
                     "7ebcb58b7888967")
GOLDEN_REVIEW_FP = ("review_fp:"
                    "9d59a73abc609e5780216e531996c4c61bdb8b2ce905c9649"
                    "dbb04f466729503")

# The call axis over the same placeholder block.
GOLDEN_CALL_FP = ("call_fp:"
                  "30bce17986a0525a32a4972e571427a099a4de3b1d76f4abbf94"
                  "49eb31ced079")

# The engine axis, over the fixed versions and source digests above. One
# golden for the whole-engine composition, then one per stand-in token: an
# unreadable *meltiro* source, and an absent direktoro (whose version and
# source both fold in as the same token).
GOLDEN_ENGINE_FP = ("engine_fp:"
                    "201320619f79a2c29f9a74a0178fe482bc0f5ecb2ba32e92f163"
                    "f985802b2a9d")
GOLDEN_ENGINE_FP_NOSOURCE = ("engine_fp:"
                             "3a8d88b0c7002c87bead3ecfed201fda8bde8521a979"
                             "e6710ec3c680d6e435c3")
GOLDEN_ENGINE_FP_NO_DIREKTORO = ("engine_fp:"
                                 "1119ce400f2e5857955723bf0261c0aef1ff9ef1"
                                 "c248c0e9a6f772698a2509bc")

# A fixed engine fingerprint standing in for the engine axis in the `run_fp`
# composition below, on the same terms as `PINNED_CALL_IDENTITY` stands in for
# direktoro's block: `run_fingerprint` hashes this string verbatim and never
# parses it, so any well-formed value exercises the fold, and holding it fixed
# is what keeps the two `run_fp` goldens a statement about the FOLD alone.
PINNED_ENGINE_FP = ("engine_fp:"
                    "9afeb473afe66aa91b41d6afd7e0082ad497bd345672f3"
                    "fb0d8649b8da02ee85")

# The whole-run identity, composed from the stage goldens and the engine
# placeholder above.
GOLDEN_RUN_FP = ("run_fp:"
                 "7e491e854abab3cdf7ad4bbbdf9a306e55ef9db9bdfa49513cfb82e"
                 "8038f07c4")
GOLDEN_RUN_FP_EXTRACTOR_ONLY = ("run_fp:"
                                "c4980cf601086e5955f3cc228d10023a05e62af9e"
                                "6319de15de981ac0f2149f8")

# End to end for the extractor role: direktoro's real call-identity block for
# the fixture pipeline's extractor model, and the config_fp it produces. This
# is the value a run against tests/fixtures/config_synthetic actually records.
GOLDEN_EXTRACTOR_CALL_IDENTITY = (
    '{"base_url":null,"decoding_params":{"max_tokens":32768},'
    '"model":"claude-opus-4-8","provider":"anthropic"}')
GOLDEN_RUN_CONFIG_FP = ("config_fp:"
                        "956725d811ee751cc363cae03f357a9674ba66b1d939e3079"
                        "e6fe318f450786c")


# ---------------------------------------------------------------------------
# Recomputing the pinned values
# ---------------------------------------------------------------------------

@pytest.fixture
def bundle(config_dir):
    return load_config_bundle(config_dir)


@pytest.fixture
def template(bundle):
    return load_template(bundle.template_path)


def _pipeline_structure(bundle, *, supports_images=True):
    """The extractor/reviewer structure component for the fixture pipeline."""
    loop = bundle.pipeline
    return structure_hash(
        int(loop["max_checks_per_field"]),
        final_review=bool(loop["final_review"]),
        supports_images=supports_images,
        check_reviewer_edits=bool(loop["check_reviewer_edits"]),
    )


def _extractor_prompt_hash(bundle, template):
    loop = bundle.pipeline
    return compute_prompt_config_hash(
        system_prompt_path=bundle.extractor_system_path,
        max_checks_per_field=int(loop["max_checks_per_field"]),
        final_review=bool(loop["final_review"]),
        reference_lists=bundle.reference_lists,
    )


def _checker_config(bundle):
    """A CheckerConfig carrying the fixture bundle's prompts and widths.

    No model: the model reaches `checker_fp` only through the call-identity
    block, which the goldens supply as `PINNED_CALL_IDENTITY`. No structure
    either: the checker's prompts render against the run's predicates, which
    `_pipeline_predicates` below supplies from the same pipeline.yaml.
    """
    loop = bundle.pipeline
    return CheckerConfig(
        max_tokens=1024,
        system_prompt_path=str(bundle.checker_system_path),
        context_chars=int(loop["checker_context_chars"]),
    )


def _pipeline_predicates(bundle):
    """The fixture pipeline's `{include_if:...}` predicate map."""
    loop = bundle.pipeline
    return stage_predicates(
        int(loop["max_checks_per_field"]), bool(loop["final_review"]))


def _review_config_text(bundle, template):
    # The config's half of the reviewer's prompt, exactly as the review
    # fingerprint takes it: paper-independent, and with the engine's own
    # sections outside the preimage.
    loop = bundle.pipeline
    return build_config_prompt_text(
        REVIEW_SYSTEM,
        system_prompt_path=bundle.review_system_path,
        max_checks_per_field=int(loop["max_checks_per_field"]),
        final_review=bool(loop["final_review"]),
        reference_lists=bundle.reference_lists,
    )


def _extractor_call_identity(bundle):
    """direktoro's real provider-call identity block for the fixture's
    extractor model, built the way the orchestrator builds it."""
    loop = bundle.pipeline
    model = loop["extractor_model"]
    sampling, thinking = split_decoding_config(loop["extractor_decoding"])
    decoding = resolved_decoding_params(
        model, sampling=sampling,
        max_tokens=int(loop["extractor_max_tokens"]), thinking=thinking)
    return canonical_json(call_identity_fields(
        model, route=model_info(model).route, decoding_params=decoding))


def _moved(name, preimage, consequence):
    """The failure message for a golden that no longer matches."""
    return (
        f"{name} has moved. Its preimage ({preimage}) is no longer built the "
        f"way it was when this golden was recorded, so {consequence} "
        f"Update the golden only alongside a stated reason for the preimage "
        f"change; recomputing it to make this test pass hides the break."
    )


# ---------------------------------------------------------------------------
# Algorithm identity
# ---------------------------------------------------------------------------

_HEX = frozenset("0123456789abcdef")

_PREFIXED_GOLDENS = {
    "instrument_fp": GOLDEN_INSTRUMENT_FP,
    "config_fp": GOLDEN_CONFIG_FP,
    "checker_fp": GOLDEN_CHECKER_FP,
    "review_fp": GOLDEN_REVIEW_FP,
    "call_fp": GOLDEN_CALL_FP,
    "engine_fp": GOLDEN_ENGINE_FP,
    "run_fp": GOLDEN_RUN_FP,
}

_BARE_HASH_GOLDENS = {
    "template_hash": GOLDEN_TEMPLATE_HASH,
    "prompts_hash": GOLDEN_PROMPTS_HASH,
    "reference_lists_hash": GOLDEN_REFERENCE_LISTS_HASH,
    "tool_set_hash": GOLDEN_TOOL_SET_HASH,
    "field_catalogue_hash": GOLDEN_FIELD_CATALOGUE_HASH,
    "prompt_hash": GOLDEN_EXTRACTOR_PROMPT_HASH,
}


class TestDigestAlgorithm:
    """Which hash function is in use, pinned on its own.

    Swapping SHA-256 for another digest moves every value in this file at
    once, and a wall of value mismatches reads like a content edit. These
    assertions name the real cause instead, and they are cheap: a known answer
    and a length/charset check."""

    def test_the_digest_function_is_sha256(self):
        # `reference_lists_hash({})` canonicalises to the two bytes `{}` and
        # hashes them, so this is the known-answer test for the digest itself:
        # SHA-256 of "{}". It is independent of every prompt, template and
        # reference list in the repo, so it moves only if the hash function or
        # the canonical-JSON form does.
        assert reference_lists_hash({}) == (
            "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
        ), (
            "the digest behind every meltiro fingerprint is no longer SHA-256 "
            "of the canonical JSON, or the canonical JSON form has changed. "
            "Every fingerprint this repo has ever published was computed with "
            "the old one and none of them can be reproduced with the new one."
        )

    @pytest.mark.parametrize("name,value", sorted(_PREFIXED_GOLDENS.items()))
    def test_fingerprints_are_a_prefix_and_a_full_sha256(self, name, value):
        # Self-prefixed, then the full untruncated digest: consumers split on
        # the first colon and compare the rest, and the six-character
        # shortening
        # lives only in a session directory name.
        prefix, _, digest = value.partition(":")
        assert prefix == name
        assert len(digest) == 64, (
            f"{name} is no longer a full-length SHA-256 digest. Truncating or "
            f"lengthening it changes every published {name} and weakens the "
            f"collision guarantee a reader relies on.")
        assert set(digest) <= _HEX, (
            f"{name} is no longer lowercase hexadecimal. A re-encoding "
            f"(base64, uppercase) makes every published {name} fail a "
            f"literal comparison even where the preimage is unchanged.")

    @pytest.mark.parametrize("name,value", sorted(_BARE_HASH_GOLDENS.items()))
    def test_component_hashes_are_bare_sha256_hex(self, name, value):
        # The component hashes are unprefixed: they are written to run.json as
        # plain digests and fed to the fingerprints as plain digests.
        assert len(value) == 64 and set(value) <= _HEX, (
            f"{name} is no longer a bare lowercase-hex SHA-256 digest. It is "
            f"recorded in run.json and folded into a stage fingerprint, so a "
            f"change of encoding moves both.")


# ---------------------------------------------------------------------------
# Component hashes over the fixture config bundle
# ---------------------------------------------------------------------------

class TestConfigContentGoldens:
    """The content hashes `tests/fixtures/config_synthetic/` determines.

    Each is an input to at least one fingerprint below and each is recorded in
    a run or exposed on the loaded bundle, so a consumer may be pinning it
    directly. Pinning them one by one is what turns a moved composite into a
    diagnosis: exactly one of these fails and says which file's serialisation
    changed."""

    def test_template_hash(self, bundle):
        assert bundle.template_hash == GOLDEN_TEMPLATE_HASH, _moved(
            "template_hash",
            "the extraction template file's bytes",
            "the value recorded in every run.json and folded into config_fp "
            "and instrument_fp has changed for an unedited template.")

    def test_prompts_hash(self, bundle):
        assert bundle.prompts_hash == GOLDEN_PROMPTS_HASH, _moved(
            "prompts_hash",
            "the three prompt files with {include:NAME} partials expanded, "
            "plus every engine section this bundle overrides",
            "every published instrument_fp is now unreproducible, and "
            "`meltiro fingerprint` prints a different identity for an "
            "unedited config bundle.")

    def test_reference_lists_hash(self, bundle):
        assert bundle.reference_lists_hash == GOLDEN_REFERENCE_LISTS_HASH, \
            _moved(
                "reference_lists_hash",
                "each list's canonical names and aliases, in file order under "
                "sorted list names",
                "config_fp, checker_fp and instrument_fp have all moved for "
                "unedited reference lists, and a consumer pinning "
                "(template_hash, reference_lists_hash) to decide whether a "
                "stored value is still legal sees a false drift.")

    def test_tool_set_hash(self, bundle, template):
        assert tool_set_hash(all_tool_definitions(template)) == \
            GOLDEN_TOOL_SET_HASH, _moved(
                "tool_set_hash",
                "the canonical JSON of every role's tool definitions",
                "config_fp, review_fp and instrument_fp have moved for an "
                "unedited template, so runs before and after this change look "
                "incomparable when the model saw the same schemas.")

    def test_field_catalogue_hash(self, template):
        assert field_catalogue_hash(template) == \
            GOLDEN_FIELD_CATALOGUE_HASH, _moved(
                "field_catalogue_hash",
                "the checker-relevant attributes of every template field",
                "every published checker_fp is unreproducible. Watch for the "
                "silent direction too: an attribute DROPPED from the hashed "
                "subset lets a template edit change every checker call while "
                "checker_fp reports the instrument unchanged.")

    def test_extractor_prompt_hash(self, bundle, template):
        assert _extractor_prompt_hash(bundle, template) == \
            GOLDEN_EXTRACTOR_PROMPT_HASH, _moved(
                "prompt_hash (the rendered, paper-independent extractor "
                "system prompt)",
                "the extractor bundle prompt rendered with an empty image "
                "label list, references substituted and partials expanded, "
                "beside this bundle's overrides of the extractor's engine "
                "sections",
                "the hash recorded in every run.json and folded into "
                "config_fp "
                "has moved for an unedited prompt file. If the render now "
                "varies per paper, config_fp stops grouping runs by config at "
                "all.")


# ---------------------------------------------------------------------------
# The instrument axis, pinned as a run records it
# ---------------------------------------------------------------------------

class TestInstrumentFingerprintGolden:
    """`instrument_fp` is the one published fingerprint a config directory
    fully determines: no model, no engine, no paper. It is what a reader uses
    to say two runs asked the same question of different models, so it is
    pinned here exactly as a run and `meltiro fingerprint` both record it."""

    def test_instrument_fp_for_the_fixture_bundle(self, bundle):
        assert bundle.instrument_fp == GOLDEN_INSTRUMENT_FP, _moved(
            "instrument_fp",
            "prompts_hash | template_hash | tool_set_hash | "
            "instrument_structure_hash | reference_lists_hash | context "
            "chars | ordered checker context fields",
            "every instrument_fp ever published is now unreproducible from "
            "this code, and the model-comparison claim it exists to support "
            "(same instrument, different API) can no longer be checked "
            "against anything already in print. If the component hashes above "
            "all still pass, the composition itself changed: an order, a "
            "separator, or a component added or dropped.")


# ---------------------------------------------------------------------------
# Stage fingerprint composition, over real config content
# ---------------------------------------------------------------------------

class TestStageFingerprintComposition:
    """The three stage fingerprints, over the fixture bundle's real prompts,
    template, tools and reference lists, with `PINNED_CALL_IDENTITY` standing
    in for direktoro's provider-call identity block.

    What this pins is the half meltiro owns: which content components ride in
    each stage's preimage, in which order, under which separator, behind which
    prefix. The block itself is pinned separately below."""

    def test_config_fp(self, bundle, template):
        fp = config_fingerprint(
            PINNED_CALL_IDENTITY,
            _extractor_prompt_hash(bundle, template),
            bundle.template_hash,
            tool_set_hash=tool_set_hash(all_tool_definitions(template)),
            structure_hash=_pipeline_structure(bundle),
            reference_hash=bundle.reference_lists_hash,
        )
        assert fp == GOLDEN_CONFIG_FP, _moved(
            "config_fp",
            "call identity | prompt_hash | template_hash | tool_set_hash | "
            "structure | reference_lists_hash",
            "every extractor-stage fingerprint this repo has published is "
            "unreproducible, resumes of existing sessions are refused as "
            "config drift, and downstream keys built on config_fp no longer "
            "match the runs they name.")

    def test_checker_fp(self, bundle, template):
        checker = _checker_config(bundle)
        predicates = _pipeline_predicates(bundle)
        # The CONFIG half of both prompts, as CheckerConfig.fingerprint takes
        # them: the checker is SENT the engine's briefing and hashes only what
        # this bundle wrote around it.
        system_text = build_checker_config_text(
            system_prompt_path=bundle.checker_system_path,
            reference_lists=bundle.reference_lists,
            predicates=predicates,
            max_checks_per_field=int(
                bundle.pipeline["max_checks_per_field"]),
        )
        fp = checker_config_fingerprint(
            PINNED_CALL_IDENTITY,
            system_text,
            checker_user_config_text(bundle.partials_dir,
                                     predicates=predicates),
            # The schema the checker's verdict must fit. Its own catalogue,
            # hashed apart from the extractor's and the reviewer's, so this
            # component moves only when the shape of a verdict does.
            tool_set_hash=tool_set_hash(checker_tool_definitions()),
            # The checker checks nothing of its own, so its structure component
            # carries only the image-capability toggle.
            structure_hash=structure_hash(0, supports_images=True),
            field_catalogue_hash_str=field_catalogue_hash(template),
            reference_hash=bundle.reference_lists_hash,
            checker_context_fields=template.get("checker_context_fields"),
            checker_context_chars=checker.context_chars,
        )
        assert fp == GOLDEN_CHECKER_FP, _moved(
            "checker_fp",
            "call identity | system prompt | user prompt template | "
            "checker tool set | structure | field_catalogue_hash | "
            "reference_lists_hash | ordered context fields | "
            "quote-context width",
            "every checker-stage fingerprint this repo has published is "
            "unreproducible, and the per-config audit trail of checker "
            "verdicts no longer lines up with the runs that produced it.")

    def test_review_fp(self, bundle, template):
        fp = review_config_fingerprint(
            PINNED_CALL_IDENTITY,
            _review_config_text(bundle, template),
            tool_set_hash=tool_set_hash(all_tool_definitions(template)),
            structure_hash=_pipeline_structure(bundle),
            reference_hash=bundle.reference_lists_hash,
        )
        assert fp == GOLDEN_REVIEW_FP, _moved(
            "review_fp",
            "call identity | the review prompt's config text | "
            "tool_set_hash | "
            "structure | reference_lists_hash",
            "every review-stage fingerprint this repo has published is "
            "unreproducible. The reviewer may edit the shipped extraction "
            "output, so this is the only recorded provenance for the pass "
            "that produced the final values.")


# ---------------------------------------------------------------------------
# The call and engine axes, and the whole-run identity
# ---------------------------------------------------------------------------

class TestAxisAndRunGoldens:
    """`call_fp`, `engine_fp` and `run_fp`.

    None of the three can be pinned as a live run records them (the first
    depends on a registry, the other two on the source that is running), so
    what is pinned is each function's preimage from fixed inputs. That is
    enough to catch a change of join, order, prefix or sentinel, which is what
    would move every published value at once."""

    def test_call_fp(self):
        assert call_fingerprint(PINNED_CALL_IDENTITY) == GOLDEN_CALL_FP, \
            _moved(
                "call_fp",
                "the provider-call identity block, hashed on its own",
                "every per-role call fingerprint recorded in run.json has "
                "moved, so the 'same instrument, different API' comparison "
                "can no longer be made against runs already published.")

    def test_engine_fp(self):
        # The whole engine in one value: each package named by the version it
        # declares and by a digest of the source that ran.
        assert engine_fingerprint(
            PINNED_MELTIRO_VERSION, PINNED_MELTIRO_SOURCE,
            PINNED_DIREKTORO_VERSION, PINNED_DIREKTORO_SOURCE) == \
            GOLDEN_ENGINE_FP, _moved(
                "engine_fp",
                "meltiro version | meltiro source digest | direktoro version "
                "| direktoro source digest",
                "every published engine_fp and every run_fp built on one has "
                "moved, and the engine axis can no longer be compared against "
                "runs already in print.")

    def test_engine_fp_with_unreadable_source(self):
        # A frozen or zipimported copy: the package's own files cannot be
        # walked, so the source component is a fixed token and the version
        # carries the identity alone. The token has to stay fixed, because a
        # digest over nothing would be a constant claiming to name whatever
        # produced it.
        assert engine_fingerprint(
            PINNED_MELTIRO_VERSION, "nosource",
            PINNED_DIREKTORO_VERSION, PINNED_DIREKTORO_SOURCE) == \
            GOLDEN_ENGINE_FP_NOSOURCE, _moved(
                "engine_fp with unreadable meltiro source",
                "meltiro version | the literal token `nosource` | direktoro "
                "version | direktoro source digest",
                "the unreadable-source token has changed, so every engine_fp "
                "recorded by a frozen or zipimported copy no longer "
                "recomputes.")

    def test_engine_fp_without_direktoro(self):
        # meltiro imports with direktoro absent, so both direktoro components
        # can genuinely be missing, and each folds in as the same fixed token
        # rather than as the empty string or `str(None)`. An engine with no
        # direktoro is a real, distinct engine: it can read and validate
        # bundles but never place a call, so it must not share a fingerprint
        # with one that can.
        assert engine_fingerprint(
            PINNED_MELTIRO_VERSION, PINNED_MELTIRO_SOURCE, None, None) == \
            GOLDEN_ENGINE_FP_NO_DIREKTORO, _moved(
                "engine_fp with no direktoro installed",
                "meltiro version | meltiro source digest | the literal token "
                "`nodirektoro` twice",
                "the absent-direktoro token has changed, so an engine_fp "
                "recorded by a --no-deps install no longer recomputes.")

    def test_the_source_digests_are_terms_of_their_own(self):
        # The property the axis is built on, asserted directly rather than
        # left to two goldens differing: with both versions held fixed, an
        # edit to either package's source moves the fingerprint. A release
        # number is a claim about the code; these are the code.
        base = engine_fingerprint(
            PINNED_MELTIRO_VERSION, PINNED_MELTIRO_SOURCE,
            PINNED_DIREKTORO_VERSION, PINNED_DIREKTORO_SOURCE)
        edited_meltiro = engine_fingerprint(
            PINNED_MELTIRO_VERSION, "3" * 64,
            PINNED_DIREKTORO_VERSION, PINNED_DIREKTORO_SOURCE)
        edited_direktoro = engine_fingerprint(
            PINNED_MELTIRO_VERSION, PINNED_MELTIRO_SOURCE,
            PINNED_DIREKTORO_VERSION, "4" * 64)
        assert base != edited_meltiro, (
            "engine_fp no longer moves when meltiro's source does, so a "
            "patched or edited copy shares a fingerprint with the release it "
            "started as and the axis names a release rather than the code.")
        assert base != edited_direktoro
        assert edited_meltiro != edited_direktoro

    def test_the_direktoro_version_is_part_of_the_engine(self):
        # The reason the component is there at all. direktoro builds the
        # call-identity block that leads every stage fingerprint, so two runs
        # under different direktoro versions were produced by different
        # engines and the axis has to say so.
        first = engine_fingerprint(
            PINNED_MELTIRO_VERSION, PINNED_MELTIRO_SOURCE, "4.5.6",
            PINNED_DIREKTORO_SOURCE)
        second = engine_fingerprint(
            PINNED_MELTIRO_VERSION, PINNED_MELTIRO_SOURCE, "4.5.7",
            PINNED_DIREKTORO_SOURCE)
        assert first != second, (
            "engine_fp no longer moves when direktoro does, so a release that "
            "changes the shape of the provider-call identity block moves every "
            "stage fingerprint while the engine axis reports no change at all.")

    def test_run_fp(self):
        assert run_fingerprint(
            GOLDEN_CONFIG_FP, GOLDEN_CHECKER_FP, GOLDEN_REVIEW_FP,
            PINNED_ENGINE_FP) == GOLDEN_RUN_FP, _moved(
                "run_fp",
                "config_fp | checker_fp | review_fp | engine_fp, each hashed "
                "verbatim in that fixed order",
                "every `llm:<run_fp>` producer string a consumer has built "
                "names a run that can no longer be reproduced. Note the "
                "stage goldens are passed in verbatim here, so this failing "
                "alone means the FOLD changed, not any stage.")

    def test_run_fp_for_an_extractor_only_ablation(self):
        # Both optional stages off. Each absence folds in as the documented
        # sentinel in its own fixed slot, which is what keeps the four ablation
        # shapes distinct and well-defined.
        assert run_fingerprint(
            GOLDEN_CONFIG_FP, None, None, PINNED_ENGINE_FP) == \
            GOLDEN_RUN_FP_EXTRACTOR_ONLY, _moved(
                "run_fp for an extractor-only run",
                "config_fp | sentinel | sentinel | engine_fp",
                "the sentinel token or its placement has changed, so every "
                "published run_fp for an ablation run is unreproducible even "
                "where the stages that DID run are untouched.")


# ---------------------------------------------------------------------------
# End to end for the extractor role, registry included
# ---------------------------------------------------------------------------

class TestExtractorFingerprintEndToEnd:
    """The extractor's `config_fp` exactly as a run against
    `tests/fixtures/config_synthetic/` records it, provider-call identity block
    included.

    That block is direktoro's, so this is the one place a golden here depends
    on a package outside this repo. It is pinned anyway, because a registry
    edit that changes a base_url, a provider, or the wire key a decoding
    parameter rides under moves every published stage fingerprint without
    touching a byte of config. That is precisely the silent break these goldens
    exist to make loud, and it should be read when it fires rather than
    recomputed."""

    def test_the_call_identity_block_for_the_fixture_extractor_model(
            self, bundle):
        assert _extractor_call_identity(bundle) == \
            GOLDEN_EXTRACTOR_CALL_IDENTITY, (
                "direktoro's provider-call identity block for "
                f"{bundle.pipeline['extractor_model']} has changed shape or "
                "content (model, provider, base_url, route, or the resolved "
                "decoding params and the wire keys they ride under). It is "
                "the first component of config_fp, checker_fp, review_fp and "
                "every call_fp, so every fingerprint this repo has published "
                "is now unreproducible even though no config file changed. "
                "Establish what moved in the registry before touching this "
                "golden.")

    def test_config_fp_a_run_against_the_fixture_bundle_records(
            self, bundle, template):
        fp = config_fingerprint(
            _extractor_call_identity(bundle),
            _extractor_prompt_hash(bundle, template),
            bundle.template_hash,
            tool_set_hash=tool_set_hash(all_tool_definitions(template)),
            structure_hash=_pipeline_structure(
                bundle,
                supports_images=model_info(
                    bundle.pipeline["extractor_model"]).supports_images),
            reference_hash=bundle.reference_lists_hash,
        )
        assert fp == GOLDEN_RUN_CONFIG_FP, _moved(
            "the extractor config_fp a real run records",
            "direktoro's call identity | prompt_hash | template_hash | "
            "tool_set_hash | structure | reference_lists_hash",
            "a run of this config bundle no longer lands on the fingerprint "
            "it used to, so its session directory, its resume check and every "
            "downstream key change together. When this fails and the "
            "composition test above passes, the cause is the call-identity "
            "block or the fixture's pipeline.yaml, not meltiro's fold.")
