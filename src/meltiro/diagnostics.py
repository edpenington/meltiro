"""Diagnostics levels: how much of a run's deterministic record is kept.

A run produces two things, kept apart in the session directory: the
EXTRACTION OUTPUT (the validated, LLM-generated result, written at every
level as `extraction_output.json` at the top) and the DIAGNOSTICS (every
deterministic record of how the process went, under `diagnostics/`).

`--diagnostics` chooses how much of that record is kept. The three levels are
strict supersets of one another:

    minimal    extraction_output.json, diagnostics/run.json,
               diagnostics/field_history.json, diagnostics/transcript.md,
               diagnostics/tool_calls.jsonl
    standard   the above plus diagnostics/instrument/
    full       the above plus diagnostics/api_calls.jsonl

`tool_calls.jsonl` is in the minimum because it is the run's memory:
`Session.replay_messages` rebuilds the conversation from it, so a session
that lacks it cannot be resumed, and `field_history.json` is derived from it.
Nothing a level omits may ever be needed to resume a session or regenerate a
derived artefact.

A transcript is written and can be re-rendered at EVERY level: the run writes
one at every stop, and `meltiro transcript` re-renders any session afterwards.
What `minimal` costs is not the document but part of what it can show — the
instrument section, the prompts and tool definitions as sent, was never
written down, so a transcript rendered from such a session says the level
stopped it from showing them in place of that section.

The level is an operational choice, not methodology, so it enters no
fingerprint. It is recorded in `run.json` under `diagnostics`, beside `caps`,
so a reader of a finished session knows why a file is absent.
"""

from meltiro.errors import SessionError

# In ascending order, each a strict superset of the one before it.
DIAGNOSTICS_LEVELS = ("minimal", "standard", "full")

# The default: everything except the verbatim wire log. It keeps the
# instrument, so a transcript can be rendered from the session alone, without
# paying for a second copy of every request and response body.
DEFAULT_DIAGNOSTICS = "standard"


def validate_diagnostics(level):
    """Return `level` if it is a legal diagnostics level, else fail loudly.

    Used at every entry point that accepts one (the CLI, the Orchestrator,
    Session.create, Session.resume) so an unknown level can never reach a run
    and quietly behave like the default.
    """
    if level not in DIAGNOSTICS_LEVELS:
        raise SessionError(
            f"Unknown diagnostics level: {level!r}. Legal levels are "
            f"{', '.join(DIAGNOSTICS_LEVELS)}. A session records the level it "
            "ran under in run.json under `diagnostics`."
        )
    return level


def captures_instrument(level):
    """Whether this level captures the instrument (the rendered prompts, the
    tool definitions, and the exhibits the message attached) into
    `diagnostics/instrument/`."""
    return validate_diagnostics(level) in ("standard", "full")


def captures_api_calls(level):
    """Whether this level captures the verbatim wire log
    (`diagnostics/api_calls.jsonl`)."""
    return validate_diagnostics(level) == "full"
