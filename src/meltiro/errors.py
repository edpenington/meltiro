"""Exceptions for the agentic extraction pipeline."""


class AgenticExtractionError(Exception):
    """Base class for all errors raised by this package."""


class ValidationFailure(AgenticExtractionError):
    """Raised by a tool dispatcher when a tool call fails validation.

    Carries the structured errors list that will be returned to the model
    in the tool_result block.
    """

    def __init__(self, errors):
        # errors is a list of {path, code, message} dicts
        self.errors = errors
        super().__init__(f"{len(errors)} validation error(s)")


class CheckerError(AgenticExtractionError):
    """Raised when the checker LLM fails (API error after retries, or
    response can't be parsed)."""


class SessionError(AgenticExtractionError):
    """Raised on session-state problems: corrupt jsonl, config_fp mismatch on
    resume, missing diagnostics/run.json, etc."""


class ResumeRefused(SessionError):
    """Raised when a resume is requested but cannot proceed safely (status not
    in_progress, config_fp mismatch, etc.). The message explains why."""


class BundleError(AgenticExtractionError):
    """Raised when a paper bundle cannot be loaded.

    Carries the full list of problems (one string per problem) so the
    caller sees every issue at once rather than the first that trips.
    """

    def __init__(self, problems, path=None):
        self.problems = list(problems)
        self.path = path
        prefix = f"Invalid paper bundle at {path}: " if path else \
            "Invalid paper bundle: "
        super().__init__(prefix + "; ".join(self.problems))


class ThinkingConfigError(AgenticExtractionError):
    """Raised when a role's thinking / reasoning-effort config cannot work.

    Three faults share this class, all refused at startup before any spend: a
    mode or effort meltiro does not accept, a shape the model's endpoint would
    reject with a 400 (direktoro's `ThinkingUnsupported`, re-raised with the
    role attached), and an output cap too small to hold a think plus its
    answer. See `meltiro.thinking`.
    """


class RatesConfigError(AgenticExtractionError):
    """Raised when `pipeline.yaml`'s `rates:` block cannot price a run.

    A present-but-unusable block is refused at startup, before any spend.
    Omitting `rates:` entirely is not a fault: each role then takes its
    rates from direktoro's price table, or runs unpriced. See
    `meltiro.rates` for the block's shape and the absence semantics.
    """


class ConfigBundleError(AgenticExtractionError):
    """Raised when a config bundle cannot be loaded or validated.

    Carries the full problem list (one string per problem)."""

    def __init__(self, problems, path=None):
        self.problems = list(problems)
        self.path = path
        prefix = f"Invalid config bundle at {path}: " if path else \
            "Invalid config bundle: "
        super().__init__(prefix + "; ".join(self.problems))
