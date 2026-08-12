"""Exceptions for the agentic extraction pipeline, and the reports two roles
must word identically.

`truncation_report` lives here because a cut-off response is a warning in one
role and a raised `CheckerError` in another, and the sentence an operator reads
has to be the same either way; this is the module both sides already import.
"""


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
    """Raised when one checker call cannot produce a verdict.

    `spent` carries the usage of any provider calls that were BILLED before
    the failure, as the same four counters a verdict reports
    (`input_tokens`, `output_tokens`, `cache_creation_tokens`,
    `cache_read_tokens`) plus `responses`, how many billed calls there were.
    It is empty for a failure that reached no provider (a missing key) or
    whose every attempt errored, and populated for one that got an answer and
    could not use it — a truncated reply, a verdict outside the vocabulary, or
    a run of replies that called no tool. Those calls cost money, and a run
    that reported them as free would understate its own spend;
    `run_checker_batch` reads this to price the degraded field honestly.

    A routed call whose response carried no charge of its own adds
    `cost_incomplete` and `unreceipted_responses` to the mapping: the dollar
    figure then covers the receipts there were, and says how many calls it
    does not cover. Costing a failure must not itself fail — a batch of paid
    sibling verdicts hangs on it — so the absence is recorded here rather than
    raised.
    """

    _NO_SPEND = {
        "responses": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
    }

    @classmethod
    def no_spend(cls):
        """The counters for a failure that billed nothing.

        Here rather than spelled again at each call site, so a caller that
        degrades a field WITHOUT a CheckerError in hand (a plumbing fault that
        arrived as some other exception) prices it in the same shape this
        class carries.
        """
        return dict(cls._NO_SPEND)

    def __init__(self, message, spent=None):
        self.spent = dict(spent) if spent else self.no_spend()
        super().__init__(message)


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


# ---------------------------------------------------------------------------
# Shared reports
# ---------------------------------------------------------------------------

def truncation_report(cap, key):
    """The one sentence a response that stopped on its output cap gets.

    `cap` is the number in force and `key` the pipeline.yaml key that set it,
    which names the role by its prefix and is the line an operator would edit.
    One wording for every role: a truncated extractor turn is a warning and a
    truncated checker reply is a raised error, and two spellings of the same
    fact would read as two different faults in a run's warnings.
    """
    role = key.removesuffix("_max_tokens")
    return (f"{role} response stopped at the max_tokens cap ({cap}, set by "
            f"{key} in pipeline.yaml), so the output is incomplete.")
