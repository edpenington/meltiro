"""The USD rate card each role prices its calls against.

A run puts three roles to work — extractor, checker, reviewer — and each runs
its own model, so pricing is per role: the extractor's calls are costed at the
extractor's rates and at nothing else's. A role's card comes from one of two
places. The operator writes it under `rates:` in `pipeline.yaml`, keyed by
role. Or direktoro's dated price table supplies it, from the rate the vendor
published on the day the table's entry was read. Either way the card carries
its own provenance — which of the two it came from, the date behind it, and
which version of the table data — and every figure a run records sits beside
the exact rates and provenance that produced it. A reader recomputes the figure
from the token counters and the card in the same record, and either gets the
same number or knows precisely why not.

A role with neither an operator card nor a table entry runs unpriced. It
records its token counters — which come from the provider and cannot rot — and
NO dollar figure at all. Not a zero: a zero reads as "this was free", which is
the one wrong answer available here. One such role also withholds the run's
total, because a sum over the rest of the run would read as the whole of it.

Rates are commercial, not methodological. Two runs differing only in the numbers
the operator typed put exactly the same questions to exactly the same models
over exactly the same paper, so a rate card reaches no fingerprint preimage. It
is not part of the instrument, and a fingerprint that moved with it would report
a methodological difference where there is none.
"""

from dataclasses import dataclass

from meltiro.errors import RatesConfigError


# The roles a run prices, which are the keys a `rates:` block takes. Each names
# the role whose model it prices, and matches that role's `*_model` key in
# `pipeline.yaml`.
ROLE_KEYS = ("extractor", "checker", "review")

# One role's rate block. Each key is USD per MILLION tokens of the counter it
# names, the unit every provider publishes its prices in, so an operator
# transcribes a price list rather than converting one.
RATE_KEYS = ("input_per_1m", "output_per_1m",
             "cache_read_per_1m", "cache_write_per_1m")

# pipeline.yaml key -> the token counter direktoro's `cost_from_rates` keys its
# rate mapping by. The translation lives here alone, so the name an operator
# writes and the name the costing function reads cannot drift apart. It is read
# in both directions: outward to cost a call, inward to build a card from a
# price-table entry.
_COUNTER = {
    "input_per_1m": "input",
    "output_per_1m": "output",
    "cache_read_per_1m": "cache_read",
    "cache_write_per_1m": "cache_write",
}

# The shape of a valid block, spelled out once so every refusal below describes
# the same thing.
_SHAPE = (f"`rates:` maps a role to its own card: {', '.join(ROLE_KEYS)}, "
          f"each carrying {', '.join(RATE_KEYS)} under it")


@dataclass(frozen=True)
class Rates:
    """One role's rate card: USD per million tokens, per token counter.

    All four rates are required together. A partial card would price a role
    correctly or incorrectly depending on which counters happened to be
    non-zero, so the completeness of the recorded figure would be an accident of
    the traffic rather than a property of the record. A provider with no prompt
    cache tier is expressed as `0.0` for both cache rates, which is what it
    charges; the run then honours the zero rather than treating it as absent.

    The last three fields are the card's provenance, recorded with it wherever
    it goes. `source` is `"operator"` for a card written in `pipeline.yaml` and
    `"table"` for one direktoro's price table supplied. A table card also
    carries `as_of`, the ISO date the vendor's page was read, and
    `table_version`, which names the table data that reading came from, so a
    figure priced from the table says exactly which table priced it. An
    operator card carries neither: its provenance is the bundle it was written
    in, which the run already records.
    """

    input_per_1m: float
    output_per_1m: float
    cache_read_per_1m: float
    cache_write_per_1m: float
    source: str = "operator"
    as_of: str = None
    table_version: int = None

    @classmethod
    def from_table(cls, entry, table_version):
        """A card built from a direktoro `PriceEntry`, dates and all.

        The entry hands its rates over keyed by token counter, and `_COUNTER`
        translates them into the key names an operator writes, so a card has one
        shape whichever source produced it and the translation stays in one
        place.
        """
        rates = entry.as_rates()
        return cls(
            source="table", as_of=entry.as_of, table_version=table_version,
            **{key: float(rates[counter])
               for key, counter in _COUNTER.items()},
        )

    def as_record(self):
        """The rate card in the shape it is written into a run's artefacts.

        The four rates under the same key names an operator writes in
        `pipeline.yaml`, so the block in the record and the block in the bundle
        are read the same way and can be compared by eye, followed by the
        provenance that says where they came from. Every key is always present:
        `as_of` and `table_version` are written as null on an operator card
        rather than left out, so a reader never has to tell "no date" apart
        from "no key". Recorded next to every cost figure derived from it.
        """
        record = {key: getattr(self, key) for key in RATE_KEYS}
        record["source"] = self.source
        record["as_of"] = self.as_of
        record["table_version"] = self.table_version
        return record

    def cost_of_call(self, *, input_tokens=0, output_tokens=0,
                     cache_read_tokens=0, cache_write_tokens=0):
        """The USD cost of one call's token counters under this rate card.

        direktoro owns the arithmetic (and refuses a non-zero counter it has no
        rate for), so the same counters cost the same everywhere and this module
        holds only the translation from the operator's key names to its own.
        Imported lazily, like every other direktoro use below the CLI, so
        `import meltiro` never needs the provider layer.
        """
        from direktoro import cost_from_rates
        return cost_from_rates(
            rates={_COUNTER[key]: getattr(self, key) for key in RATE_KEYS},
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )


def parse_rates(pipeline):
    """The rate cards a pipeline mapping configures, as `{role: Rates}`.

    Only the roles the `rates:` block names appear in the result. `rates:`
    absent gives `{}`: every role then takes its card from direktoro's price
    table, or runs unpriced where the table has no entry for its model.

    Present-but-unusable is a different thing and is refused loudly here, before
    any spend — a bare `rates:` with nothing under it, a four-key card written
    at the top level instead of under a role, an unknown role name, a role block
    missing a rate or carrying an unknown key, a negative or non-numeric value.
    Each of those would otherwise leave an operator believing a role was priced
    when it was not, or priced against something other than what they wrote.

    Every value is read for presence with `is None`, never for truth, so a
    legitimate `0.0` rate is honoured rather than swallowed as absent.

    Raises `RatesConfigError` naming every problem in the block at once.
    """
    if "rates" not in pipeline:
        return {}
    block = pipeline["rates"]
    if not isinstance(block, dict):
        raise RatesConfigError(
            f"pipeline.yaml's `rates:` must be a mapping, got "
            f"{type(block).__name__}. {_SHAPE}. Remove the key entirely to "
            f"take every role's rates from direktoro's price table.")
    if not block:
        raise RatesConfigError(
            f"pipeline.yaml's `rates:` block is empty. {_SHAPE}. Remove the "
            f"key entirely to take every role's rates from direktoro's price "
            f"table.")
    flat = [key for key in RATE_KEYS if key in block]
    if flat:
        raise RatesConfigError(
            f"pipeline.yaml's `rates:` names the rate key(s) "
            f"{', '.join(repr(k) for k in flat)} directly under it. Each role "
            f"runs its own model and so prices its own calls: {_SHAPE}.")
    unknown_roles = sorted(k for k in block if k not in ROLE_KEYS)
    if unknown_roles:
        raise RatesConfigError(
            f"pipeline.yaml's `rates:` names unknown role(s) "
            f"{', '.join(repr(k) for k in unknown_roles)}. A run prices three "
            f"roles: {', '.join(ROLE_KEYS)}.")

    cards = {}
    problems = []
    for role in ROLE_KEYS:
        if role not in block:
            continue
        card = _parse_role_card(role, block[role], problems)
        if card is not None:
            cards[role] = card
    if problems:
        raise RatesConfigError(
            "pipeline.yaml's `rates:` block is invalid: "
            + "; ".join(problems) + ".")
    return cards


def _parse_role_card(role, block, problems):
    """One role's four-key block as a `Rates`, or None once it has a fault.

    Every fault found is appended to `problems`, named with its role, so a
    bundle with two bad cards reports both faults in both cards at once rather
    than one per attempted run.
    """
    listed = ", ".join(RATE_KEYS)
    if not isinstance(block, dict):
        problems.append(
            f"{role}: must be a mapping of {listed}, got "
            f"{type(block).__name__}")
        return None
    before = len(problems)
    unknown = sorted(k for k in block if k not in RATE_KEYS)
    if unknown:
        problems.append(
            f"{role}: unknown key(s) {', '.join(repr(k) for k in unknown)}; a "
            f"role's card takes exactly {listed}")
    missing = [k for k in RATE_KEYS if block.get(k) is None]
    if missing:
        problems.append(
            f"{role}: missing rate(s) {', '.join(missing)}; all four are "
            f"required together (a model with no prompt-cache tier takes 0.0 "
            f"for the cache rates), so a recorded cost is never complete only "
            f"by accident")
    for key in RATE_KEYS:
        value = block.get(key)
        if value is None:
            continue  # already reported as missing
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(
                f"{role}: {key} must be a number (USD per million tokens), "
                f"got {value!r}")
        elif value < 0:
            problems.append(
                f"{role}: {key} must be zero or positive, got {value!r}")
    if len(problems) > before:
        return None
    return Rates(**{key: float(block[key]) for key in RATE_KEYS})
