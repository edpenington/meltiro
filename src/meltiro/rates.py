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
# transcribes a price list rather than converting one. All four are required
# together (see `Rates`).
RATE_KEYS = ("input_per_1m", "output_per_1m",
             "cache_read_per_1m", "cache_write_per_1m")

# The one OPTIONAL rate, and the reason it is optional rather than a fifth
# required key: a cache entry is written at one of two time-to-live tiers and
# billed differently for each, and this engine asks for neither of them
# explicitly — every `cache_control` it writes is the plain `ephemeral` marker,
# which is the 5-minute tier (`meltiro.prompt_builder`). So a run costed by
# `cache_write_per_1m` alone is correctly costed, and a card that omits this
# key is complete for the traffic this engine produces.
#
# It exists because a PROVIDER may still report 1-hour writes — a response
# carrying `cache_creation_1h_input_tokens` — and those tokens bill at a
# different multiple of the base input rate. Costing them at the 5-minute rate
# would understate the charge by nearly two fifths, silently. Without a rate
# for them the costing raises instead (direktoro's `cost_from_rates` refuses a
# counter it was handed tokens but no rate for), so the gap is loud rather than
# absorbed.
OPTIONAL_RATE_KEYS = ("cache_write_1h_per_1m",)

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
    "cache_write_1h_per_1m": "cache_write_1h",
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
    # Appended at the END, after the provenance fields rather than beside the
    # rate it belongs with, because this record is constructed positionally in
    # places: a field inserted in the middle would rebind every argument after
    # it silently, putting a provenance string where a rate goes. None means
    # the card states no 1-hour write rate, which is every card today.
    cache_write_1h_per_1m: float = None

    @classmethod
    def from_table(cls, entry, table_version):
        """A card built from a direktoro `PriceEntry`, dates and all.

        The entry hands its rates over keyed by token counter, and `_COUNTER`
        translates them into the key names an operator writes, so a card has one
        shape whichever source produced it and the translation stays in one
        place.

        A table entry carries the 5-minute write rate and no 1-hour one (see
        `PriceEntry.as_rates`), so `cache_write_1h_per_1m` stays None on a table
        card: only the counters the entry states are read, and a rate the table
        does not publish is not invented from one it does.
        """
        rates = entry.as_rates()
        return cls(
            source="table", as_of=entry.as_of, table_version=table_version,
            **{key: float(rates[counter])
               for key, counter in _COUNTER.items() if counter in rates},
        )

    def as_record(self):
        """The rate card in the shape it is written into a run's artefacts.

        The rates under the same key names an operator writes in
        `pipeline.yaml`, so the block in the record and the block in the bundle
        are read the same way and can be compared by eye, followed by the
        provenance that says where they came from. Every key is always present:
        `as_of`, `table_version` and `cache_write_1h_per_1m` are written as null
        rather than left out, so a reader never has to tell "no rate" apart from
        "no key". Recorded next to every cost figure derived from it.
        """
        record = {key: getattr(self, key)
                  for key in (*RATE_KEYS, *OPTIONAL_RATE_KEYS)}
        record["source"] = self.source
        record["as_of"] = self.as_of
        record["table_version"] = self.table_version
        return record

    def _rate_mapping(self):
        """This card as the counter-keyed mapping `cost_from_rates` takes.

        A rate the card does not state is OMITTED rather than passed as None or
        as zero. That is what makes the refusal work: `cost_from_rates` raises
        for a counter it was handed tokens but no rate for, and passing a zero
        would price those tokens at nothing instead — the under-reporting the
        whole arrangement exists to prevent.
        """
        return {_COUNTER[key]: getattr(self, key)
                for key in (*RATE_KEYS, *OPTIONAL_RATE_KEYS)
                if getattr(self, key) is not None}

    def cost_of_call(self, *, input_tokens=0, output_tokens=0,
                     cache_read_tokens=0, cache_write_tokens=0,
                     cache_write_1h_tokens=0):
        """The USD cost of one call's token counters under this rate card.

        direktoro owns the arithmetic (and refuses a non-zero counter it has no
        rate for), so the same counters cost the same everywhere and this module
        holds only the translation from the operator's key names to its own.
        Imported lazily, like every other direktoro use below the CLI, so
        `import meltiro` never needs the provider layer.

        CACHE WRITES ARE TWO COUNTERS because they are two prices.
        `cache_write_tokens` is the 5-minute tier, `cache_write_1h_tokens` the
        1-hour one, and a caller that folded them together would price whichever
        tier its rate was not read for at the wrong multiple. A run of this
        engine writes only 5-minute entries, so the second is normally zero;
        when a provider reports otherwise, a card with no 1-hour rate raises
        here rather than costing those tokens at nothing.
        """
        from direktoro import cost_from_rates
        return cost_from_rates(
            rates=self._rate_mapping(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_write_1h_tokens=cache_write_1h_tokens,
        )


def cache_write_split(usage):
    """One response's cache writes as `(five_minute_tokens, one_hour_tokens)`.

    A `NormalisedUsage` carries three numbers: the two per-TTL counters and
    their unsplit total. When the provider reports the split they agree, and
    this returns it. When it reports only the total — an Anthropic response
    whose `usage.cache_creation` object is absent — the two counters are zero
    and the total is the only figure there is.

    That remainder is attributed to the 5-MINUTE tier, and it is a reading
    rather than a guess: this engine writes only the plain `ephemeral`
    cache_control marker, which IS the 5-minute tier (see
    `meltiro.prompt_builder`), and asks for a 1-hour entry nowhere. direktoro
    refuses the same remainder in `cost_from_usage` precisely because IT cannot
    know which tier was asked for; the caller that asked can, and this is that
    caller.

    A counter that is absent or is not a whole number reads as zero, so a
    usage record synthesised by a consumer — or by a test — that carries only
    the fields it cares about is treated as one reporting no split rather than
    raising. Nothing is lost by that: an unread split leaves its tokens in the
    unattributed remainder, and the remainder is priced, so the total is
    covered either way.
    """
    def count(field):
        value = getattr(usage, field, 0)
        return value if isinstance(value, int) and not isinstance(
            value, bool) else 0

    five_minute = count("cache_creation_5m_input_tokens")
    one_hour = count("cache_creation_1h_input_tokens")
    unattributed = count("cache_creation_input_tokens") - five_minute - one_hour
    if unattributed > 0:
        five_minute += unattributed
    return five_minute, one_hour


def cost_with_coverage(cost, figure, missing):
    """One dollar figure, worded so its coverage travels with it.

    For the sites that print money where a call's charge could not be read off
    its response: the tokens were counted and the charge was not, so any sum
    over the rest covers fewer calls than were made. `figure` is the caller's
    own rendering of `cost` — the sites differ in precision and in markup, and
    each keeps its own — and `missing` is how many calls that figure leaves
    out, or None when even the count was not recorded.

    Three readings, and the wording is the whole of what tells them apart:

    - a figure with money in it is a FLOOR, and says `at least`, which is the
      difference between a small bill and an understated one;
    - a floor of exactly zero is not a floor any reader should be handed.
      Every receipt there was is already in it and it is still nothing, so it
      states that there was no receipted charge rather than `at least
      $0.0000`, which reads as a run that all but paid for itself;
    - a figure that states no money at all leaves the missing calls nothing to
      be a floor of, so they are named as missing from any figure — an
      unpriced run and a missing receipt are two separate gaps, and stacking
      `at least` on a null would claim a number nobody has.
    """
    calls = "some call(s)" if missing is None else f"{missing} call(s)"
    if not isinstance(cost, (int, float)):
        return (f"{figure} — {calls} returned no receipt and are missing "
                f"from any figure")
    if cost == 0:
        return f"no receipted charge ({calls} returned no receipt)"
    return f"at least {figure} ({calls} returned no receipt)"


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
    accepted = (*RATE_KEYS, *OPTIONAL_RATE_KEYS)
    if not isinstance(block, dict):
        problems.append(
            f"{role}: must be a mapping of {listed}, got "
            f"{type(block).__name__}")
        return None
    before = len(problems)
    unknown = sorted(k for k in block if k not in accepted)
    if unknown:
        problems.append(
            f"{role}: unknown key(s) {', '.join(repr(k) for k in unknown)}; a "
            f"role's card takes {listed}, optionally with "
            f"{', '.join(OPTIONAL_RATE_KEYS)}")
    missing = [k for k in RATE_KEYS if block.get(k) is None]
    if missing:
        problems.append(
            f"{role}: missing rate(s) {', '.join(missing)}; all four are "
            f"required together (a model with no prompt-cache tier takes 0.0 "
            f"for the cache rates), so a recorded cost is never complete only "
            f"by accident")
    # The optional rate is held to the same value rules as the required four —
    # a rate is a rate — and only its ABSENCE is allowed. Absent means the card
    # states no 1-hour write rate, and a call that reports 1-hour writes then
    # raises at costing time rather than being priced at the 5-minute rate or
    # at nothing.
    for key in accepted:
        value = block.get(key)
        if value is None:
            continue  # required: already reported as missing; optional: unset
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(
                f"{role}: {key} must be a number (USD per million tokens), "
                f"got {value!r}")
        elif value < 0:
            problems.append(
                f"{role}: {key} must be zero or positive, got {value!r}")
    if len(problems) > before:
        return None
    return Rates(**{key: float(block[key]) for key in accepted
                    if block.get(key) is not None})
