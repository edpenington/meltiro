"""Meltiro: agentic LLM-assisted data extraction for systematic reviews.

The names re-exported here are the stable, library-first surface a downstream
consumer imports. Import them from the package
root, `from meltiro import validate_value`, not from their defining modules:
the deep module paths are internal and may move between releases, while this
surface is what the consumer contract pins.

Grouped by what they do:

  - Validation: `validate_value` (single-field, the per-save entry point),
    `validate_extraction_output` (batch re-validation), and `ValidationResult`.
  - Config bundle: `load_config_bundle` and the frozen `ConfigBundle` it
    returns (carrying the content fingerprints a consumer pins).
  - Paper bundle: `load_bundle` and the frozen `PaperBundle` it returns. The
    format itself belongs to `alteksto`, a declared dependency of this
    package: it specifies what a bundle is, validates one, and is where a
    bundle is built. A consumer wanting the verdict without the loading
    imports `alteksto.bundle.validate_bundle` from there, which is the same
    call `load_bundle` refuses behind.
  - Run statuses: the `RUN_STATUSES` vocabulary plus the `TERMINAL_STATUSES`
    and `VALIDATED_STATUSES` subsets.

The model registry is not re-exported here. It belongs to the shared
`direktoro` package, a declared dependency of this one, and a consumer that
wants it imports it from there.

Everything above is importable with direktoro absent. That is what lets a
bundle-reading consumer install this wheel `--no-deps` — a way of skipping the
PROVIDER LAYER (direktoro and the provider SDKs it brings), not a claim that
the package has no dependencies. `alteksto` and `pyyaml` are still required by
this import itself (`meltiro.bundle` and `meltiro.reference_lists`), and
`python-dotenv` by the CLI, so such a consumer installs those three by hand.
`meltiro.cli` imports direktoro at module scope and is outside the promise:
the contract is about `import meltiro`, not about the command.
"""

from meltiro.bundle import PaperBundle, load_bundle
from meltiro.config_bundle import ConfigBundle, load_config_bundle
from meltiro.statuses import (
    RUN_STATUSES,
    TERMINAL_STATUSES,
    VALIDATED_STATUSES,
)
from meltiro.validators import (
    ValidationResult,
    validate_extraction_output,
    validate_value,
)

__version__ = "0.4.0"

__all__ = [
    "__version__",
    # Validation
    "validate_value",
    "validate_extraction_output",
    "ValidationResult",
    # Config bundle
    "load_config_bundle",
    "ConfigBundle",
    # Paper bundle
    "load_bundle",
    "PaperBundle",
    # Run statuses
    "RUN_STATUSES",
    "TERMINAL_STATUSES",
    "VALIDATED_STATUSES",
]
