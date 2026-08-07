"""Tests for the `meltiro validate` and `meltiro fingerprint` subcommands.

Both are thin wrappers over the importable functions (validate_value via
validate_extraction_output, and load_config_bundle). No network, no API key.
"""

import json

import pytest

from meltiro import cli
from meltiro.config_bundle import load_config_bundle


def _run(argv):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    code = excinfo.value.code
    return 0 if code is None else code


def _write_output(tmp_path, obj):
    path = tmp_path / "extraction_output.json"
    path.write_text(json.dumps(obj), encoding="utf-8")
    return str(path)


# A real gauge canonical name from the shipped reference list.
_REAL_GAUGE = "Bracket Load Rating (BLR)"


class TestValidateCommand:
    def test_all_valid_exit_0(self, tmp_path, config_dir, capsys):
        # Evidence on every required-evidence field, because the default
        # producer kind is `llm` and holds them to the template's flag. The
        # quotes are not adjudicated here (no --paper), which the verdict says.
        out = _write_output(tmp_path, {
            # Keyed by the role that recorded it, as a real run writes it.
            "initial_check": {"extractor": {"text_readable": True,
                                            "figure_tables_included": True}},
            "study": {"widget_class": {"value": "load-bearing bracket",
                                       "evidence": "<q>a bracket</q>"}},
            "records": [{
                "record_id": "relationship_1",
                "gauge": {"value": _REAL_GAUGE, "evidence": "<q>the BLR</q>"},
                "outcome_category": {"value": "service life",
                                     "evidence": "<q>service life</q>"},
            }],
            "quality_check": {},
        })
        code = _run(["validate", "--config", str(config_dir), out])
        printed = capsys.readouterr().out
        assert code == 0
        assert "OK:" in printed

    def test_off_list_reference_and_bad_option_exit_1(
            self, tmp_path, config_dir, capsys):
        out = _write_output(tmp_path, {
            "records": [{
                "record_id": "relationship_1",
                "gauge": {"value": "Totally Made Up Tool", "evidence": None},
                "outcome_category": {"value": "Not A Real Category",
                                     "evidence": None},
            }],
        })
        code = _run(["validate", "--config", str(config_dir), out])
        printed = capsys.readouterr().out
        assert code == 1
        assert "off_list_reference" in printed
        assert "invalid_option" in printed
        assert "record.relationship_1.gauge" in printed

    def test_unknown_field_reported(self, tmp_path, config_dir, capsys):
        out = _write_output(tmp_path, {
            "study": {"not_a_real_field": {"value": "x", "evidence": None}},
        })
        code = _run(["validate", "--config", str(config_dir), out])
        printed = capsys.readouterr().out
        assert code == 1
        assert "unknown_field" in printed

    def test_volunteered_quote_not_checked_without_paper(
            self, tmp_path, config_dir, capsys):
        # With no paper bundle there is nothing to check a quote against, so
        # no verbatim verdict is reported: reporting one would fail the quote
        # for an argument the caller omitted rather than for anything wrong
        # with the extraction.
        out = _write_output(tmp_path, {
            "study": {"widget_class": {
                "value": "bracket",
                "evidence": "<q>xyzzy definitely not in the paper</q>"}},
        })
        code = _run(["validate", "--config", str(config_dir), out])
        assert code == 0

    def test_volunteered_quote_checked_with_paper(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys):
        out = _write_output(tmp_path, {
            "study": {"widget_class": {
                "value": "bracket",
                "evidence": "<q>xyzzy definitely not in the paper</q>"}},
        })
        code = _run(["validate", "--config", str(config_dir), out,
                     "--paper", str(bundle_minimal_dir)])
        printed = capsys.readouterr().out
        assert code == 1
        assert "quote_not_in_text" in printed

    def test_bad_json_exit_1(self, tmp_path, config_dir, capsys):
        path = tmp_path / "broken.json"
        path.write_text("{ not json", encoding="utf-8")
        code = _run(["validate", "--config", str(config_dir), str(path)])
        err = capsys.readouterr().err
        assert code == 1
        assert "could not read extraction output" in err


class TestTheEvidenceContractIsReachable:
    """`meltiro validate` must be able to re-verify the evidence contract.

    The gate is real in `validate_envelope`, which the live extraction runs
    through. Re-validation is the third party's copy of that check: an auditor
    holding a published extraction output has no other way to ask whether a
    value the engine asserted was ever supported. A command that cannot put
    the question is worse than no command, because it answers `OK`.
    """

    # An `evidence: required` study field asserting a value with nothing
    # behind it: the exact shape a fabricated extraction takes.
    _UNSUPPORTED = {
        "study": {"widget_class": {"value": "load-bearing bracket",
                                   "evidence": None, "notes": None}},
    }

    def test_unsupported_value_fails_by_default(
            self, tmp_path, config_dir, capsys):
        out = _write_output(tmp_path, dict(self._UNSUPPORTED))
        code = _run(["validate", "--config", str(config_dir), out])
        printed = capsys.readouterr().out
        assert code == 1
        assert "evidence_required" in printed
        assert "study.widget_class" in printed

    def test_the_default_producer_is_llm(self, tmp_path, config_dir, capsys):
        # Stating the default explicitly must not change the verdict: the
        # command an auditor types with no flags is the enforcing one.
        out = _write_output(tmp_path, dict(self._UNSUPPORTED))
        assert _run(["validate", "--config", str(config_dir), out,
                     "--producer", "llm"]) == 1

    def test_human_producer_demands_no_evidence(
            self, tmp_path, config_dir, capsys):
        # Hand-authored comparison data: nobody promised evidence for it, so
        # the same file passes on its values alone.
        out = _write_output(tmp_path, dict(self._UNSUPPORTED))
        code = _run(["validate", "--config", str(config_dir), out,
                     "--producer", "human"])
        assert code == 0

    def test_the_verdict_names_what_was_and_was_not_checked(
            self, tmp_path, config_dir, capsys):
        out = _write_output(tmp_path, {
            "study": {"widget_class": {"value": "bracket",
                                       "evidence": "<q>a bracket</q>"}},
        })
        _run(["validate", "--config", str(config_dir), out])
        printed = capsys.readouterr().out
        # The evidence contract ran; the quotes were not checked against any
        # paper, and the verdict says so rather than claiming the file whole.
        assert "checked: the evidence contract" in printed
        assert "not checked" in printed and "--paper" in printed

    def test_human_verdict_says_the_contract_was_not_checked(
            self, tmp_path, config_dir, capsys):
        out = _write_output(tmp_path, dict(self._UNSUPPORTED))
        _run(["validate", "--config", str(config_dir), out,
              "--producer", "human"])
        printed = capsys.readouterr().out
        assert "not checked: the evidence contract" in printed
        assert "--producer llm" in printed

    def test_both_halves_run_together_with_a_paper(
            self, tmp_path, config_dir, bundle_minimal_dir, capsys):
        # Presence and verbatim are separate questions and both are asked:
        # one field asserts a value with no evidence, another quotes text that
        # is in no paper.
        out = _write_output(tmp_path, {
            "study": {
                "widget_class": {"value": "bracket", "evidence": None},
                "design": {"value": "cohort",
                           "evidence": "<q>xyzzy not in the paper</q>"},
            },
        })
        code = _run(["validate", "--config", str(config_dir), out,
                     "--paper", str(bundle_minimal_dir)])
        printed = capsys.readouterr().out
        assert code == 1
        assert "evidence_required" in printed
        assert "quote_not_in_text" in printed

    def test_an_unknown_producer_is_refused_by_the_parser(
            self, tmp_path, config_dir):
        out = _write_output(tmp_path, dict(self._UNSUPPORTED))
        assert _run(["validate", "--config", str(config_dir), out,
                     "--producer", "robot"]) == 2


class TestFingerprintCommand:
    def test_prints_components_and_family(self, config_dir, capsys):
        code = _run(["fingerprint", "--config", str(config_dir)])
        printed = capsys.readouterr().out
        assert code == 0
        payload = json.loads(printed)
        cb = load_config_bundle(config_dir)
        assert payload["template_hash"] == cb.template_hash
        assert payload["reference_lists_hash"] == cb.reference_lists_hash
        assert payload["prompts_hash"] == cb.prompts_hash
        assert payload["instrument_fp"] == \
            cb.instrument_fp

    def test_does_not_print_config_fp_and_says_why(self, config_dir, capsys):
        code = _run(["fingerprint", "--config", str(config_dir)])
        printed = capsys.readouterr().out
        assert code == 0
        payload = json.loads(printed)
        assert "config_fp" not in payload
        assert "model" in payload["note"]

    def test_output_is_stable(self, config_dir, capsys):
        assert _run(["fingerprint", "--config", str(config_dir)]) == 0
        first = capsys.readouterr().out
        assert _run(["fingerprint", "--config", str(config_dir)]) == 0
        second = capsys.readouterr().out
        # Non-empty and a real payload, so a command that printed nothing at
        # all could not satisfy the equality below.
        assert json.loads(first)["instrument_fp"]
        assert first == second

    def test_bad_config_exit_1(self, tmp_path, capsys):
        code = _run(["fingerprint", "--config", str(tmp_path / "nope")])
        err = capsys.readouterr().err
        assert code == 1
        assert "does not exist" in err
