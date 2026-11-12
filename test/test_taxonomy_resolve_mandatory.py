"""taxonomy.resolve_mandatory: the resolution chain without its fallback.

normalize_to_mandatory must keep answering exactly as before -- the whole
pipeline types rooms through it -- while a caller that cannot afford a guess
(the SFT exporter) can now see that a name was never placed.
"""


import pytest

from preprocessing_annotations.automation.taxonomy import (
    DEFAULT_MANDATORY_TYPE,
    VALID_TYPES,
    normalize_to_mandatory,
    resolve_mandatory,
)

# Names measured in corpus_batch2_v2's SFT targets (2026-10-29), one per
# resolution path that reaches the default.
UNPLACEABLE = ["SUITE 200", "A1", "", "   ", "XQ77"]
PLACEABLE = {
    "STORAGE": "STORAGE ROOM",      # exact variant
    "STOR": "STORAGE ROOM",         # short token, exact variant
    "EXAM RM": "CLINIC",            # clinic entries added 2026-10-27
    "COMPACTOR": "STORAGE ROOM",    # via the extended table, not the fallback
    "CORRIDOR": "CORRIDOR",         # already mandatory
}


@pytest.mark.parametrize("raw", UNPLACEABLE)
def test_unplaceable_names_resolve_to_nothing(raw):
    assert resolve_mandatory(raw) is None


@pytest.mark.parametrize("raw,expected", sorted(PLACEABLE.items()))
def test_placeable_names_resolve_to_their_class(raw, expected):
    assert resolve_mandatory(raw) == expected


@pytest.mark.parametrize("raw", UNPLACEABLE + list(PLACEABLE))
def test_normalize_still_answers_with_a_valid_class(raw):
    """The wrapper's contract: always an SFT-compliant type."""
    answer = normalize_to_mandatory(raw)
    assert answer in VALID_TYPES
    assert answer == (resolve_mandatory(raw) or DEFAULT_MANDATORY_TYPE)


def test_the_default_is_a_real_class_not_a_sentinel():
    assert DEFAULT_MANDATORY_TYPE in VALID_TYPES
