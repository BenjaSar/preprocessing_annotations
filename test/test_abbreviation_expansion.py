"""Shared abbreviation expander (automation/abbreviations.py).

The boundary after an abbreviation is tested on the raw remainder. The old
per-class copies stripped first and then looked for a space, so a space could
never match and "<ABBREV> <WORD>" never expanded: "JAN" and "JAN 101" expanded
while "JAN CLOSET" did not, leaving a janitor closet classified as CLOSET.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.automation.abbreviations import expand_abbreviation
from preprocessing_annotations.automation.sft_validator import (
    SemanticRoomValidator,
    TaxonomyNormalizer,
)


@pytest.mark.parametrize("raw, expanded", [
    ("JAN CLOSET", "JANITOR CLOSET"),      # the defect: word after the abbreviation
    ("MECH #3", "MECHANICAL ROOM #3"),
    ("JAN", "JANITOR"),                    # forms that already worked
    ("JAN 101", "JANITOR 101"),
    ("BR 1", "BEDROOM 1"),
    ("BR1", "BEDROOM 1"),
])
def test_abbreviation_expands(raw, expanded):
    assert expand_abbreviation(raw) == expanded


@pytest.mark.parametrize("raw", ["BRIDGE", "CONFIDENT", "JANE"])
def test_letter_after_abbreviation_is_not_an_abbreviation(raw):
    """A longer word that merely starts with an abbreviation must not expand."""
    assert expand_abbreviation(raw) == raw


def test_single_letter_blocked_by_default():
    assert expand_abbreviation("K") == "K"
    assert expand_abbreviation("K", block_single_letter=False) != "K"


def test_unknown_name_is_returned_unchanged():
    assert expand_abbreviation("XYZZY HALL") == "XYZZY HALL"


def test_both_call_sites_use_the_shared_expander():
    for owner in (SemanticRoomValidator(), TaxonomyNormalizer()):
        assert owner._expand_abbreviation("JAN CLOSET") == "JANITOR CLOSET"


def test_jan_closet_now_classifies_as_a_janitor_closet():
    assert TaxonomyNormalizer().normalize("JAN CLOSET") == "JANITOR CLOSET"
