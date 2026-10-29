"""Clinic labels resolve to CLINIC, not to the STORAGE ROOM fallback.

Each of these already had an OCR pattern (OCRConfig.room_name_patterns) and
no taxonomy entry, so it became a room candidate and then fell through
normalize_to_mandatory's least-intrusive default. room_type is an SFT
training target, so the fallback was teaching a wrong class.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing_annotations.automation.taxonomy import (
    normalize_to_mandatory,
    resolve_context_dependent_types,
)
from preprocessing_annotations.config import PipelineConfig


@pytest.mark.parametrize("label", [
    "EXAM RM", "EXAM RM 2", "EXAM ROOM", "TRIAGE", "TRIAGE 1",
    "NURSE STATION", "DRAW", "DRAW 2", "PROCEDURE",
])
def test_clinic_label_resolves_to_clinic(label):
    assert normalize_to_mandatory(label) == "CLINIC"


@pytest.mark.parametrize("label", ["exam rm", "Draw", "Triage"])
def test_clinic_label_is_case_insensitive(label):
    assert normalize_to_mandatory(label) == "CLINIC"


@pytest.mark.parametrize("label", ["STORAGE", "STOR", "STORAGE ROOM", "VAULT"])
def test_real_storage_labels_are_unchanged(label):
    """The fallback's legitimate users must keep their class."""
    assert normalize_to_mandatory(label) == "STORAGE ROOM"


@pytest.mark.parametrize("label,expected", [
    # User directive 2026-10-27. CAFE/SHOWROOM/EQUIPMENT ROOM already
    # resolved correctly and are pinned here so they cannot drift.
    ("CAR WASH", "OTHER"),
    ("BAYS", "OTHER"),
    ("DETAIL BAYS", "OTHER"),
    ("ZOOM", "STORAGE ROOM"),
    ("Zoom 3", "STORAGE ROOM"),
    ("CAFE", "CAFETERIA"),
    ("SHOWROOM", "RETAIL"),
    ("EQUIPMENT ROOM", "EQUIPMENT ROOM"),
])
def test_dealership_and_retail_labels(label, expected):
    assert normalize_to_mandatory(label) == expected


def test_lab_takes_clinic_when_the_page_is_clinical():
    """LAB means a different room in a clinic than in a school, and the label
    alone cannot say which -- the page's other labels do."""
    names = ["EXAM RM", "LAB"]
    types = [normalize_to_mandatory(name) for name in names]

    assert resolve_context_dependent_types(names, types) == ["CLINIC", "CLINIC"]


def test_lab_keeps_its_own_class_on_a_school_page():
    names = ["CLASSROOM", "LAB"]
    types = [normalize_to_mandatory(name) for name in names]

    assert resolve_context_dependent_types(names, types) == ["CLASSROOM", "CLASSROOM"]


def test_lab_alone_is_not_evidence_for_itself():
    """A page whose only clinical hint is the ambiguous label stays as-is."""
    assert resolve_context_dependent_types(["LAB"], ["CLASSROOM"]) == ["CLASSROOM"]


def test_pages_without_ambiguous_labels_are_untouched():
    names = ["EXAM RM", "CORRIDOR"]
    types = [normalize_to_mandatory(name) for name in names]

    assert resolve_context_dependent_types(names, types) == types


@pytest.mark.parametrize("label", [
    "EXAM RM", "EXAM RM 2", "TRIAGE", "TRIAGE 1", "NURSE STATION",
    "DRAW", "DRAW 2", "PROCEDURE",
])
def test_every_mapped_clinic_label_still_has_an_ocr_pattern(label):
    """The two halves ship together: a taxonomy entry whose label no pattern
    produces is dead, and a pattern with no entry is what caused this bug."""
    patterns = PipelineConfig().ocr.room_name_patterns
    assert any(re.match(pattern, label, re.IGNORECASE) for pattern in patterns)
