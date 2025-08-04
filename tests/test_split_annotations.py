#!/usr/bin/env python3

"""Test script to verify the optimized split annotation handling"""
import pytest
import spacy
from pathlib import Path
import sys
import os

# Add the parent directory to the path to import medspacy_io
sys.path.insert(0, str(Path(__file__).parent.parent))

from medspacy_io.reader.ehost_reader import EhostDocReader


@pytest.fixture
def nlp():
    """Fixture to provide spacy nlp object"""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # If model not available, use blank model
        return spacy.blank("en")


@pytest.fixture
def test_data_paths():
    """Fixture to provide test data paths relative to project root"""
    project_root = Path(__file__).parent.parent
    return {
        "xml_file": project_root
        / "tests/data/ehost_split_anno/saved/doc2.txt.knowtator.xml",
        "txt_file": project_root / "tests/data/ehost_split_anno/corpus/doc2.txt",
        "schema_file": project_root
        / "tests/data/ehost_split_anno/config/projectschema.xml",
    }


def test_split_annotation(nlp, test_data_paths):
    """Test that EhostDocReader properly handles split annotations"""
    xml_file = test_data_paths["xml_file"]
    txt_file = test_data_paths["txt_file"]
    schema_file = test_data_paths["schema_file"]

    # Verify test files exist
    assert xml_file.exists(), f"Test XML file not found: {xml_file}"
    assert txt_file.exists(), f"Test text file not found: {txt_file}"
    assert schema_file.exists(), f"Test schema file not found: {schema_file}"

    # Create reader
    reader = EhostDocReader(
        nlp=nlp, schema_file=str(schema_file), store_anno_string=True
    )

    # Parse annotations (explicitly set sort_spans=False to get OrderedDict)
    result = reader.parse_to_dicts(str(xml_file), sort_spans=False)
    spans, classes, attributes, relations = result

    # Verify basic parsing worked
    assert len(spans) > 0, "No spans were parsed"
    assert len(classes) > 0, "No classes were parsed"

    # Look specifically for the split annotation (EHOST_Instance_17)
    assert "EHOST_Instance_17" in spans, "Split annotation EHOST_Instance_17 not found"

    split_span = spans["EHOST_Instance_17"]
    start, end = split_span[0], split_span[1]

    # Verify the split annotation spans correctly from 495 to 635
    assert start == 495, f"Expected start position 495, got {start}"
    assert end == 635, f"Expected end position 635, got {end}"

    # Verify span text is included when store_anno_string=True
    assert (
        len(split_span) == 3
    ), "Expected span to include text when store_anno_string=True"
    span_text = split_span[2]
    assert (
        span_text == "skin was closed at the level of the skin ... incised"
    ), f"Expected specific span text, got: {span_text}"

    # Verify the class is properly assigned
    assert "EHOST_Instance_17" in classes, "Split annotation class not found"
    assert (
        classes["EHOST_Instance_17"][0] == "CONCEPT"
    ), f"Expected class 'CONCEPT', got {classes['EHOST_Instance_17'][0]}"


def test_split_annotation_text_verification(nlp, test_data_paths):
    """Test that split annotation properly represents discontinuous text spans"""
    xml_file = test_data_paths["xml_file"]
    txt_file = test_data_paths["txt_file"]
    schema_file = test_data_paths["schema_file"]

    reader = EhostDocReader(
        nlp=nlp, schema_file=str(schema_file), store_anno_string=True
    )
    spans, classes, attributes, relations = reader.parse_to_dicts(
        str(xml_file), sort_spans=False
    )

    # Read the source text
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Get the split annotation
    split_span = spans["EHOST_Instance_17"]
    start, end = split_span[0], split_span[1]

    # According to the XML, this should cover two discontinuous spans:
    # <span start="628" end="635" /> = "incised"
    # <span start="495" end="535" /> = "skin was closed at the level of the skin"

    span1_text = text[495:535]  # First span
    span2_text = text[628:635]  # Second span

    assert (
        span1_text == "skin was closed at the level of the skin"
    ), f"First span text incorrect: {span1_text}"
    assert span2_text == "incised", f"Second span text incorrect: {span2_text}"

    # The full span should include everything from 495 to 635
    full_text = text[start:end]
    assert "skin was closed at the level of the skin" in full_text
    assert "incised" in full_text
    assert len(full_text) == 140, f"Expected full span length 140, got {len(full_text)}"


if __name__ == "__main__":
    # Allow running the test directly for debugging
    import sys

    pytest.main([__file__] + sys.argv[1:])
