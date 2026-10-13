import pytest
from spacy.lang.en import English
from spacy.tokens import Doc, Span, SpanGroup

from medspacy_io.reader import BratDocReader
from medspacy_io.writer import BratWriter


def test_exports_entities_groups_overlap_and_deduplicates(tmp_path):
    doc = English()("red apple pie")
    doc.ents = [Span(doc, 1, 2, label="Food")]
    doc.spans["foods"] = SpanGroup(
        doc, spans=[Span(doc, 1, 2, label="Food"), Span(doc, 0, 2, label="Food")]
    )
    doc.spans["Dessert"] = SpanGroup(doc, spans=[Span(doc, 1, 3)])
    doc.spans["Fruit"] = SpanGroup(doc, spans=[Span(doc, 1, 2, label="Fruit")])
    BratWriter().write_docs_to_brat([doc], ["sample"], tmp_path)

    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == doc.text
    assert (tmp_path / "sample.ann").read_text(encoding="utf-8").splitlines() == [
        "T1\tFood 0 9\tred apple",
        "T2\tFood 4 9\tapple",
        "T3\tFruit 4 9\tapple",
        "T4\tDessert 4 13\tapple pie",
    ]
    config = (tmp_path / "annotation.conf").read_text(encoding="utf-8")
    assert config == (
        "[entities]\nDessert\nFood\nFruit\n\n[relations]\n\n[events]\n\n[attributes]\n"
    )


def test_attributes_relations_and_collection_config(tmp_path):
    if not Span.has_extension("brat_test_flag"):
        Span.set_extension("brat_test_flag", default=False)
    if not Span.has_extension("brat_test_certainty"):
        Span.set_extension("brat_test_certainty", default=None)
    if not Doc.has_extension("relations"):
        Doc.set_extension("relations", default=[])

    doc = English()("Aspirin helps pain")
    source = Span(doc, 0, 1, label="Drug")
    target = Span(doc, 2, 3, label="Symptom")
    doc.ents = [source, target]
    source._.brat_test_flag = True
    source._.brat_test_certainty = "Certain"
    doc._.relations = [(source, target, "Treats")]

    second = English()("headache")
    second.ents = [Span(second, 0, 1, label="Symptom")]
    second.ents[0]._.brat_test_certainty = "Possible"

    writer = BratWriter(["brat_test_flag", "brat_test_certainty"])
    writer.write_docs_to_brat([doc, second], ["first", "second"], tmp_path)
    assert (tmp_path / "first.ann").read_text(encoding="utf-8").splitlines() == [
        "T1\tDrug 0 7\tAspirin",
        "T2\tSymptom 14 18\tpain",
        "A1\tbrat_test_certainty T1 Certain",
        "A2\tbrat_test_flag T1",
        "R1\tTreats Arg1:T1 Arg2:T2",
    ]
    config = (tmp_path / "annotation.conf").read_text(encoding="utf-8")
    assert "Treats Arg1:Drug, Arg2:Symptom" in config
    assert "brat_test_flag Arg:<ENTITY>" in config
    assert "brat_test_certainty Arg:<ENTITY>, Value:Certain|Possible" in config


def test_unlabeled_relation_endpoint_uses_unique_group(tmp_path):
    if not Doc.has_extension("relations"):
        Doc.set_extension("relations", default=[])
    doc = English()("red apple")
    source = Span(doc, 0, 1)
    target = Span(doc, 1, 2, label="Fruit")
    doc.spans["Color"] = [source]
    doc.ents = [target]
    doc._.relations = [(source, target, "Describes")]
    BratWriter().write_docs_to_brat([doc], ["sample"], tmp_path)
    assert "R1\tDescribes Arg1:T1 Arg2:T2" in (tmp_path / "sample.ann").read_text(
        encoding="utf-8"
    )


def test_preflight_rejects_existing_outputs_and_bad_relations(tmp_path):
    if not Doc.has_extension("relations"):
        Doc.set_extension("relations", default=[])
    doc = English()("one two")
    doc.ents = [Span(doc, 0, 1, label="Thing")]
    writer = BratWriter()
    writer.write_docs_to_brat([doc], ["first"], tmp_path)
    original = (tmp_path / "first.ann").read_bytes()
    with pytest.raises(FileExistsError):
        writer.write_docs_to_brat([doc], ["first"], tmp_path)
    assert (tmp_path / "first.ann").read_bytes() == original

    doc._.relations = [(doc.ents[0], Span(doc, 1, 2, label="Thing"), "Related")]
    with pytest.raises(ValueError, match="was not exported"):
        writer.write_docs_to_brat([doc], ["other"], tmp_path, overwrite=True)
    assert not (tmp_path / "other.ann").exists()


def test_overwrite_replaces_targets_but_preserves_unrelated_files(tmp_path):
    doc = English()("old")
    writer = BratWriter()
    writer.write_docs_to_brat([doc], ["sample"], tmp_path)
    (tmp_path / "keep.txt").write_text("keep", encoding="utf-8")
    new_doc = English()("new")
    new_doc.ents = [Span(new_doc, 0, 1, label="Thing")]
    writer.write_docs_to_brat([new_doc], ["sample"], tmp_path, overwrite=True)
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "new"
    assert "Thing" in (tmp_path / "sample.ann").read_text(encoding="utf-8")
    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_rejects_duplicate_ids_and_invalid_labels(tmp_path):
    doc = English()("one")
    with pytest.raises(ValueError, match="unique"):
        BratWriter().write_docs_to_brat([doc, doc], ["same", "same"], tmp_path)
    with pytest.raises(ValueError, match="unique"):
        BratWriter().write_docs_to_brat([doc, doc], ["Same", "same"], tmp_path)
    with pytest.raises(ValueError, match="Invalid report ID"):
        BratWriter().write_docs_to_brat([doc], ["../escape"], tmp_path)
    doc.ents = [Span(doc, 0, 1, label="Bad label")]
    with pytest.raises(ValueError, match="Invalid BRAT entity"):
        BratWriter().write_docs_to_brat([doc], ["sample"], tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_round_trip_simple_spans_and_attribute(tmp_path):
    if not Span.has_extension("Status"):
        Span.set_extension("Status", default=None)
    doc = English()("red apple pie")
    doc.ents = [Span(doc, 1, 2, label="Food")]
    doc.ents[0]._.Status = "Present"
    doc.spans["Food"] = [Span(doc, 0, 2, label="Food")]
    BratWriter(["Status"]).write_docs_to_brat([doc], ["sample"], tmp_path)
    read_doc = BratDocReader(
        nlp=English(), support_overlap=True, schema_file=tmp_path / "annotation.conf"
    ).read(tmp_path / "sample.txt")
    read_spans = {
        (span.start_char, span.end_char, span.label_) for span in read_doc.spans["Food"]
    }
    assert read_spans == {
        (0, 9, "Food"),
        (4, 9, "Food"),
    }
    assert any(span._.Status == "Present" for span in read_doc.spans["Food"])
