"""Write spaCy documents as BRAT standoff annotation collections."""

import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from spacy.tokens import Doc, Span


_TYPE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_REPORT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    "{}{}".format(prefix, number)
    for prefix in ("COM", "LPT")
    for number in range(1, 10)
}


class BratWriter:
    """Export continuous spans, selected attributes, and binary relations.

    All ``doc.ents`` and ``doc.spans`` groups are included. An unlabeled span
    uses its span group name as its BRAT entity type. Identical ranges with the
    same type are written once, including when present in both sources.
    """

    def __init__(self, span_attributes: Optional[Sequence[str]] = None):
        self.span_attributes = tuple(span_attributes or ())
        for name in self.span_attributes:
            self._check_type_name(name, "attribute")
            if not Span.has_extension(name):
                raise ValueError("Span extension {!r} is not registered".format(name))
        if len(set(self.span_attributes)) != len(self.span_attributes):
            raise ValueError("span_attributes contains duplicate names")

    @staticmethod
    def _check_type_name(name: str, kind: str) -> None:
        if not isinstance(name, str) or not _TYPE_NAME.fullmatch(name):
            raise ValueError("Invalid BRAT {} name: {!r}".format(kind, name))

    @staticmethod
    def _span_key(span: Span, group_name: Optional[str] = None):
        label = span.label_ or group_name
        if not label:
            raise ValueError("Unlabeled span in doc.ents has no BRAT entity type")
        BratWriter._check_type_name(label, "entity")
        if span.start_char >= span.end_char:
            raise ValueError("Cannot export an empty span")
        return (span.start_char, span.end_char, label)

    def _attribute_values(self, span: Span):
        values = {}
        for name in self.span_attributes:
            value = getattr(span._, name)
            if (
                value is None
                or value is False
                or (isinstance(value, str) and not value)
            ):
                continue
            if value is True:
                values[name] = True
            elif isinstance(value, str) and _TYPE_NAME.fullmatch(value):
                values[name] = value
            else:
                raise ValueError(
                    "Attribute {!r} has unsupported BRAT value {!r}".format(
                        name, value
                    )
                )
        return values

    def _render_doc(self, doc: Doc):
        spans = {}
        for span, group_name in [(span, None) for span in doc.ents] + [
            (span, group_name)
            for group_name, group in doc.spans.items()
            for span in group
        ]:
            key = self._span_key(span, group_name)
            attributes = self._attribute_values(span)
            if key in spans and spans[key] != attributes:
                raise ValueError(
                    "Conflicting attributes for duplicate span {!r}".format(key)
                )
            spans[key] = attributes

        ordered_keys = sorted(spans)
        ids = {key: "T{}".format(i) for i, key in enumerate(ordered_keys, 1)}
        by_range = defaultdict(list)
        for key in ordered_keys:
            by_range[key[:2]].append(key)

        lines = []
        attribute_types = defaultdict(set)
        for key in ordered_keys:
            start, end, label = key
            reference = re.sub(r"\s+", " ", doc.text[start:end])
            lines.append(
                "{}\t{} {} {}\t{}".format(ids[key], label, start, end, reference)
            )

        attribute_index = 1
        for key in ordered_keys:
            for name, value in sorted(spans[key].items()):
                attribute_types[name].add(value)
                suffix = "" if value is True else " " + value
                lines.append(
                    "A{}\t{} {}{}".format(attribute_index, name, ids[key], suffix)
                )
                attribute_index += 1

        def endpoint_id(span):
            if not isinstance(span, Span) or span.doc is not doc:
                raise ValueError("Relation endpoint must be a span in the exported Doc")
            if span.label_:
                key = self._span_key(span)
                if key not in ids:
                    raise ValueError(
                        "Relation endpoint {!r} was not exported".format(key)
                    )
                return ids[key], key[2]
            matches = by_range.get((span.start_char, span.end_char), ())
            if len(matches) != 1:
                message = "Unlabeled relation endpoint at {}:{} is missing or ambiguous"
                raise ValueError(
                    message.format(span.start_char, span.end_char)
                )
            key = matches[0]
            return ids[key], key[2]

        relation_types = set()
        relations = doc._.relations if Doc.has_extension("relations") else ()
        for index, relation in enumerate(relations or (), 1):
            if not isinstance(relation, (tuple, list)) or len(relation) != 3:
                raise ValueError("Relations must be (source span, target span, label)")
            source, target, label = relation
            self._check_type_name(label, "relation")
            source_id, source_type = endpoint_id(source)
            target_id, target_type = endpoint_id(target)
            relation_types.add((label, source_type, target_type))
            lines.append(
                "R{}\t{} Arg1:{} Arg2:{}".format(
                    index, label, source_id, target_id
                )
            )

        return (
            "\n".join(lines) + ("\n" if lines else ""),
            {key[2] for key in ordered_keys},
            relation_types,
            attribute_types,
        )

    def write_docs_to_brat(
        self,
        docs: Iterable[Doc],
        report_ids: Iterable[str],
        directory: Union[str, Path],
        overwrite: bool = False,
    ) -> None:
        """Write a BRAT collection and its generated ``annotation.conf``.

        ``report_ids`` are safe filename stems; each produces ``.txt`` and
        ``.ann`` files. Existing target files are preserved unless ``overwrite``
        is true. Other files in the output directory are never removed.
        """
        docs = list(docs)
        report_ids = list(report_ids)
        if len(docs) != len(report_ids):
            raise ValueError("docs and report_ids must have the same length")
        for report_id in report_ids:
            if (
                not isinstance(report_id, str)
                or not _REPORT_NAME.fullmatch(report_id)
                or report_id in (".", "..")
                or report_id.endswith(".")
                or report_id.lower().endswith((".txt", ".ann"))
                or report_id.split(".", 1)[0].upper() in _WINDOWS_RESERVED
            ):
                raise ValueError("Invalid report ID: {!r}".format(report_id))
        if len({report_id.casefold() for report_id in report_ids}) != len(report_ids):
            raise ValueError("report_ids must be unique (ignoring case)")

        directory = Path(directory)
        planned = {}
        entity_types = set()
        relation_types = set()
        attribute_types = defaultdict(set)
        for doc, report_id in zip(docs, report_ids):
            if not isinstance(doc, Doc):
                raise TypeError("docs must contain spaCy Doc objects")
            annotation, entities, relations, attributes = self._render_doc(doc)
            planned[directory / (report_id + ".txt")] = doc.text
            planned[directory / (report_id + ".ann")] = annotation
            entity_types.update(entities)
            relation_types.update(relations)
            for name, values in attributes.items():
                attribute_types[name].update(values)

        sections = ["[entities]", *sorted(entity_types), "", "[relations]"]
        for label, source_type, target_type in sorted(relation_types):
            sections.append(
                "{} Arg1:{}, Arg2:{}".format(label, source_type, target_type)
            )
        sections.extend(("", "[events]", "", "[attributes]"))
        for name, values in sorted(attribute_types.items()):
            if True in values and len(values) > 1:
                raise ValueError(
                    "Attribute {!r} mixes binary and valued annotations".format(name)
                )
            definition = "{} Arg:<ENTITY>".format(name)
            if values != {True}:
                definition += ", Value:" + "|".join(sorted(values))
            sections.append(definition)
        planned[directory / "annotation.conf"] = "\n".join(sections) + "\n"

        if directory.exists() and not directory.is_dir():
            raise ValueError("Output path is not a directory: {}".format(directory))
        if not overwrite:
            existing = [path for path in planned if path.exists()]
            if existing:
                raise FileExistsError(
                    "BRAT output already exists: {}".format(existing[0])
                )
        directory.mkdir(parents=True, exist_ok=True)
        for path, content in planned.items():
            with path.open(
                "w" if overwrite else "x", encoding="utf-8", newline=""
            ) as output:
                output.write(content)
