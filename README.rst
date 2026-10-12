
medspacy-io
------------

medspacy-io is a collection of modules to facilitate reading text from various sources and writing to various sources.


Installation
------------

    pip install medspacy-io


How to use
------------

Please refer to the samples in the tests.

BRAT export
-----------

``BratWriter`` writes spaCy documents as BRAT ``.txt``/``.ann`` pairs and
always generates an ``annotation.conf`` for the collection::

    from medspacy_io.writer import BratWriter
    from spacy.tokens import Span

    if not Span.has_extension("Status"):
        Span.set_extension("Status", default=None)
    writer = BratWriter(span_attributes=["Status"])
    writer.write_docs_to_brat(docs, report_ids, "brat_output")

The writer includes ``doc.ents`` and every group in ``doc.spans``. A span's
label is its BRAT entity type; an unlabeled span uses its group name. Identical
spans with the same type are written once. Selected span extensions are written
as attributes, and ``doc._.relations`` triples are written as binary relations.
Existing target files are protected by default; pass ``overwrite=True`` to
replace them. Continuous spans are supported. Discontinuous spans, BRAT events,
notes, normalizations, and equivalence groups are not exported.

A Colab notebook demo is available at:

https://colab.research.google.com/drive/12EBOaKej9uPmGAb5dV2oB1goEWrwXjSD
    

