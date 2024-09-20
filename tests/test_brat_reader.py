import logging
import unittest
from pathlib import Path

from spacy.lang.en import English
from spacy.tokens import Doc

from medspacy_io.reader import BratDirReader
from medspacy_io.reader import BratDocReader

import os
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

class TestBratReader(unittest.TestCase):

    def test_reader_initail(self):
        breader = BratDocReader(nlp=English())
        assert (hasattr(breader, 'encoding'))
        assert (not breader.encoding)

    def test_parse_to_dicts(self):
        if Doc.has_extension("concepts"):
            Doc.remove_extension("concepts")
        breader = BratDocReader(nlp=English())
        spans, classes, attributes, relations = breader.parse_to_dicts(
            Path('data/brat_test_corpus/000-introduction.ann').read_text())
        assert (len(spans) == 12)
        assert (len(classes) == 17)
        assert (len(attributes) == 6)
        assert (len(relations) == 5)

    def test_set_attributes(self):
        if Doc.has_extension("concepts"):
            Doc.remove_extension("concepts")
        BratDocReader(nlp=English(), schema_file='data/brat_test_corpus/annotation.conf')
        nlp = English()
        doc = nlp('test status attribute')
        span = doc[1:2]
        assert (hasattr(span._, 'Negation'))
        assert (hasattr(span._, 'Confidence'))

    def test_read(self):
        if Doc.has_extension("concepts"):
            Doc.remove_extension("concepts")
        breader = BratDocReader(nlp=English(), schema_file='data/brat_test_corpus/annotation.conf')
        doc = breader.read('data/brat_test_corpus/000-introduction.txt')
        assert (len(doc.ents) == 12)
        assert (str(doc.ents[0].label_) == 'Gene_expression')
        assert (str(doc.ents[1].label_) == 'Protein')
        assert (str(doc.ents[2].label_) == 'Negative_regulation')
        assert (str(doc.ents[3].label_) == 'Positive_regulation')
        assert (str(doc.ents[4].label_) == 'Protein')
        assert (str(doc.ents[5].label_) == 'Gene_expression')
        assert (str(doc.ents[6].label_) == 'Protein')
        assert (str(doc.ents[7].label_) == 'Complex')
        assert (str(doc.ents[8].label_) == 'Protein')
        assert (str(doc.ents[9].label_) == 'Positive_regulation')
        assert (str(doc.ents[10].label_) == 'Simple_chemical')
        assert (str(doc.ents[11].label_) == 'Protein')

    def test_read_doc_name(self):
        breader = BratDocReader(nlp=English(), schema_file='data/brat_test_corpus/annotation.conf')

        doc = breader.read('data/brat_test_corpus/000-introduction.txt')
        assert (doc._.doc_name == '000-introduction.txt')
        breader.doc_name_depth = 1
        doc = breader.read('data/brat_test_corpus/000-introduction.txt')
        assert (doc._.doc_name ==  str(Path('brat_test_corpus', '000-introduction.txt')))
        breader = BratDocReader(nlp=English(), schema_file='data/brat_test_corpus/annotation.conf',
                                doc_name_depth=2)
        doc = breader.read('data/brat_test_corpus/000-introduction.txt')
        print(doc._.doc_name)
        assert (doc._.doc_name ==  str(Path('data', 'brat_test_corpus', '000-introduction.txt')))

    def test_read_overlap(self):
        breader = BratDocReader(nlp=English(), schema_file='data/brat_test_corpus/annotation.conf',
                                support_overlap=True, store_anno_string=True)
        doc = breader.read('data/brat_test_corpus/000-introduction.txt')
        print(len(doc.spans))
        assert (len(doc.spans) == 6)
        assert ('Gene_expression' in doc.spans)
        assert ('Protein' in doc.spans)
        assert ('Negative_regulation' in doc.spans)
        assert ('Positive_regulation' in doc.spans)
        assert ('Complex' in doc.spans)
        assert ('Simple_chemical' in doc.spans)
        assert (len(doc.spans['Gene_expression']) == 2)
        assert (len(doc.spans['Protein']) == 5)
        assert (len(doc.spans['Negative_regulation']) == 1)
        assert (len(doc.spans['Positive_regulation']) == 2)
        assert (len(doc.spans['Complex']) == 1)
        assert (len(doc.spans['Simple_chemical']) == 1)
        concepts = doc.spans
        counter = 0
        for concept_type, annos in concepts.items():
            for anno in annos:
                if str(anno) != anno._.span_txt:
                    counter += 1
                    print(concept_type)
                    print(anno)
                    print(anno._.span_txt)
                    print('\n')
        assert (counter == 1)

    def test_check_spans(self):
        if Doc.has_extension("concepts"):
            Doc.remove_extension("concepts")
        breader = BratDocReader(nlp=English(), schema_file='data/brat_test_corpus/annotation.conf',
                                support_overlap=True, store_anno_string=True, encoding='UTF8',
                                log_level=logging.DEBUG)
        doc = breader.read('data/brat_test_corpus/000-introduction.txt')
        for span in doc.ents:
            if span._.span_txt.replace('\n', ' ') not in str(span).replace('\n', ' '):
                print(span._.span_txt, '<>', span)
            assert (span._.span_txt == 'complicated panic' or (
                    span._.span_txt.replace('\n', ' ') in str(span).replace('\n', ' ')))

    def test_check_spans2(self):
        if Doc.has_extension("concepts"):
            Doc.remove_extension("concepts")
        breader = BratDocReader(nlp=English(), schema_file='data/brat_test_corpus/annotation.conf',
                                support_overlap=False, store_anno_string=True, encoding='UTF8',
                                log_level=logging.DEBUG)
        doc = breader.read('data/brat_test_corpus/040-text_span_annotation.txt')
        for span in doc.ents:
            assert (span._.span_txt.replace('\n', ' ') in str(span).replace('\n', ' '))

        def test_dir_reader(self):
            if Doc.has_extension("concepts"):
                Doc.remove_extension("concepts")
            dir_reader = BratDirReader(nlp=English(), support_overlap=True, recursive=True,
                                       schema_file='data/brat_test_corpus/annotation.conf')
            docs = dir_reader.read(txt_dir='data/brat_test_corpus/')
            assert (len(docs) == 2)
            doc = docs[0]
            assert (len(doc.spans) == 6)
            assert ('Gene_expression' in doc.spans)
            assert ('Protein' in doc.spans)
            assert ('Negative_regulation' in doc.spans)
            assert ('Positive_regulation' in doc.spans)
            assert ('Complex' in doc.spans)
            assert ('Simple_chemical' in doc.spans)
            assert (len(doc.spans['Gene_expression']) == 2)
            assert (len(doc.spans['Protein']) == 5)
            assert (len(doc.spans['Negative_regulation']) == 1)
            assert (len(doc.spans['Positive_regulation']) == 2)
            assert (len(doc.spans['Complex']) == 1)
            assert (len(doc.spans['Simple_chemical']) == 1)
