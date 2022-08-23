import logging
import unittest

from spacy.lang.en import English
from spacy.tokens import Doc

from medspacy_io.reader import EhostDirReader
from medspacy_io.reader import EhostDocReader


class TestEhostReader(unittest.TestCase):

    def test_reader_initail(self):
        ereader = EhostDocReader(nlp=English())
        assert (hasattr(ereader, 'use_adjudication'))
        assert (not ereader.use_adjudication)

    def test_parse_to_dicts(self):
        if Doc.has_extension("concepts"):
            Doc.remove_extension("concepts")
        ereader = EhostDocReader(nlp=English())
        spans, classes, attributes, relations = ereader.parse_to_dicts('data/ehost_test_corpus/saved/doc1.txt.knowtator.xml')
        assert (len(spans) == 7)
        assert (len(classes) == 7)
        assert (len(attributes) == 6)

    def test_set_attributes(self):
        if Doc.has_extension("concepts"):
            Doc.remove_extension("concepts")
        EhostDocReader(nlp=English(), schema_file='data/ehost_test_corpus/config/projectschema.xml')
        nlp = English()
        doc = nlp('test status attribute')
        assert (hasattr(doc[1:2]._, 'status'))
        assert (doc[1:2]._.status == 'present')

    def test_read(self):
        if Doc.has_extension("concepts"):
            Doc.remove_extension("concepts")
        ereader = EhostDocReader(nlp=English(), schema_file='data/ehost_test_corpus/config/projectschema.xml')
        doc = ereader.read('data/ehost_test_corpus/corpus/doc1.txt')
        self.eval(doc)

        def test_read_doc_name(self):
            ereader = EhostDocReader(nlp=English(), schema_file='data/ehost_test_corpus/config/projectschema.xml')

        doc = ereader.read('data/ehost_test_corpus/corpus/doc1.txt')
        assert (doc._.doc_name == 'doc1.txt')
        ereader.doc_name_depth = 1
        doc = ereader.read('data/ehost_test_corpus/corpus/doc1.txt')
        assert (doc._.doc_name == r'corpus/doc1.txt')
        ereader = EhostDocReader(nlp=English(), schema_file='data/ehost_test_corpus/config/projectschema.xml',
                                 doc_name_depth=2)
        doc = ereader.read('data/ehost_test_corpus/corpus/doc1.txt')
        assert (doc._.doc_name == r'ehost_test_corpus/corpus/doc1.txt')

    def test_read_doc_name(self):
        ereader = EhostDocReader(nlp=English(), schema_file='data/ehost_test_corpus/config/projectschema.xml')
        doc = ereader.read('data/ehost_test_corpus/corpus/doc1.txt')
        assert(doc._.doc_name=='doc1.txt')
        ereader.doc_name_depth=1
        doc = ereader.read('data/ehost_test_corpus/corpus/doc1.txt')
        assert (doc._.doc_name==r'corpus/doc1.txt')
        ereader = EhostDocReader(nlp=English(), schema_file='data/ehost_test_corpus/config/projectschema.xml',
                                 doc_name_depth=2)
        doc = ereader.read('data/ehost_test_corpus/corpus/doc1.txt')
        assert (doc._.doc_name==r'ehost_test_corpus/corpus/doc1.txt')


    def test_read_overlap(self):
        ereader = EhostDocReader(nlp=English(), schema_file='data/ehost_test_corpus2/config/projectschema.xml',
                                 support_overlap=True)
        doc = ereader.read('data/ehost_test_corpus2/corpus/doc1.txt')
        assert (len(doc.spans) == 3)
        assert (len(doc.spans['PreAnnotated']) == 1)
        doc = ereader.read('data/ehost_test_corpus2/corpus/doc2.txt')
        assert (len(doc.spans) == 7)
        assert (len(doc.spans['Exclusions']) == 2)
        assert (len(doc.spans['Doc_Level_Purulence_Assessment']) == 2)

    def test_check_spans(self):
        if Doc.has_extension("concepts"):
            Doc.remove_extension("concepts")
        ereader = EhostDocReader(nlp=English(), schema_file='data/ehost_test_corpus/config/projectschema.xml',
                                 support_overlap=False, store_anno_string=True, encoding='UTF8',
                                 log_level=logging.DEBUG)
        doc = ereader.read('data/ehost_test_corpus/corpus/doc2.txt')
        for span in doc.ents:
            print(span._.span_txt, '<>', span)
            assert (span._.span_txt.replace('\n', ' ') in str(span).replace('\n', ' '))

    def test_check_spans2(self):
        ereader = EhostDocReader(nlp=English(), schema_file='data/ehost_test_corpus2/config/projectschema.xml',
                                 support_overlap=True, store_anno_string=True, log_level=logging.DEBUG)
        doc = ereader.read('data/ehost_test_corpus2/corpus/doc2.txt')
        for spans in doc.spans.values():
            for span in spans:
                print(span._.span_txt, '<>', span)
                assert (span._.span_txt.replace('\n', ' ') in str(span).replace('\n', ' '))

    def test_dir_reader(self):
        if Doc.has_extension("concepts"):
            Doc.remove_extension("concepts")
        dir_reader = EhostDirReader(nlp=English(), recursive=True,
                                    schema_file='data/ehost_test_corpus/config/projectschema.xml')
        docs = dir_reader.read(txt_dir='data/ehost_test_corpus/')
        assert (len(docs) == 2)
        for doc in docs:
            self.eval(doc)

    def test_dir_reader2(self):
        if Doc.has_extension("concepts"):
            Doc.remove_extension("concepts")
        dir_reader = EhostDirReader(nlp=English(), support_overlap=True,recursive=True,
                                    schema_file='data/ehost_test_corpus/config/projectschema.xml')
        docs = dir_reader.read(txt_dir='data/ehost_test_corpus/')
        assert (len(docs) == 2)
        for doc in docs:
            assert (len(doc.spans) == 7)
            assert ('Doc_Level_Purulence_Assessment' in doc.spans)
            assert (str(doc.spans['Doc_Level_Purulence_Assessment'][0]) == 'CHIEF')
            assert ('Purulent' in doc.spans)
            assert (str(doc.spans['Purulent'][0]) == 'Abdominal pain')
            assert ('Non-Purulent' in doc.spans)
            assert (str(doc.spans['Non-Purulent'][0]) == 'PRESENT')
            assert ('Incision_and_Drainage' in doc.spans)
            assert (str(doc.spans['Incision_and_Drainage'][0]) == 'patient')
            assert ('PreAnnotated' in doc.spans)
            assert (str(doc.spans['PreAnnotated'][0]) == 'year-old')
            assert ('Nonspecific_SSTI' in doc.spans)
            assert (str(doc.spans['Nonspecific_SSTI'][0]) == 'X. The patient')
            assert ('Exclusions' in doc.spans)
            assert (str(doc.spans['Exclusions'][0]) == 'presented')

    def eval(self, doc):
        assert (len(doc.ents) == 7)
        assert (str(doc.ents[0]) == 'CHIEF')
        assert (str(doc.ents[1]) == 'Abdominal pain')
        assert (str(doc.ents[2]) == 'PRESENT')
        assert (str(doc.ents[3]) == 'patient')
        # there is a slightly mismatch of the token, because SpaCy tokenize '71-year-old' into
        # '71-year', '-', 'old', EhostDocReader adjust the annotation spans to align with the tokens
        assert (str(doc.ents[4]) == 'year-old')
        assert (str(doc.ents[5]) == 'X. The patient')
        assert (str(doc.ents[6]) == 'presented')
        assert (doc.ents[0].label_ == 'Doc_Level_Purulence_Assessment')
        assert (doc.ents[1].label_ == 'Purulent')
        assert (doc.ents[2].label_ == 'Non-Purulent')
        assert (doc.ents[3].label_ == 'Incision_and_Drainage')
        assert (doc.ents[4].label_ == 'PreAnnotated')
        assert (doc.ents[5].label_ == 'Nonspecific_SSTI')
        assert (doc.ents[6].label_ == 'Exclusions')
