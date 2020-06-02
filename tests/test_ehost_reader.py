import unittest
from spacy.lang.en import English
from medspacy_io.reader import EhostDocReader
from medspacy_io.reader import EhostDirReader


class TestEhostReader(unittest.TestCase):

    def test_reader_initail(self):
        ereader = EhostDocReader(nlp=English())
        assert (hasattr(ereader, 'use_adjudication'))
        assert (not ereader.use_adjudication)

    def test_parse_to_dicts(self):
        ereader = EhostDocReader(nlp=English())
        spans, classes, attributes = ereader.parse_to_dicts('data/ehost_test_corpus/saved/doc1.txt.knowtator.xml')
        assert (len(spans) == 7)
        assert (len(classes) == 7)
        assert (len(attributes) == 6)

    def test_set_attributes(self):
        EhostDocReader(nlp=English(), schema_file='data/ehost_test_corpus/config/projectschema.xml')
        nlp = English()
        doc = nlp('test status attribute')
        assert (hasattr(doc[1:2]._, 'status'))
        assert (doc[1:2]._.status == 'present')

    def test_read(self):
        ereader = EhostDocReader(nlp=English(), schema_file='data/ehost_test_corpus/config/projectschema.xml')
        doc = ereader.read('data/ehost_test_corpus/corpus/doc1.txt')
        self.eval(doc)

    def test_read_overlap(self):
        ereader = EhostDocReader(nlp=English(), schema_file='data/ehost_test_corpus/config/projectschema.xml',
                                 support_overlap=True)
        doc = ereader.read('data/ehost_test_corpus2/corpus/doc1.txt')
        assert (len(doc._.concepts) == 7)
        assert (len(doc._.concepts['Incision_and_Drainage']) == 2)
        doc = ereader.read('data/ehost_test_corpus2/corpus/doc2.txt')
        assert (len(doc._.concepts) == 7)
        assert (len(doc._.concepts['Exclusions']) == 2)
        assert (len(doc._.concepts['Doc_Level_Purulence_Assessment']) == 2)

    def test_dir_reader(self):
        dir_reader = EhostDirReader(txt_dir='data/ehost_test_corpus/',
                                    nlp=English(),
                                    docReaderClass=EhostDocReader, recursive=True,
                                    schema_file='data/ehost_test_corpus/config/projectschema.xml')
        docs = dir_reader.read()
        assert (len(docs) == 2)
        for doc in docs:
            self.eval(doc)

    def test_dir_reader2(self):
        dir_reader = EhostDirReader(txt_dir='data/ehost_test_corpus/',
                                    nlp=English(), support_overlap=True,
                                    docReaderClass=EhostDocReader, recursive=True,
                                    schema_file='data/ehost_test_corpus/config/projectschema.xml')
        docs = dir_reader.read()
        assert (len(docs) == 2)
        for doc in docs:
            self.eval(doc)

    def eval(self, doc):
        assert (len(doc.ents) == 7)
        assert (str(doc.ents[0]) == 'CHIEF')
        assert (str(doc.ents[1]) == 'Abdominal pain')
        assert (str(doc.ents[2]) == 'PRESENT')
        assert (str(doc.ents[3]) == 'patient')
        # there is a slightly mismatch of the token, because SpaCy tokenize '71-year-old' into
        # '71-year', '-', 'old', EhostDocReader adjust the annotation spans to align with the tokens
        assert (str(doc.ents[4]) == '71-year-old')
        assert (str(doc.ents[5]) == 'X. The patient')
        assert (str(doc.ents[6]) == 'presented')
        assert (doc.ents[0].label_ == 'Doc_Level_Purulence_Assessment')
        assert (doc.ents[1].label_ == 'Purulent')
        assert (doc.ents[2].label_ == 'Non-Purulent')
        assert (doc.ents[3].label_ == 'Incision_and_Drainage')
        assert (doc.ents[4].label_ == 'PreAnnotated')
        assert (doc.ents[5].label_ == 'Nonspecific_SSTI')
        assert (doc.ents[6].label_ == 'Exclusions')
