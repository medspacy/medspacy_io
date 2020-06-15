import unittest
from spacy.lang.en import English
from medspacy_io.reader import EhostDocReader
from medspacy_io.reader import EhostDirReader
from PyRuSH import PyRuSHSentencizer
from medspacy_io.vectorizer import Vectorizer
from spacy.tokens.doc import Doc


class TestEhostReader(unittest.TestCase):

    def setUp(self) -> None:
        self.nlp = English()
        self.nlp.add_pipe(PyRuSHSentencizer('conf/rush_rules.tsv'))

    def test_to_sents_df(self):
        ereader = EhostDocReader(nlp=self.nlp, schema_file='data/ehost_test_corpus2/config/projectschema.xml',
                                 support_overlap=True)
        doc = ereader.read('data/ehost_test_corpus2/corpus/doc1.txt')
        print(len(list(doc.sents)))
        assert (len(doc._.concepts) == 3)
        assert (len(doc._.concepts['Nonspecific_SSTI']) == 1)
        df = Vectorizer.to_sents_df(doc)
        # print(df.shape)
        assert (df.shape[0] == 4)
        df = Vectorizer.to_sents_df(doc, track_doc_name=True)
        # print(df.shape)
        assert (df.shape[1] == 4)
        df = Vectorizer.to_sents_df(doc, sent_window=2)
        # print(df.shape)
        assert (df.shape[0] == 5)
        df = Vectorizer.to_sents_df(doc, sent_window=2, track_doc_name=True)
        # print(df.shape)
        assert (df.shape[0] == 5)
        assert (df.shape[1] == 4)

    def test_to_sents_nparray(self):
        ereader = EhostDocReader(nlp=self.nlp, schema_file='data/ehost_test_corpus2/config/projectschema.xml',
                                 support_overlap=True)
        doc = ereader.read('data/ehost_test_corpus2/corpus/doc1.txt')
        print(len(list(doc.sents)))
        assert (len(doc._.concepts) == 3)
        assert (len(doc._.concepts['Nonspecific_SSTI']) == 1)
        df = Vectorizer.to_sents_nparray(doc)
        print(df)
        assert (df.shape[0] == 4)
        df = Vectorizer.to_sents_nparray(doc, sent_window=2)
        print(df.shape)
        assert (df.shape[0] == 5)
        df = Vectorizer.to_sents_nparray(doc, sent_window=2, track_doc_name=True)
        print(df.shape)
        assert (df.shape[0] == 5)
        assert (df.shape[1] == 4)

    def test_to_sents_df_on_attr_value(self):
        ereader = EhostDocReader(nlp=self.nlp, schema_file='data/ehost_test_corpus2/config/projectschema.xml',
                                 support_overlap=True)
        doc = ereader.read('data/ehost_test_corpus2/corpus/doc1.txt')
        df = Vectorizer.to_sents_df(doc,
                                    type_filter={"Nonspecific_SSTI": {'status': {'present': 'PRES_Nonspecific_SSTI'}}})
        print(df.shape)
        print(df)
        assert (df.shape[0] == 4)
        assert (df.iloc[0].y == 'PRES_Nonspecific_SSTI')
        df = Vectorizer.to_sents_df(doc, sent_window=2,
                                    type_filter={"Nonspecific_SSTI": {'status': {'present': 'PRES_Nonspecific_SSTI'}}})
        print(df)
        assert (df.shape[0] == 3)
        df = Vectorizer.to_sents_df(doc, sent_window=2,
                                    type_filter={"Nonspecific_SSTI": {'status': {'present': 'PRES_Nonspecific_SSTI'}}},
                                    track_doc_name=True)
        print(df)
        assert (df.shape[0] == 3)
        assert (df.shape[1] == 4)

    def test_to_sents_df_on_attr_value2(self):
        ereader = EhostDocReader(nlp=self.nlp, schema_file='data/ehost_test_corpus2/config/projectschema.xml',
                                 support_overlap=True)
        doc = ereader.read('data/ehost_test_corpus2/corpus/doc1.txt')
        df = Vectorizer.to_sents_df(doc,
                                    type_filter={"Nonspecific_SSTI": {'status': {'negated': 'PRES_Nonspecific_SSTI'}}})
        print(df.shape)
        print(df)
        assert (df.shape[0] == 4)
        assert (df.iloc[0].y == 'NEG')
        df = Vectorizer.to_sents_df(doc, sent_window=2,
                                    type_filter={"Nonspecific_SSTI": {'status': {'negated': 'PRES_Nonspecific_SSTI'}}})
        print(df.shape)
        assert (df.shape[0] == 3)
        print(df)
        assert (df.iloc[0].y == 'NEG')

    def test_to_sents_df_on_attr_value3(self):
        ereader = EhostDocReader(nlp=self.nlp, schema_file='data/ehost_test_corpus2/config/projectschema.xml',
                                 support_overlap=True)
        doc = ereader.read('data/ehost_test_corpus2/corpus/doc1.txt')
        df = Vectorizer.to_sents_df(doc,
                                    type_filter={"Nonspecific_SSTI": {'status': {'present': 'PRES_Nonspecific_SSTI'},
                                                                      'test': {'v2': "TYPE_1"}}})
        print(df.shape)
        print(df)
        assert (df.shape[0] == 5)
        assert (df.iloc[0].y == 'PRES_Nonspecific_SSTI')
        assert (df.iloc[1].y == 'TYPE_1')
        df = Vectorizer.to_sents_df(doc, sent_window=2,
                                    type_filter={"Nonspecific_SSTI": {'status': {'present': 'PRES_Nonspecific_SSTI'},
                                                                      'test': {'v2': "TYPE_1"}}})
        print(df.shape)
        assert (df.shape[0] == 5)

    def test_docs_to_sents_df(self):
        if Doc.has_extension("concepts"):
            Doc.remove_extension("concepts")
        dir_reader = EhostDirReader(nlp=self.nlp, support_overlap=False,
                                    docReaderClass=EhostDocReader, recursive=True,
                                    schema_file='data/ehost_test_corpus/config/projectschema.xml')

        docs = dir_reader.read(txt_dir='data/ehost_test_corpus/')
        df = Vectorizer.docs_to_sents_df(docs, type_filter=set(), track_doc_name=True)
        print(df)
        assert (df.shape[0] == 12)
        df = Vectorizer.docs_to_sents_df(docs, type_filter=set())
        print(df)
        df = Vectorizer.docs_to_sents_df(docs, sent_window=2)
        assert (df.shape[0] == 20)

    def test_docs_to_sents_df2(self):
        dir_reader = EhostDirReader(nlp=self.nlp, support_overlap=True,
                                    docReaderClass=EhostDocReader, recursive=True,
                                    schema_file='data/ehost_test_corpus2/config/projectschema.xml')
        docs = dir_reader.read(txt_dir='data/ehost_test_corpus2/')
        df = Vectorizer.docs_to_sents_df(docs)
        assert (df.shape[0] == 12)
        df = Vectorizer.docs_to_sents_df(docs, sent_window=2)
        assert (df.shape[0] == 19)
        df = Vectorizer.docs_to_sents_df(docs, sent_window=2, track_doc_name=True)
        assert (df.shape[0] == 19)
        assert (df.shape[1] == 4)
