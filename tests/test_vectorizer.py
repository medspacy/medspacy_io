import unittest
from collections import OrderedDict
import os

from spacy.lang.en import English
from spacy.tokens.span import Span
from PyRuSH import PyRuSHSentencizer
from medspacy_io.reader import EhostDocReader
from medspacy_io.reader import EhostDirReader
from medspacy_io.vectorizer import Vectorizer
from spacy.tokens.doc import Doc
import pandas as pd


class TestEhostReader(unittest.TestCase):

    def setUp(self) -> None:
        script_directory = os.path.dirname(os.path.abspath(__file__))
        # Change the current working directory to the script's directory
        os.chdir(script_directory)
        self.nlp = English()
        self.nlp.add_pipe("medspacy_pyrush", config={'rules_path':'conf/rush_rules.tsv'})

    def test_to_sents_df(self):
        ereader = EhostDocReader(nlp=self.nlp, schema_file='data/ehost_test_corpus2/config/projectschema.xml',
                                 support_overlap=True)
        doc = ereader.read('data/ehost_test_corpus2/corpus/doc1.txt')
        print(len(list(doc.sents)))
        assert (len(doc.spans) == 3)
        assert (len(doc.spans['Nonspecific_SSTI']) == 1)
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
        assert (len(doc.spans) == 3)
        assert (len(doc.spans['Nonspecific_SSTI']) == 1)
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
                                    recursive=True,
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
                                    recursive=True,
                                    schema_file='data/ehost_test_corpus2/config/projectschema.xml')
        docs = dir_reader.read(txt_dir='data/ehost_test_corpus2/')
        df = Vectorizer.docs_to_sents_df(docs)
        assert (df.shape[0] == 12)
        df = Vectorizer.docs_to_sents_df(docs, sent_window=2)
        assert (df.shape[0] == 19)
        df = Vectorizer.docs_to_sents_df(docs, sent_window=2, track_doc_name=True)
        assert (df.shape[0] == 19)
        assert (df.shape[1] == 4)

    def test_get_output_labels(self):
        type_filter = {
            "Nonspecific_SSTI": {'status': {'present': 'PRES_Nonspecific_SSTI', 'historical': 'HIS_Nonspecific_SSTI'},
                                 'SSI': 'SSI'}}
        output_labels = OrderedDict()
        Vectorizer.get_output_labels(type_filter, output_labels)
        print(output_labels)
        assert (len(output_labels) == 3)
        assert ('PRES_Nonspecific_SSTI' in output_labels)
        assert ('HIS_Nonspecific_SSTI' in output_labels)
        assert ('SSI' in output_labels)


    def test_to_seq_data_dict_on_types(self):
        ereader = EhostDocReader(nlp=self.nlp, schema_file='data/ehost_test_corpus2/config/projectschema.xml',
                                 support_overlap=True)
        doc = ereader.read('data/ehost_test_corpus2/corpus/doc1.txt')
        print(list(doc.sents))
        res=Vectorizer.to_seq_data_dict(doc,type_filter=['Nonspecific_SSTI','PreAnnotated'])
        # print('\n'.join([str(item) for item in res.items()]))
        for i,s in enumerate(res['X']):
            self.print(res,i)
            print('\n')
        assert (res['Nonspecific_SSTI'][2][0]=='Nonspecific_SSTI')
        assert (res['Nonspecific_SSTI'][2][1]=='Nonspecific_SSTI')
        assert (res['PreAnnotated'][1][9]=='PreAnnotated')
        assert (res['PreAnnotated'][1][10]=='PreAnnotated')
        assert (res['PreAnnotated'][1][11]=='PreAnnotated')


    def test_to_seq_data_dict_on_types2(self):
        ereader = EhostDocReader(nlp=self.nlp, schema_file='data/ehost_test_corpus2/config/projectschema.xml',
                                 support_overlap=True)
        doc = ereader.read('data/ehost_test_corpus2/corpus/doc1.txt')
        print(list(doc.sents))
        res=Vectorizer.to_seq_data_dict(doc,type_filter=['Nonspecific_SSTI','PreAnnotated'],sent_window=2)
        # print('\n'.join([str(item) for item in res.items()]))
        for i in range(0, len(res['X'])):
            self.print(res,i)
            print('\n')
        assert(len(res['X'])==3)
        assert(res['PreAnnotated'][0][0]=='O')
        assert(res['PreAnnotated'][0][7]=='[SEP]')
        assert(res['PreAnnotated'][0][16]=='O')
        assert(res['PreAnnotated'][0][17]=='PreAnnotated')
        assert(res['PreAnnotated'][0][18]=='PreAnnotated')
        assert(res['PreAnnotated'][0][19]=='PreAnnotated')
        assert(res['Nonspecific_SSTI'][0][20]=='O')
        assert(res['Nonspecific_SSTI'][2][0]=='Nonspecific_SSTI')
        assert(res['Nonspecific_SSTI'][2][1]=='Nonspecific_SSTI')
        assert(res['Nonspecific_SSTI'][2][2]=='O')

    def test_to_seq_data_dict_on_types3(self):
        ereader = EhostDocReader(nlp=self.nlp, schema_file='data/ehost_test_corpus2/config/projectschema.xml',
                                 support_overlap=True)
        doc = ereader.read('data/ehost_test_corpus2/corpus/doc1.txt')
        print(list(doc.sents))
        res=Vectorizer.to_seq_data_dict(doc,type_filter=['Nonspecific_SSTI','PreAnnotated'],
                                        sent_window=2,
                                        sep_token=None,default_label='NULL')
        # print('\n'.join([str(item) for item in res.items()]))
        for i in range(0, len(res['X'])):
            self.print(res,i)
            print('\n')
        assert(len(res['X'])==3)
        assert(res['PreAnnotated'][0][0]=='NULL')
        assert(res['PreAnnotated'][0][7]=='NULL')
        assert(res['PreAnnotated'][0][15]=='NULL')
        assert(res['PreAnnotated'][0][16]=='PreAnnotated')
        assert(res['PreAnnotated'][0][17]=='PreAnnotated')
        assert(res['PreAnnotated'][0][18]=='PreAnnotated')
        assert(res['Nonspecific_SSTI'][0][20]=='NULL')
        assert(res['Nonspecific_SSTI'][2][0]=='Nonspecific_SSTI')
        assert(res['Nonspecific_SSTI'][2][1]=='Nonspecific_SSTI')
        assert(res['Nonspecific_SSTI'][2][2]=='NULL')


    def test_to_seq_data_dict_on_types4(self):
        ereader = EhostDocReader(nlp=self.nlp, schema_file='data/ehost_test_corpus2/config/projectschema.xml',
                                 support_overlap=True)
        doc = ereader.read('data/ehost_test_corpus2/corpus/doc1.txt')
        print(list(doc.sents))
        res=Vectorizer.to_seq_data_dict(doc,type_filter=['Nonspecific_SSTI','PreAnnotated'],sent_window=3)
        # print('\n'.join([str(item) for item in res.items()]))
        for i in range(0, len(res['X'])):
            self.print(res,i)
            print('\n')
        assert(len(res['X'])==2)
        assert(res['PreAnnotated'][0][0]=='O')
        assert(res['PreAnnotated'][0][7]=='[SEP]')
        assert(res['PreAnnotated'][0][16]=='O')
        assert(res['PreAnnotated'][0][17]=='PreAnnotated')
        assert(res['PreAnnotated'][0][18]=='PreAnnotated')
        assert(res['PreAnnotated'][0][19]=='PreAnnotated')
        assert(res['Nonspecific_SSTI'][0][25]=='[SEP]')
        assert(res['Nonspecific_SSTI'][0][26]=='Nonspecific_SSTI')
        assert(res['Nonspecific_SSTI'][0][27]=='Nonspecific_SSTI')
        assert(res['Nonspecific_SSTI'][0][28]=='O')

        assert(res['PreAnnotated'][1][8]=='O')
        assert(res['PreAnnotated'][1][9]=='PreAnnotated')
        assert(res['PreAnnotated'][1][10]=='PreAnnotated')
        assert(res['PreAnnotated'][1][11]=='PreAnnotated')
        assert(res['Nonspecific_SSTI'][1][12]=='O')
        assert(res['Nonspecific_SSTI'][1][17]=='[SEP]')
        assert(res['Nonspecific_SSTI'][1][18]=='Nonspecific_SSTI')
        assert(res['Nonspecific_SSTI'][1][19]=='Nonspecific_SSTI')
        assert(res['Nonspecific_SSTI'][1][20]=='O')

    def test_to_seq_data_dict_on_types5(self):
        ereader = EhostDocReader(nlp=self.nlp, schema_file='data/ehost_test_corpus2/config/projectschema.xml',
                                 support_overlap=True)
        doc = ereader.read('data/ehost_test_corpus2/corpus/doc1.txt')
        print(list(doc.sents))
        res=Vectorizer.to_seq_data_dict(doc,type_filter={"Nonspecific_SSTI": {'status': {'present': 'PRES_NS_SSTI'},
                                                                              'test':{'v2':"TEST"}
                                                                              },
                                                         "PreAnnotated":"PREANNO"
                                                         },sent_window=1, data_dict=OrderedDict(),output_labels={})
        # print('\n'.join([str(item) for item in res.items()]))
        for i in range(0, len(res['X'])):
            self.print(res,i)
            print('\n')
        assert(len(res['X'])==4)
        assert(res['PREANNO'][1][0]=='O')
        assert(res['PREANNO'][1][8]=='O')
        assert(res['PREANNO'][1][9]=='PREANNO')
        assert(res['PREANNO'][1][10]=='PREANNO')
        assert(res['PREANNO'][1][11]=='PREANNO')
        assert(res['PREANNO'][1][12]=='O')

        assert(res['PRES_NS_SSTI'][2][0]=='PRES_NS_SSTI')
        assert(res['PRES_NS_SSTI'][2][1]=='PRES_NS_SSTI')
        assert(res['PRES_NS_SSTI'][2][2]=='O')

        assert(res['TEST'][2][0]=='TEST')
        assert(res['TEST'][2][1]=='TEST')
        assert(res['TEST'][2][2]=='O')

    def print(self, res, id=0):
        sent=res['X'][id]
        tokens=res['tokens'][id]
        inter_dict={'tokens':tokens}
        for i, l in enumerate(list(res.keys())[2:-1]):
            inter_dict[l]=res[l][id]
        df=pd.DataFrame(inter_dict)
        print(df)
        pass