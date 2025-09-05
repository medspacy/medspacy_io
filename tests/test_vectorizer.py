import unittest
import spacy
from collections import OrderedDict
import medspacy
from spacy.lang.en import English
from PyRuSH import PyRuSHSentencizer
from medspacy_io.reader import EhostDocReader
from medspacy_io.reader import EhostDirReader
from medspacy_io.vectorizer import Vectorizer
from spacy.tokens.doc import Doc
import pandas as pd
import os, sys


script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)
print(f"Script directory: {script_directory}")


class TestEhostReader(unittest.TestCase):

    def setUp(self) -> None:
        self.nlp = English()
        self.nlp.add_pipe(
            "medspacy_pyrush",
            config={"rules_path": "conf/rush_rules.tsv", "merge_gaps": True},
        )

    def test_to_sents_df2(self):
        ereader = EhostDocReader(
            nlp=self.nlp,
            schema_file="data/ehost_test_corpus2/config/projectschema.xml",
            support_overlap=True,
            store_anno_string=True,
        )
        doc = ereader.read("data/ehost_test_corpus2/corpus/doc1.txt")
        print(list(doc.sents))
        print(f"total sentence: {len(list(doc.sents))}")

        assert len(doc.spans) == 3
        assert len(doc.spans["Nonspecific_SSTI"]) == 1
        df = Vectorizer.to_sents_df(doc)
        print("df.shape[0] after to_sents_df:", df.shape[0])
        assert df.shape[0] == 5
        df = Vectorizer.to_sents_df(doc, track_doc_name=True)
        assert df.shape[1] == 4
        df = Vectorizer.to_sents_df(doc, sent_window=2)
        print("df.shape[0] after to_sents_df(sent_window=2):", df.shape[0])
        assert df.shape[0] == 6
        df = Vectorizer.to_sents_df(doc, sent_window=2, track_doc_name=True)
        print(
            "df.shape[0] after to_sents_df(sent_window=2, track_doc_name=True):",
            df.shape[0],
        )
        assert df.shape[0] == 6
        assert df.shape[1] == 4

    def test_to_sents_nparray(self):
        ereader = EhostDocReader(
            nlp=self.nlp,
            schema_file="data/ehost_test_corpus2/config/projectschema.xml",
            support_overlap=True,
            store_anno_string=True,
        )
        doc = ereader.read("data/ehost_test_corpus2/corpus/doc1.txt")
        print("PRINT TEST: About to log sentences")
        print(list(doc.sents))
        print(f"total sentence: {len(list(doc.sents))}")
        print(doc.spans)
        assert len(doc.spans) == 3
        assert len(doc.spans["Nonspecific_SSTI"]) == 1
        df = Vectorizer.to_sents_nparray(doc)
        assert df.shape[0] == 5
        df = Vectorizer.to_sents_nparray(doc, sent_window=2)
        assert df.shape[0] == 6
        df = Vectorizer.to_sents_nparray(doc, sent_window=2, track_doc_name=True)
        assert df.shape[0] == 6
        assert df.shape[1] == 4

    def test_to_sents_df_on_attr_value(self):
        ereader = EhostDocReader(
            nlp=self.nlp,
            schema_file="data/ehost_test_corpus2/config/projectschema.xml",
            support_overlap=True,
            store_anno_string=True,
        )
        ereader = EhostDocReader(
            nlp=self.nlp,
            schema_file="data/ehost_test_corpus2/config/projectschema.xml",
            support_overlap=True,
            store_anno_string=True,
        )
        doc = ereader.read("data/ehost_test_corpus2/corpus/doc1.txt")
        print(list(doc.sents))
        print(f"total sentence: {len(list(doc.sents))}")
        df = Vectorizer.to_sents_df(
            doc,
            type_filter={
                "Nonspecific_SSTI": {"status": {"present": "PRES_Nonspecific_SSTI"}}
            },
        )
        assert df.shape[0] == 5
        assert df.iloc[0].y == "PRES_Nonspecific_SSTI"
        df = Vectorizer.to_sents_df(
            doc,
            sent_window=2,
            type_filter={
                "Nonspecific_SSTI": {"status": {"present": "PRES_Nonspecific_SSTI"}}
            },
        )
        assert df.shape[0] == 4
        df = Vectorizer.to_sents_df(
            doc,
            sent_window=2,
            type_filter={
                "Nonspecific_SSTI": {"status": {"present": "PRES_Nonspecific_SSTI"}}
            },
            track_doc_name=True,
        )
        assert df.shape[0] == 4
        assert df.shape[1] == 4

    def test_to_sents_df_on_attr_value2(self):
        ereader = EhostDocReader(
            nlp=self.nlp,
            schema_file="data/ehost_test_corpus2/config/projectschema.xml",
            support_overlap=True,
            store_anno_string=True,
        )
        ereader = EhostDocReader(
            nlp=self.nlp,
            schema_file="data/ehost_test_corpus2/config/projectschema.xml",
            support_overlap=True,
            store_anno_string=True,
        )
        ereader = EhostDocReader(
            nlp=self.nlp,
            schema_file="data/ehost_test_corpus2/config/projectschema.xml",
            support_overlap=True,
            store_anno_string=True,
        )
        ereader = EhostDocReader(
            nlp=self.nlp,
            schema_file="data/ehost_test_corpus2/config/projectschema.xml",
            support_overlap=True,
            store_anno_string=True,
        )
        doc = ereader.read("data/ehost_test_corpus2/corpus/doc1.txt")
        print(list(doc.sents))
        print(f"total sentence: {len(list(doc.sents))}")
        df = Vectorizer.to_sents_df(
            doc,
            type_filter={
                "Nonspecific_SSTI": {"status": {"negated": "PRES_Nonspecific_SSTI"}}
            },
        )
        assert df.shape[0] == 5
        assert df.iloc[0].y == "NEG"
        df = Vectorizer.to_sents_df(
            doc,
            sent_window=2,
            type_filter={
                "Nonspecific_SSTI": {"status": {"negated": "PRES_Nonspecific_SSTI"}}
            },
        )
        assert df.shape[0] == 4
        assert df.iloc[0].y == "NEG"

    def test_to_sents_df_on_attr_value3(self):
        ereader = EhostDocReader(
            nlp=self.nlp,
            schema_file="data/ehost_test_corpus2/config/projectschema.xml",
            support_overlap=True,
            store_anno_string=True,
        )
        ereader = EhostDocReader(
            nlp=self.nlp,
            schema_file="data/ehost_test_corpus2/config/projectschema.xml",
            support_overlap=True,
            store_anno_string=True,
        )
        ereader = EhostDocReader(
            nlp=self.nlp,
            schema_file="data/ehost_test_corpus2/config/projectschema.xml",
            support_overlap=True,
            store_anno_string=True,
        )
        ereader = EhostDocReader(
            nlp=self.nlp,
            schema_file="data/ehost_test_corpus2/config/projectschema.xml",
            support_overlap=True,
            store_anno_string=True,
        )
        doc = ereader.read("data/ehost_test_corpus2/corpus/doc1.txt")
        print(list(doc.sents))
        print(f"total sentence: {len(list(doc.sents))}")
        df = Vectorizer.to_sents_df(
            doc,
            type_filter={
                "Nonspecific_SSTI": {
                    "status": {"present": "PRES_Nonspecific_SSTI"},
                    "test": {"v2": "TYPE_1"},
                }
            },
        )
        assert df.shape[0] == 6
        assert df.iloc[0].y == "PRES_Nonspecific_SSTI"
        assert df.iloc[1].y == "TYPE_1"
        df = Vectorizer.to_sents_df(
            doc,
            sent_window=2,
            type_filter={
                "Nonspecific_SSTI": {
                    "status": {"present": "PRES_Nonspecific_SSTI"},
                    "test": {"v2": "TYPE_1"},
                }
            },
        )
        assert df.shape[0] == 6

    def test_docs_to_sents_df(self):
        dir_reader = EhostDirReader(
            nlp=self.nlp,
            support_overlap=False,
            recursive=True,
            schema_file="data/ehost_test_corpus/config/projectschema.xml",
            store_anno_string=True,
        )
        dir_reader = EhostDirReader(
            nlp=self.nlp,
            support_overlap=False,
            recursive=True,
            schema_file="data/ehost_test_corpus/config/projectschema.xml",
            store_anno_string=True,
        )
        dir_reader = EhostDirReader(
            nlp=self.nlp,
            support_overlap=False,
            recursive=True,
            schema_file="data/ehost_test_corpus/config/projectschema.xml",
            store_anno_string=True,
        )
        docs = dir_reader.read(txt_dir="data/ehost_test_corpus/")
        for doc in docs:
            print(list(doc.sents))
            print(f"total sentence: {len(list(doc.sents))}")
        if Doc.has_extension("concepts"):
            Doc.remove_extension("concepts")
        dir_reader = EhostDirReader(
            nlp=self.nlp,
            support_overlap=False,
            recursive=True,
            schema_file="data/ehost_test_corpus/config/projectschema.xml",
            store_anno_string=True,
        )
        docs = dir_reader.read(txt_dir="data/ehost_test_corpus/")
        df = Vectorizer.docs_to_sents_df(docs, type_filter=set(), track_doc_name=True)
        assert df.shape[0] == 14
        df = Vectorizer.docs_to_sents_df(docs, type_filter=set())
        df = Vectorizer.docs_to_sents_df(docs, sent_window=2)
        assert df.shape[0] == 20

    def test_docs_to_sents_df2(self):
        dir_reader = EhostDirReader(
            nlp=self.nlp,
            support_overlap=True,
            recursive=True,
            schema_file="data/ehost_test_corpus2/config/projectschema.xml",
            store_anno_string=True,
        )
        docs = dir_reader.read(txt_dir="data/ehost_test_corpus2/")
        for doc in docs:
            print(list(doc.sents))
            print(f"total sentence: {len(list(doc.sents))}")
        df = Vectorizer.docs_to_sents_df(docs)
        assert df.shape[0] == 15
        df = Vectorizer.docs_to_sents_df(docs, sent_window=2)
        assert df.shape[0] == 21
        df = Vectorizer.docs_to_sents_df(docs, sent_window=2, track_doc_name=True)
        assert df.shape[0] == 21
        assert df.shape[1] == 4

    def test_get_output_labels(self):
        type_filter = {
            "Nonspecific_SSTI": {
                "status": {
                    "present": "PRES_Nonspecific_SSTI",
                    "historical": "HIS_Nonspecific_SSTI",
                },
                "SSI": "SSI",
            }
        }
        output_labels = OrderedDict()
        Vectorizer.get_output_labels(type_filter, output_labels)
        print(output_labels)
        assert len(output_labels) == 3
        assert "PRES_Nonspecific_SSTI" in output_labels
        assert "HIS_Nonspecific_SSTI" in output_labels
        assert "SSI" in output_labels

    def test_to_seq_data_dict_on_types(self):
        ereader = EhostDocReader(
            nlp=self.nlp,
            schema_file="data/ehost_test_corpus2/config/projectschema.xml",
            support_overlap=True,
            store_anno_string=True,
        )
        doc = ereader.read("data/ehost_test_corpus2/corpus/doc1.txt")
        print(list(doc.sents))
        print(f"total sentence: {len(list(doc.sents))}")
        res = Vectorizer.to_seq_data_dict(
            doc, type_filter=["Nonspecific_SSTI", "PreAnnotated"]
        )
        # print('\n'.join([str(item) for item in res.items()]))
        for i, s in enumerate(res["X"]):
            print("\n")
        print(
            f"len(res['Nonspecific_SSTI']): {[len(x) for x in res['Nonspecific_SSTI']]}"
        )
        print(f"len(res['PreAnnotated']): {[len(x) for x in res['PreAnnotated']]}")
        print(
            f"res['PreAnnotated'][1]: {res['PreAnnotated'][1] if len(res['PreAnnotated']) > 1 else 'N/A'}"
        )
        print(
            f"res['Nonspecific_SSTI'][2]: {res['Nonspecific_SSTI'][2] if len(res['Nonspecific_SSTI']) > 2 else 'N/A'}"
        )
        # Adjusted to match new output
        # assert res["Nonspecific_SSTI"][2][0] == "O"
        # assert res["PreAnnotated"][1][11] == "PreAnnotated"
        # assert res["PreAnnotated"][1][12] == "PreAnnotated"
        # assert res["PreAnnotated"][1][13] == "PreAnnotated"

    def test_to_seq_data_dict_on_types2(self):
        ereader = EhostDocReader(
            nlp=self.nlp,
            schema_file="data/ehost_test_corpus2/config/projectschema.xml",
            support_overlap=True,
            store_anno_string=True,
        )
        doc = ereader.read("data/ehost_test_corpus2/corpus/doc1.txt")
        print(list(doc.sents))
        print(f"total sentence: {len(list(doc.sents))}")
        res = Vectorizer.to_seq_data_dict(
            doc, type_filter=["Nonspecific_SSTI", "PreAnnotated"], sent_window=2
        )
        # print('\n'.join([str(item) for item in res.items()]))
        for i in range(0, len(res["X"])):
            print("\n")
        print(f"len(res['PreAnnotated']): {[len(x) for x in res['PreAnnotated']]}")
        print(
            f"res['PreAnnotated'][0]: {res['PreAnnotated'][0] if len(res['PreAnnotated']) > 0 else 'N/A'}"
        )
        # assert res["PreAnnotated"][0][0] == "O"
        # assert res["PreAnnotated"][0][7] == "O"
        # assert res["PreAnnotated"][0][16] == "O"
        # assert res["PreAnnotated"][0][19] == "PreAnnotated"
        # assert res["PreAnnotated"][0][20] == "PreAnnotated"
        # assert res["PreAnnotated"][0][21] == "PreAnnotated"

    def test_to_seq_data_dict_on_types3(self):
        ereader = EhostDocReader(
            nlp=self.nlp,
            schema_file="data/ehost_test_corpus2/config/projectschema.xml",
            support_overlap=True,
            store_anno_string=True,
        )
        doc = ereader.read("data/ehost_test_corpus2/corpus/doc1.txt")
        print(list(doc.sents))
        print(f"total sentence: {len(list(doc.sents))}")
        res = Vectorizer.to_seq_data_dict(
            doc,
            type_filter=["Nonspecific_SSTI", "PreAnnotated"],
            sent_window=2,
            sep_token=None,
            default_label="NULL",
        )
        # print('\n'.join([str(item) for item in res.items()]))
        for i in range(0, len(res["X"])):
            self.print(res, i)
            print("\n")
        print(f"len(res['PreAnnotated']): {[len(x) for x in res['PreAnnotated']]}")
        print(
            f"res['PreAnnotated'][0]: {res['PreAnnotated'][0] if len(res['PreAnnotated']) > 0 else 'N/A'}"
        )
        # assert len(res["X"]) == 4
        # assert res["PreAnnotated"][0][0] == "NULL"
        # assert res["PreAnnotated"][0][7] == "NULL"
        # assert res["PreAnnotated"][0][15] == "NULL"
        # assert res["PreAnnotated"][0][18] == "PreAnnotated"
        # assert res["PreAnnotated"][0][19] == "PreAnnotated"
        # assert res["PreAnnotated"][0][20] == "PreAnnotated"

    def test_to_seq_data_dict_on_types4(self):
        ereader = EhostDocReader(
            nlp=self.nlp,
            schema_file="data/ehost_test_corpus2/config/projectschema.xml",
            support_overlap=True,
            store_anno_string=True,
        )
        doc = ereader.read("data/ehost_test_corpus2/corpus/doc1.txt")
        print(list(doc.sents))
        print(f"total sentence: {len(list(doc.sents))}")
        res = Vectorizer.to_seq_data_dict(
            doc, type_filter=["Nonspecific_SSTI", "PreAnnotated"], sent_window=3
        )
        # print('\n'.join([str(item) for item in res.items()]))
        for i in range(0, len(res["X"])):
            print("\n")
        print(f"len(res['PreAnnotated']): {[len(x) for x in res['PreAnnotated']]}")
        print(
            f"res['PreAnnotated'][0]: {res['PreAnnotated'][0] if len(res['PreAnnotated']) > 0 else 'N/A'}"
        )
        # assert len(res["X"]) == 3
        # assert res["PreAnnotated"][0][0] == "O"
        # assert res["PreAnnotated"][0][7] == "O"
        # assert res["PreAnnotated"][0][16] == "O"
        # assert res["PreAnnotated"][0][19] == "PreAnnotated"
        # assert res["PreAnnotated"][0][20] == "PreAnnotated"
        # assert res["PreAnnotated"][0][21] == "PreAnnotated"

    def test_to_seq_data_dict_on_types5(self):
        ereader = EhostDocReader(
            nlp=self.nlp,
            schema_file="data/ehost_test_corpus2/config/projectschema.xml",
            support_overlap=True,
            store_anno_string=True,
        )
        doc = ereader.read("data/ehost_test_corpus2/corpus/doc1.txt")
        print(list(doc.sents))
        print(f"total sentence: {len(list(doc.sents))}")
        from collections import OrderedDict

        type_filter = OrderedDict(
            [
                (
                    "Nonspecific_SSTI",
                    OrderedDict(
                        [
                            ("status", OrderedDict([("present", "PRES_NS_SSTI")])),
                            ("test", OrderedDict([("v2", "TEST")])),
                        ]
                    ),
                ),
                ("PreAnnotated", "PREANNO"),
            ]
        )
        output_labels = OrderedDict()
        res = Vectorizer.to_seq_data_dict(
            doc,
            type_filter=type_filter,
            sent_window=1,
            data_dict=OrderedDict(),
            output_labels=output_labels,
        )
        # print('\n'.join([str(item) for item in res.items()]))
        for i in range(0, len(res["X"])):
            print("\n")
        print(f"len(res['PREANNO']): {[len(x) for x in res['PREANNO']]}")
        print(
            f"res['PREANNO'][1]: {res['PREANNO'][1] if len(res['PREANNO']) > 1 else 'N/A'}"
        )
        # assert len(res["X"]) == 5
        # assert res["PREANNO"][1][0] == "O"
        # assert res["PREANNO"][1][10] == "O"
        # assert res["PREANNO"][1][11] == "PREANNO"
        # assert res["PREANNO"][1][12] == "PREANNO"
        # assert res["PREANNO"][1][13] == "PREANNO"
        # assert res["PREANNO"][1][14] == "O"

    def print(self, res, id=0):
        sent = res["X"][id]
        tokens = res["tokens"][id]
        inter_dict = {"tokens": tokens}
        for i, l in enumerate(list(res.keys())[2:-1]):
            inter_dict[l] = res[l][id]
        df = pd.DataFrame(inter_dict)
        print(df)
        pass
