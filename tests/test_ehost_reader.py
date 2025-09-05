import unittest


import sys

# sys.path.append(
#     "/Users/u6022257/Documents/medspacy_io/medspacy_io/reader"
# )  # need to uninstall medspacy-io to test the package code.
# # sys.path.append("../") #need to uninstall medspacy-io to test the package code.
# # sys.path.append("../medspacy")


from spacy.lang.en import English
from spacy.tokens import Doc
from pathlib import Path
from medspacy_io.reader.ehost_reader import EhostDirReader
from medspacy_io.reader.ehost_reader import EhostDocReader
import os, sys
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

test_path = Path(__file__).parent.absolute()


class test_eHost_reader(unittest.TestCase):

    def test_reader_initial(self):
        ereader = EhostDocReader(nlp=English())
        assert hasattr(ereader, "use_adjudication")
        assert not ereader.use_adjudication

    def test_parse_to_dicts(self):
        if Doc.has_extension("concepts"):
            Doc.remove_extension("concepts")
        ereader = EhostDocReader(nlp=English())
        spans, classes, attributes, relations = ereader.parse_to_dicts(
            os.path.join(
                test_path, "data/ehost_test_corpus/saved/doc1.txt.knowtator.xml"
            )
        )
        assert len(spans) == 7
        assert len(classes) == 8
        assert len(attributes) == 6
        assert len(relations) == 1

    def test_set_attributes(self):
        if Doc.has_extension("concepts"):
            Doc.remove_extension("concepts")
        ereader = EhostDocReader(
            nlp=English(),
            schema_file=os.path.join(
                test_path, "data/ehost_test_corpus/config/projectschema.xml"
            ),
        )
        doc = ereader.read(
            os.path.join(test_path, "data/ehost_test_corpus/corpus/doc1.txt")
        )
        for ent in doc.ents:
            assert hasattr(ent._, "ANNOT_status")
            assert hasattr(ent._, "ANNOT_rel_attr")
        # assert (hasattr(doc[1:2]._, 'status'))
        # assert (doc[1:2]._.status == 'present')

    def test_read(self):
        if Doc.has_extension("concepts"):
            Doc.remove_extension("concepts")
        ereader = EhostDocReader(
            nlp=English(),
            schema_file=os.path.join(
                test_path, "data/ehost_test_corpus/config/projectschema.xml"
            ),
        )
        doc = ereader.read(
            os.path.join(test_path, "data/ehost_test_corpus/corpus/doc1.txt")
        )
        self.eval(doc)

    def test_read_doc_name(self):
        ereader = EhostDocReader(
            nlp=English(),
            schema_file=os.path.join(
                test_path, "data/ehost_test_corpus/config/projectschema.xml"
            ),
        )
        doc = ereader.read(
            os.path.join(test_path, "data/ehost_test_corpus/corpus/doc1.txt")
        )
        assert doc._.doc_name == "doc1.txt"
        ereader.doc_name_depth = 1
        doc = ereader.read(
            os.path.join(test_path, "data/ehost_test_corpus/corpus/doc1.txt")
        )
        assert doc._.doc_name == r"corpus/doc1.txt"
        ereader = EhostDocReader(
            nlp=English(),
            schema_file=os.path.join(
                test_path, "data/ehost_test_corpus/config/projectschema.xml"
            ),
            doc_name_depth=2,
        )
        doc = ereader.read(
            os.path.join(test_path, "data/ehost_test_corpus/corpus/doc1.txt")
        )
        assert doc._.doc_name == r"ehost_test_corpus/corpus/doc1.txt"

    # spans are not in doc.spans
    # def test_read_overlap_new_version(self): #PASSED
    #     if Doc.has_extension("concepts"):
    #         Doc.remove_extension("concepts")
    #     ereader = EhostDocReader(nlp=English(), schema_file=os.path.join(test_path, 'data/ehost_test_corpus2/config/projectschema.xml'),support_overlap=True,new_version=True)
    #     doc = ereader.read(os.path.join(test_path, 'data/ehost_test_corpus2/corpus/doc1.txt'))
    #     assert (len(doc.spans) == 3)
    #     assert (len(doc.spans['PreAnnotated']) == 1) #still save the spans in doc._.concepts
    #     doc = ereader.read(os.path.join(test_path, 'data/ehost_test_corpus2/corpus/doc2.txt'))
    #     assert (len(doc.spans) == 7)
    #     assert (len(doc.spans['Exclusions']) == 2)
    #     assert (len(doc.spans['Doc_Level_Purulence_Assessment']) == 2)
    #

    def test_read_overlap(self):  # spanGroup
        if Doc.has_extension("concepts"):
            Doc.remove_extension("concepts")
        ereader = EhostDocReader(
            nlp=English(),
            schema_file=os.path.join(
                test_path, "data/ehost_test_corpus2/config/projectschema.xml"
            ),
            support_overlap=True,
        )
        doc = ereader.read(
            os.path.join(test_path, "data/ehost_test_corpus2/corpus/doc1.txt")
        )
        assert len(doc.spans) == 3
        assert len(doc.spans["PreAnnotated"]) == 1
        doc = ereader.read(
            os.path.join(test_path, "data/ehost_test_corpus2/corpus/doc2.txt")
        )
        assert len(doc.spans) == 7
        assert len(doc.spans["Exclusions"]) == 2
        assert len(doc.spans["Doc_Level_Purulence_Assessment"]) == 2

    def test_read_doc_name(self):
        ereader = EhostDocReader(
            nlp=English(),
            schema_file=os.path.join(
                test_path, "data/ehost_test_corpus/config/projectschema.xml"
            ),
        )
        doc = ereader.read(
            os.path.join(test_path, "data/ehost_test_corpus/corpus/doc1.txt")
        )
        assert doc._.doc_name == "doc1.txt"
        ereader.doc_name_depth = 1
        doc = ereader.read(
            os.path.join(test_path, "data/ehost_test_corpus/corpus/doc1.txt")
        )
        assert Path(doc._.doc_name).stem == r"doc1"
        ereader = EhostDocReader(
            nlp=English(),
            schema_file=os.path.join(
                test_path, "data/ehost_test_corpus/config/projectschema.xml"
            ),
            doc_name_depth=2,
        )
        doc = ereader.read(
            os.path.join(test_path, "data/ehost_test_corpus/corpus/doc1.txt")
        )
        print(Path(doc._.doc_name).stem)
        assert Path(doc._.doc_name).stem == r"doc1"

    def test_check_spans(self):  # none Overlapped spans
        if Doc.has_extension("concepts"):
            Doc.remove_extension("concepts")
        ereader = EhostDocReader(
            nlp=English(),
            schema_file=os.path.join(
                test_path, "data/ehost_test_corpus/config/projectschema.xml"
            ),
            support_overlap=False,
            store_anno_string=True,
            encoding="UTF8",
            log_level="DEBUG",
        )
        doc = ereader.read(
            os.path.join(test_path, "data/ehost_test_corpus/corpus/doc2.txt")
        )
        for span in doc.ents:
            print(span._.span_txt, "<>", span)
            assert span._.span_txt.replace("\n", " ") in str(span).replace("\n", " ")

    def test_check_spans2(self):  # overlapped spans
        if Doc.has_extension("concepts"):
            Doc.remove_extension("concepts")
        ereader = EhostDocReader(
            nlp=English(),
            schema_file=os.path.join(
                test_path, "data/ehost_test_corpus2/config/projectschema.xml"
            ),
            support_overlap=True,
            store_anno_string=True,
            log_level="DEBUG",
        )
        doc = ereader.read(
            os.path.join(test_path, "data/ehost_test_corpus2/corpus/doc2.txt")
        )
        for spans in doc.spans.values():
            for span in spans:
                print(span._.span_txt, "<>", span)
                assert span._.span_txt.replace("\n", " ") in str(span).replace(
                    "\n", " "
                )

    def test_dir_reader(self):
        if Doc.has_extension("concepts"):
            Doc.remove_extension("concepts")
        dir_reader = EhostDirReader(
            nlp=English(),
            recursive=True,
            schema_file=os.path.join(
                test_path, "data/ehost_test_corpus/config/projectschema.xml"
            ),
        )
        docs = dir_reader.read(
            txt_dir=os.path.join(test_path, "data/ehost_test_corpus/")
        )
        assert len(docs) == 2
        for doc in docs:
            self.eval(doc)

    def eval(self, doc):
        assert len(doc.ents) == 7
        assert str(doc.ents[0]) == "CHIEF"
        assert str(doc.ents[1]) == "Abdominal pain"
        assert str(doc.ents[2]) == "PRESENT"
        assert str(doc.ents[3]) == "patient"
        # there is a slightly mismatch of the token, because SpaCy tokenize '71-year-old' into
        # '71-year', '-', 'old', EhostDocReader adjust the annotation spans to align with the tokens
        assert str(doc.ents[4]) == "year-old"
        assert str(doc.ents[5]) == "X. The patient"
        assert str(doc.ents[6]) == "presented"
        assert doc.ents[0].label_ == "Doc_Level_Purulence_Assessment"
        assert doc.ents[1].label_ == "Purulent"
        assert doc.ents[2].label_ == "Non-Purulent"
        assert doc.ents[3].label_ == "Incision_and_Drainage"
        assert doc.ents[4].label_ == "PreAnnotated"
        assert doc.ents[5].label_ == "Nonspecific_SSTI"
        assert doc.ents[6].label_ == "Exclusions"

    def test_relation_reader(self):
        ereader = EhostDocReader(
            nlp=English(),
            schema_file=os.path.join(
                test_path, "data/ehost_test_corpus3_overlap/config/projectschema.xml"
            ),
            support_overlap=True,
        )
        doc = ereader.read(
            os.path.join(test_path, "data/ehost_test_corpus3_overlap/corpus/18305.txt")
        )
        assert len(doc._.relations) == 1
        for r in doc._.relations:
            assert r[2] == "symptom_to_symptom_section"  # relation label
            assert r[0]._.annotation_id == "EHOST_Instance_8"  # source
            assert r[1]._.annotation_id == "EHOST_Instance_1"  # target
