from typing import List, Set, Dict

from spacy.tokens.doc import Doc
from quicksectx import IntervalTree
from spacy.tokens.span import Span


class Vectorizer:
    def __init(self):
        pass

    def to_sents_df(self, doc: Doc, sent_window: int = 1, type_filter: Set[str] = set(),
                    default_label: str = "NEG", data_dict={'X': [], 'concept': [], 'y': []}):
        import pandas as pd
        data_dict = self.to_data_dict(doc, sent_window=sent_window, type_filter=type_filter,
                                      default_label=default_label, data_dict=data_dict)
        df = pd.DataFrame(data_dict)
        return df

    def to_sents_nparray(self, doc: Doc, sent_window: int = 1, type_filter: Set[str] = set(),
                         default_label: str = "NEG", data_dict={'X': [], 'concept': [], 'y': []}):
        import numpy as np
        data_dict = self.to_data_dict(doc, sent_window=sent_window, type_filter=type_filter,
                                      default_label=default_label, data_dict=data_dict)
        rows = []
        for i in range(0, len(data_dict['X'])):
            rows.append([data_dict['X'][i], data_dict['concept'][i], data_dict['y'][i]])
        sents_nparray = np.array(rows)
        return sents_nparray

    def to_data_dict(self, doc: Doc, sent_window: int = 1, type_filter: Set[str] = set(),
                     default_label: str = "NEG", data_dict={'X': [], 'concept': [], 'y': []}) -> Dict:
        sent_idx = IntervalTree()
        print('\n---\n'.join([str(s) for s in doc.sents]))
        sents = list(doc.sents)
        context_sents = []
        for i in range(0, len(sents) - sent_window + 1):
            begin_sent = sents[i]
            end_sent = sents[i + sent_window - 1]
            sent_idx.add(begin_sent.start, end_sent.end, len(context_sents))
            context_sents.append(sents[i:i + sent_window])

        concepts = []
        if hasattr(doc._, "concepts"):
            for type in doc._.concepts:
                if len(type_filter) == 0 or type in type_filter:
                    concepts.extend(doc._.concepts[type])
        else:
            concepts = [ent for ent in doc.ents if (len(type_filter) == 0 or ent.label in type_filter)]

        labeled_sents_id = set()
        for concept in concepts:
            context_sents_ids = sent_idx.search(concept.start, concept.end)
            for id in context_sents_ids:
                labeled_sents_id.add(id.data)
                context = context_sents[id.data]
                if concept.start >= context[0].start and concept.end <= context[-1].end:
                    data_dict['X'].append(' '.join([str(s) for s in context]))
                    data_dict['y'].append(concept.label_)
                    data_dict['concept'].append(str(concept))
        for i, context in enumerate(context_sents):
            if i not in labeled_sents_id:
                data_dict['X'].append(' '.join([str(s) for s in context]))
                data_dict['y'].append(default_label)
                data_dict['concept'].append('')
        return data_dict

    def docs_to_sents_data_dict(self, docs: List[Doc], sent_window: int = 1, type_filter: Set[str] = set(),
                                default_label: str = "NEG"):
        data_dict = {'X': [], 'concept': [], 'y': []}
        for doc in docs:
            self.to_data_dict(doc, sent_window=sent_window, type_filter=type_filter,
                              default_label=default_label, data_dict=data_dict)
        return data_dict

    def docs_to_sents_df(self, docs: List[Doc], sent_window: int = 1, type_filter: Set[str] = set(),
                         default_label: str = "NEG"):
        import pandas as pd
        data_dict = self.docs_to_sents_data_dict(docs, sent_window=sent_window, type_filter=type_filter,
                                                 default_label=default_label)
        df = pd.DataFrame(data_dict)
        return df

    def docs_to_sents_nparray(self, docs: List[Doc], sent_window: int = 1, type_filter: Set[str] = set(),
                              default_label: str = "NEG"):
        import numpy as np
        data_dict = self.docs_to_sents_data_dict(docs, sent_window=sent_window, type_filter=type_filter,
                                                 default_label=default_label)
        rows = []
        for i in range(0, len(data_dict['X'])):
            rows.append([data_dict['X'][i], data_dict['concept'][i], data_dict['y'][i]])
        sents_nparray = np.array(rows)
        return sents_nparray
