from typing import List, Set, Dict

import numpy as np
import pandas as pd
from numpy.core.multiarray import ndarray
from quicksectx import IntervalTree
from spacy.tokens.doc import Doc


class Vectorizer:

    @staticmethod
    def to_sents_df(doc: Doc, sent_window: int = 1, type_filter: Set[str] = set(),
                    default_label: str = "NEG") -> pd.DataFrame:
        """
        Convert a SpaCy doc into pandas DataFrame. Assuming the doc has been labeled based on concepts(snippets), Vectorizer
        extends the input to the concepts' context sentences (depends on the sent_window size), generate labeled context
        sentences data, and return a pandas DataFrame (with three columns: 'X'---the text of context sentences,'concepts'---
        the text of labeled concepts, 'y'---label)
        @param doc: a SpaCy Doc
        @param sent_window: The window size (in sentences) around the target concept that need to be pulled
        @param type_filter: Whether and what types of annotation will be used generate the output DataFrame
        @param default_label: If there is no labeled concept in the context sentences, label it with this default_label
        @return: a pandas DataFrame
        """
        data_dict = Vectorizer.to_data_dict(doc, sent_window=sent_window, type_filter=type_filter,
                                            default_label=default_label, data_dict={'X': [], 'concept': [], 'y': []})
        df = pd.DataFrame(data_dict)
        return df

    @staticmethod
    def to_sents_nparray(doc: Doc, sent_window: object = 1, type_filter: object = set(),
                         default_label: object = "NEG") -> ndarray:
        """
        Convert a SpaCy doc into numpy array. Assuming the doc has been labeled based on concepts(snippets), Vectorizer
        extends the input to the concepts' context sentences (depends on the sent_window size), generate labeled context
        sentences data, and return a numpy array (with three columns: 'X'---the text of context sentences,'concepts'---
        the text of labeled concepts, 'y'---label)
        @param doc: a SpaCy Doc
        @param sent_window: The window size (in sentences) around the target concept that need to be pulled
        @param type_filter: Whether and what types of annotation will be used generate the output DataFrame
        @param default_label: If there is no labeled concept in the context sentences, label it with this default_label
        @return: a numpy array
        """
        data_dict = Vectorizer.to_data_dict(doc, sent_window=sent_window, type_filter=type_filter,
                                            default_label=default_label, data_dict={'X': [], 'concept': [], 'y': []})
        rows = []
        for i in range(0, len(data_dict['X'])):
            rows.append([data_dict['X'][i], data_dict['concept'][i], data_dict['y'][i]])
        sents_nparray = np.array(rows)
        return sents_nparray

    @staticmethod
    def to_data_dict(doc: Doc, sent_window: object = 1, type_filter: object = set(),
                     default_label: object = "NEG", data_dict: object = {'X': [], 'concept': [], 'y': []}) -> Dict:
        """
        Convert a SpaCy doc into a labeled data dictionary. Assuming the doc has been labeled based on concepts(snippets), Vectorizer
        extends the input to the concepts' context sentences (depends on the sent_window size), generate labeled context
        sentences data, and return a dictionary (with three keys: 'X'---the text of context sentences,'concepts'---
        the text of labeled concepts, 'y'---label)
        @param doc: a SpaCy Doc
        @param sent_window: The window size (in sentences) around the target concept that need to be pulled
        @param type_filter: Whether and what types of annotation will be used generate the output DataFrame
        @param default_label: If there is no labeled concept in the context sentences, label it with this default_label
        @return: a dictionary
        """
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

    @staticmethod
    def docs_to_sents_data_dict(docs: List[Doc], sent_window: int = 1, type_filter: Set[str] = set(),
                                default_label: str = "NEG"):
        """
        Convert a list of SpaCy docs into a labeled data dictionary. Assuming the doc has been labeled based on concepts(snippets), Vectorizer
        extends the input to the concepts' context sentences (depends on the sent_window size), generate labeled context
        sentences data, and return a dictionary (with three keys: 'X'---the text of context sentences,'concepts'---
        the text of labeled concepts, 'y'---label)
        @param doc: a list of SpaCy Docs
        @param sent_window: The window size (in sentences) around the target concept that need to be pulled
        @param type_filter: Whether and what types of annotation will be used generate the output DataFrame
        @param default_label: If there is no labeled concept in the context sentences, label it with this default_label
        @return: a dictionary
        """
        data_dict = {'X': [], 'concept': [], 'y': []}
        for doc in docs:
            Vectorizer.to_data_dict(doc, sent_window=sent_window, type_filter=type_filter,
                                    default_label=default_label, data_dict=data_dict)
        return data_dict

    @staticmethod
    def docs_to_sents_df(docs: List[Doc], sent_window: int = 1, type_filter: Set[str] = set(),
                         default_label: str = "NEG") -> pd.DataFrame:
        """
        Convert a list of SpaCy docs into pandas DataFrame. Assuming the doc has been labeled based on concepts(snippets), Vectorizer
        extends the input to the concepts' context sentences (depends on the sent_window size), generate labeled context
        sentences data, and return a pandas DataFrame (with three columns: 'X'---the text of context sentences,'concepts'---
        the text of labeled concepts, 'y'---label)
        @param doc: a list of SpaCy Doc
        @param sent_window: The window size (in sentences) around the target concept that need to be pulled
        @param type_filter: Whether and what types of annotation will be used generate the output DataFrame
        @param default_label: If there is no labeled concept in the context sentences, label it with this default_label
        @return: a pandas DataFrame
        """
        data_dict = Vectorizer.docs_to_sents_data_dict(docs, sent_window=sent_window, type_filter=type_filter,
                                                       default_label=default_label)
        df = pd.DataFrame(data_dict)
        return df

    @staticmethod
    def docs_to_sents_nparray(self, docs: List[Doc], sent_window: int = 1, type_filter: Set[str] = set(),
                              default_label: str = "NEG"):
        """
        Convert a list of SpaCy docs into numpy array. Assuming the doc has been labeled based on concepts(snippets), Vectorizer
        extends the input to the concepts' context sentences (depends on the sent_window size), generate labeled context
        sentences data, and return a numpy array (with three columns: 'X'---the text of context sentences,'concepts'---
        the text of labeled concepts, 'y'---label)
        @param doc: a list of SpaCy Doc
        @param sent_window: The window size (in sentences) around the target concept that need to be pulled
        @param type_filter: Whether and what types of annotation will be used generate the output DataFrame
        @param default_label: If there is no labeled concept in the context sentences, label it with this default_label
        @return: a numpy array
        """
        data_dict = Vectorizer.docs_to_sents_data_dict(docs, sent_window=sent_window, type_filter=type_filter,
                                                       default_label=default_label)
        rows = []
        for i in range(0, len(data_dict['X'])):
            rows.append([data_dict['X'][i], data_dict['concept'][i], data_dict['y'][i]])
        sents_nparray = np.array(rows)
        return sents_nparray
