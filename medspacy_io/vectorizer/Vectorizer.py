from typing import List, Set, Dict, Union
import numpy as np
import pandas as pd
from numpy.core.multiarray import ndarray
from quicksectx import IntervalTree
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span


class Vectorizer:

    @staticmethod
    def to_sents_df(doc: Doc, sent_window: int = 1, type_filter: Union[Set[str], Dict] = set(),
                    default_label: str = "NEG", track_doc_name: bool = False) -> pd.DataFrame:
        """
        Convert a SpaCy doc into pandas DataFrame. Assuming the doc has been labeled based on concepts(snippets), Vectorizer
        extends the input to the concepts' context sentences (depends on the sent_window size), generate labeled context
        sentences data, and return a pandas DataFrame (with three columns: 'X'---the text of context sentences,'concepts'---
        the text of labeled concepts, 'y'---label)
        @param doc: a SpaCy Doc
        @param sent_window: The window size (in sentences) around the target concept that need to be pulled
        @param type_filter: Specify whether and what types of annotation will be used generate the output DataFrame, this
        parameter can be defined as a set (only concept names are included) or a dictionary (where attributes and values
        can be included), which maps a matched concept (string and its context string) to a new value in "y"
        column in the output. The structure of expected dictionary will be:
        concept_type->attr1->value1->...(other attr->value pairs if needed)->mapped key name
        @param default_label: If there is no labeled concept in the context sentences, label it with this default_label
        @param track_doc_name: Whether add doc name to an additional column to track the output vectors
        @return: a pandas DataFrame

        """
        data_dict = {'X': [], 'concept': [], 'y': [], 'doc_name': []} if track_doc_name else {'X': [], 'concept': [],
                                                                                              'y': []}
        data_dict = Vectorizer.to_data_dict(doc, sent_window=sent_window, type_filter=type_filter,
                                            default_label=default_label, data_dict=data_dict)
        df = pd.DataFrame(data_dict)
        return df

    @staticmethod
    def to_sents_nparray(doc: Doc, sent_window: int = 1, type_filter: Union[Set[str], Dict] = set(),
                         default_label: str = "NEG", track_doc_name: bool = False) -> ndarray:
        """
        Convert a SpaCy doc into numpy array. Assuming the doc has been labeled based on concepts(snippets), Vectorizer
        extends the input to the concepts' context sentences (depends on the sent_window size), generate labeled context
        sentences data, and return a numpy array (with three columns: 'X'---the text of context sentences,'concepts'---
        the text of labeled concepts, 'y'---label)
        @param doc: a SpaCy Doc
        @param sent_window: The window size (in sentences) around the target concept that need to be pulled
        @param type_filter: Specify whether and what types of annotation will be used generate the output array, this
        parameter can be defined as a set (only concept names are included) or a dictionary (where attributes and values
        can be included), which maps a matched concept (string and its context string) to a new value in "y"
        column in the output. The structure of expected dictionary will be:
        concept_type->attr1->value1->...(other attr->value pairs if needed)->mapped key name
        @param default_label: If there is no labeled concept in the context sentences, label it with this default_label
        @param track_doc_name: Whether add doc name to an additional column to track the output vectors
        @return: a numpy array
        """
        data_dict = {'X': [], 'concept': [], 'y': [], 'doc_name': []} if track_doc_name else {'X': [], 'concept': [],
                                                                                              'y': []}
        data_dict = Vectorizer.to_data_dict(doc, sent_window=sent_window, type_filter=type_filter,
                                            default_label=default_label, data_dict=data_dict)
        rows = []
        for i in range(0, len(data_dict['X'])):
            if track_doc_name:
                rows.append([data_dict['X'][i], data_dict['concept'][i], data_dict['y'][i], data_dict['doc_name'][i]])
            else:
                rows.append([data_dict['X'][i], data_dict['concept'][i], data_dict['y'][i]])
        sents_nparray = np.array(rows)
        return sents_nparray

    @staticmethod
    def to_data_dict(doc: Doc, sent_window: int = 1, type_filter: Union[Set[str], Dict] = set(),
                     default_label: str = "NEG", data_dict: dict = {'X': [], 'concept': [], 'y': []}) -> Dict:
        """
        Convert a SpaCy doc into a labeled data dictionary. Assuming the doc has been labeled based on concepts(snippets), Vectorizer
        extends the input to the concepts' context sentences (depends on the sent_window size), generate labeled context
        sentences data, and return a dictionary (with three keys: 'X'---the text of context sentences,'concepts'---
        the text of labeled concepts, 'y'---label)
        @param doc: a SpaCy Doc
        @param sent_window: The window size (in sentences) around the target concept that need to be pulled
        @param type_filter: Specify whether and what types of annotation will be used generate the output DataFrame, this
        parameter can be defined as a set (only concept names are included) or a dictionary (where attributes and values
        can be included), which maps a matched concept (string and its context string) to a new value in "y"
        column in the output. The structure of expected dictionary will be:
        concept_type->attr1->value1->...(other attr->value pairs if needed)->mapped key name
        @param default_label: If there is no labeled concept in the context sentences, label it with this default_label
        @param data_dict: a dictionary to hold the output and pass on across documents, so that a corpus can be aggregated
        @param sent_idx: an IntervalTree built with all sentences in the doc
        @param context_sents: a 2-d list of sentences with predefined window size.
        @return: a dictionary
        """
        sent_idx = IntervalTree()
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

        get_doc_name = 'doc_name' in data_dict
        doc_name = doc._.doc_name if get_doc_name else ''

        if isinstance(type_filter, Set):
            data_dict = Vectorizer.to_data_dict_on_types(concepts=concepts,
                                                         type_filter=type_filter,
                                                         default_label=default_label,
                                                         data_dict=data_dict,
                                                         sent_idx=sent_idx, context_sents=context_sents,
                                                         doc_name=doc_name)
        elif isinstance(type_filter, Dict):
            if len(type_filter) == 0:
                data_dict = Vectorizer.to_data_dict_on_types(concepts=concepts,
                                                             default_label=default_label,
                                                             data_dict=data_dict,
                                                             sent_idx=sent_idx, context_sents=context_sents,
                                                             doc_name=doc_name)
            else:
                data_dict = Vectorizer.to_data_dict_on_type_attr_values(concepts=concepts,
                                                                        type_filter=type_filter,
                                                                        default_label=default_label,
                                                                        data_dict=data_dict,
                                                                        sent_idx=sent_idx, context_sents=context_sents,
                                                                        doc_name=doc_name)
        else:
            raise TypeError(
                'The arg: "type_filter" needs to be either a set of concept names or a dictionary. Not a {}:\n\t{}'.format(
                    type(type_filter), str(type_filter)))
        return data_dict

    @staticmethod
    def to_data_dict_on_types(concepts: List[Span], type_filter: Set = set(),
                              default_label: str = "NEG", data_dict: dict = {'X': [], 'concept': [], 'y': []},
                              sent_idx: IntervalTree = None, context_sents: List[List[Span]] = None,
                              doc_name: str = '') -> Dict:
        """
        Convert a SpaCy doc into a labeled data dictionary. Assuming the doc has been labeled based on concepts(snippets), Vectorizer
        extends the input to the concepts' context sentences (depends on the sent_window size), generate labeled context
        sentences data, and return a dictionary (with three keys: 'X'---the text of context sentences,'concepts'---
        the text of labeled concepts, 'y'---label)
        @param concepts: a list of concepts (in Span type)
        @param type_filter: a set of type names that need to be included to be vectorized
        @param default_label: If there is no labeled concept in the context sentences, label it with this default_label
        @param data_dict: a dictionary to hold the output and pass on across documents, so that a corpus can be aggregated
        @param sent_idx: an IntervalTree built with all sentences in the doc
        @param context_sents: a 2-d list of sentences with predefined window size.
        @param doc_name: doc file name (for tracking purpose)
        @return: a dictionary
        """
        if sent_idx is None or context_sents is None:
            return data_dict
        get_doc_name = 'doc_name' in data_dict
        labeled_sents_id = set()
        for concept in concepts:
            if len(type_filter) > 0 and concept.label_ not in type_filter:
                continue
            context_sents_ids = sent_idx.search(concept.start, concept.end)
            for id in context_sents_ids:
                labeled_sents_id.add(id.data)
                context = context_sents[id.data]
                if concept.start >= context[0].start and concept.end <= context[-1].end:
                    data_dict['X'].append(' '.join([str(s) for s in context]))
                    data_dict['y'].append(concept.label_)
                    data_dict['concept'].append(str(concept))
                    if get_doc_name:
                        data_dict['doc_name'].append(doc_name)
        for i, context in enumerate(context_sents):
            if i not in labeled_sents_id:
                data_dict['X'].append(' '.join([str(s) for s in context]))
                data_dict['y'].append(default_label)
                data_dict['concept'].append('')
                if get_doc_name:
                    data_dict['doc_name'].append(doc_name)
        return data_dict

    @staticmethod
    def to_data_dict_on_type_attr_values(concepts: List[Span], type_filter: Dict = dict(),
                                         default_label: str = "NEG",
                                         data_dict: dict = {'X': [], 'concept': [], 'y': []},
                                         sent_idx: IntervalTree = None, context_sents: List[List[Span]] = None,
                                         doc_name: str = '') -> Dict:
        """
        Convert a SpaCy doc into a labeled data dictionary. Assuming the doc has been labeled based on concepts(snippets), Vectorizer
        extends the input to the concepts' context sentences (depends on the sent_window size), generate labeled context
        sentences data, and return a dictionary (with three keys: 'X'---the text of context sentences,'concepts'---
        the text of labeled concepts, 'y'---label)
        @param concepts: a list of concepts (in Span type)
        @param type_filter: Whether and what types of annotation with what attribute values will be used generate the
        output DataFrame, this parameter is defined as a dictionary (where attributes and values can be included), which
        maps a matched concept (string and its context string) to a new value in "y" column in the output. The
        structure of expected dictionary will be:
        concept_type->attr1->value1->...(other attr->value pairs if needed)->mapped key name
        @param default_label: If there is no labeled concept in the context sentences, label it with this default_label
        @param data_dict: a dictionary to hold the output and pass on across documents, so that a corpus can be aggregated
        @param sent_idx: an IntervalTree built with all sentences in the doc
        @param context_sents: a 2-d list of sentences with predefined window size.
        @param doc_name: doc file name (for tracking purpose)
        @return: a dictionary
        """
        if sent_idx is None or context_sents is None:
            return data_dict
        get_doc_name = 'doc_name' in data_dict

        labeled_sents_id = set()
        for concept in concepts:
            conclusions = Vectorizer.get_mapped_names(concept=concept, type_filter=type_filter)
            if len(conclusions) > 0:
                context_sents_ids = sent_idx.search(concept.start, concept.end)
                for id in context_sents_ids:
                    labeled_sents_id.add(id.data)
                    context = context_sents[id.data]
                    if concept.start >= context[0].start and concept.end <= context[-1].end:
                        for conclusion in conclusions:
                            data_dict['X'].append(' '.join([str(s) for s in context]))
                            data_dict['y'].append(conclusion)
                            data_dict['concept'].append(str(concept))
                            if get_doc_name:
                                data_dict['doc_name'].append(doc_name)
        # add unlabeled sentences as default label
        for i, context in enumerate(context_sents):
            if i not in labeled_sents_id:
                data_dict['X'].append(' '.join([str(s) for s in context]))
                data_dict['y'].append(default_label)
                data_dict['concept'].append('')
                if get_doc_name:
                    data_dict['doc_name'].append(doc_name)
        return data_dict

    @staticmethod
    def get_mapped_names(concept: Span, type_filter: Dict) -> List[str]:
        """

        @param concept: a concept Span
        @param type_filter: a dictionary (where attributes and values can be included), which
        maps a matched concept (string and its context string) to a new value in "y" column in the output
        dictionary. The structure of expected dictionary will be:
        concept_type->attr1->value1->...(other attr->value pairs if needed)->mapped key name
        @return: a list of matched (on concept type, attribute and value pairs) new concept names
        """
        type_name = concept.label_
        value = type_filter[type_name]
        keys = []
        Vectorizer.get_mapped_name_by_attr_values(concept=concept, type_filter=value, keys=keys)
        return keys

    @staticmethod
    def get_mapped_name_by_attr_values(concept: Span, type_filter: Union[Dict, str] = None, keys: List[str] = []):
        """

        @param concept: a concept Span
        @param type_filter: a dictionary (where attributes and values can be included), which
        maps a matched concept (string and its context string) to a new value in "y" column in the output
        dictionary. The structure of expected dictionary will be:
        concept_type->attr1->value1->...(other attr->value pairs if needed)->mapped key name
        @param keys: a list of matched (on concept type, attribute and value pairs) new concept names that passed on
        during recursion
        """
        if type_filter is None or len(type_filter) == 0:
            return
        elif not isinstance(type_filter, Dict):
            keys.append(type_filter)
            return
        for attr in type_filter:
            if not hasattr(concept._, attr):
                return keys
            value = getattr(concept._, attr)
            if value in type_filter[attr]:
                Vectorizer.get_mapped_name_by_attr_values(concept=concept, type_filter=type_filter[attr][value],
                                                          keys=keys)
        pass

    @staticmethod
    def docs_to_sents_data_dict(docs: List[Doc], sent_window: int = 1, type_filter: Union[Set[str], Dict] = set(),
                                default_label: str = "NEG", track_doc_name: bool = False):
        """
        Convert a list of SpaCy docs into a labeled data dictionary. Assuming the doc has been labeled based on concepts(snippets), Vectorizer
        extends the input to the concepts' context sentences (depends on the sent_window size), generate labeled context
        sentences data, and return a dictionary (with three keys: 'X'---the text of context sentences,'concepts'---
        the text of labeled concepts, 'y'---label)
        @param doc: a list of SpaCy Docs
        @param sent_window: The window size (in sentences) around the target concept that need to be pulled
        @param type_filter: Specify whether and what types of annotation will be used generate the output dictionary, this
        parameter can be defined as a set (only concept names are included) or a dictionary (where attributes and values
        can be included), which maps a matched concept (string and its context string) to a new value in "y"
        column in the output. The structure of expected dictionary will be:
        concept_type->attr1->value1->...(other attr->value pairs if needed)->mapped key name
        @param default_label: If there is no labeled concept in the context sentences, label it with this default_label
        @param track_doc_name: Whether add doc name to an additional column to track the output vectors
        @return: a dictionary
        """
        data_dict = {'X': [], 'concept': [], 'y': [], 'doc_name': []} if track_doc_name else {'X': [], 'concept': [],
                                                                                              'y': []}
        for doc in docs:
            Vectorizer.to_data_dict(doc, sent_window=sent_window, type_filter=type_filter,
                                    default_label=default_label, data_dict=data_dict)
        return data_dict

    @staticmethod
    def docs_to_sents_df(docs: List[Doc], sent_window: int = 1, type_filter: Union[Set[str], Dict] = set(),
                         default_label: str = "NEG", track_doc_name: bool = False) -> pd.DataFrame:
        """
        Convert a list of SpaCy docs into pandas DataFrame. Assuming the doc has been labeled based on concepts(snippets), Vectorizer
        extends the input to the concepts' context sentences (depends on the sent_window size), generate labeled context
        sentences data, and return a pandas DataFrame (with three columns: 'X'---the text of context sentences,'concepts'---
        the text of labeled concepts, 'y'---label)
        @param doc: a list of SpaCy Doc
        @param sent_window: The window size (in sentences) around the target concept that need to be pulled
        @param type_filter: Specify whether and what types of annotation will be used generate the output DataFrame, this
        parameter can be defined as a set (only concept names are included) or a dictionary (where attributes and values
        can be included), which maps a matched concept (string and its context string) to a new value in "y"
        column in the output. The structure of expected dictionary will be:
        concept_type->attr1->value1->...(other attr->value pairs if needed)->mapped key name
        @param default_label: If there is no labeled concept in the context sentences, label it with this default_label
        @param track_doc_name: Whether add doc name to an additional column to track the output vectors
        @return: a pandas DataFrame
        """
        data_dict = Vectorizer.docs_to_sents_data_dict(docs, sent_window=sent_window, type_filter=type_filter,
                                                       default_label=default_label, track_doc_name=track_doc_name)
        df = pd.DataFrame(data_dict)
        return df

    @staticmethod
    def docs_to_sents_nparray(docs: List[Doc], sent_window: int = 1, type_filter: Union[Set[str], Dict] = set(),
                              default_label: str = "NEG", track_doc_name: bool = False):
        """
        Convert a list of SpaCy docs into numpy array. Assuming the doc has been labeled based on concepts(snippets), Vectorizer
        extends the input to the concepts' context sentences (depends on the sent_window size), generate labeled context
        sentences data, and return a numpy array (with three columns: 'X'---the text of context sentences,'concepts'---
        the text of labeled concepts, 'y'---label)
        @param doc: a list of SpaCy Doc
        @param sent_window: The window size (in sentences) around the target concept that need to be pulled
        @param type_filter: Specify whether and what types of annotation will be used generate the output array, this
        parameter can be defined as a set (only concept names are included) or a dictionary (where attributes and values
        can be included), which maps a matched concept (string and its context string) to a new value in "y"
        column in the output. The structure of expected dictionary will be:
        concept_type->attr1->value1->...(other attr->value pairs if needed)->mapped key name
        @param default_label: If there is no labeled concept in the context sentences, label it with this default_label
        @param track_doc_name: Whether add doc name to an additional column to track the output vectors
        @return: a numpy array
        """
        data_dict = Vectorizer.docs_to_sents_data_dict(docs, sent_window=sent_window, type_filter=type_filter,
                                                       default_label=default_label,track_doc_name=track_doc_name)
        rows = []
        for i in range(0, len(data_dict['X'])):
            if track_doc_name:
                rows.append([data_dict['X'][i], data_dict['concept'][i], data_dict['y'][i], data_dict['doc_name'][i]])
            else:
                rows.append([data_dict['X'][i], data_dict['concept'][i], data_dict['y'][i]])
        sents_nparray = np.array(rows)
        return sents_nparray
