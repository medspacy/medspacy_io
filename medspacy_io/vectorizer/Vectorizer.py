from typing import List, Set, Dict, Union, Tuple
from collections import OrderedDict
import numpy as np
import pandas as pd
from numpy.core.multiarray import ndarray
from quicksectx import IntervalTree
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from spacy.tokens.token import Token
import re


class Vectorizer:
    paragraph_delimiters = [r'\n\s+\s*', r'^[A-Z][a-z/&]*:$]', r'^[A-Z]{2,}$']

    # Compile the regex patterns into a single regex
    delimiter_regex = re.compile('|'.join(paragraph_delimiters), re.MULTILINE)

    @staticmethod
    def to_seq_df(doc: Doc, sent_window: int = 1, type_filter: Union[Set[str], Dict] = set(),
                  default_label: str = "O", sent_win_size: int = 1, max_tokens: int = 200,
                  pad_token: Union[str, None] = None, sep_token: Union[str, None] = '[SEP]',
                  track_doc_name: bool = False) -> pd.DataFrame:
        """
        Convert a SpaCy doc into pandas DataFrame using BIO sequence labels. Assuming the doc has been labeled based on
        concepts(snippets). Vectorizer extends the input to the same sentence (or the sentences around based on the sent_win_size,
        sentence window size), adding BI labels to the concepts.
        @param doc: a SpaCy Doc
        @param sent_window: The window size (in sentences) around the target concept that need to be pulled
        @param type_filter: Specify whether and what types of annotation will be used generate the output DataFrame, this
        parameter can be defined as a set (only concept names are included) or a dictionary (where attributes and values
        can be included), which maps a matched concept (string and its context string) to a new value in "y"
        column in the output. The structure of expected dictionary will be:
        concept_type->attr1->value1->...(other attr->value pairs if needed)->mapped key name
        @param default_label: The default label given to the tokens that do no have any annotations with given types (type_filter)
        @param sent_win_size: Number of sentences to be included to create the vector row around the annotation.
        @param max_tokens: Maximum tokens to limit the length of the vector
        @param pad_token: If None, the given sentence will be padded to the fixed length using given string. Otherwise, it
        will not be padded
        @param sep_token:  A special token separating two different sentences in the same input. If None, will not be inserted.
        @param track_doc_name:
        .. py:staticmethod:: to_data_seq_dict_on_types()
        @return:a pandas DataFrame
        """
        data_dict = {'X': [], 'tokens': [[]], 'labels': [[]], 'y': [], 'doc_name': []} \
            if track_doc_name else \
            {'X': [], 'tokens': [[]], 'labels': [[]], 'y': []}
        df = pd.DataFrame(data_dict)
        return df

    @staticmethod
    def to_sents_df(doc: Doc, sent_window: int = 1, type_filter: Union[Set[str], Dict] = set(),
                    default_label: str = "NEG", track_doc_name: bool = False, parag_context_top_n=0) -> pd.DataFrame:
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
        @param parag_context_top_n: use top n characters as paragraph context to generate paragcontext column.
        @return: a pandas DataFrame

        """
        data_dict = {'X': [], 'concept': [], 'y': []}
        if track_doc_name:
            data_dict['doc_name'] = []
        if parag_context_top_n > 0:
            data_dict['paragcontext'] = []
        data_dict = Vectorizer.to_data_dict(doc, sent_window=sent_window, type_filter=type_filter,
                                            default_label=default_label, data_dict=data_dict,
                                            parag_context_top_n=parag_context_top_n)
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
    def strip_whitespace(text, start, end):
        """Adjust the start and end indices to strip whitespace around the span."""
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end

    @staticmethod
    def add_default_paragraphs(doc: Doc):
        """Add paragraphs as spans to the spaCy Doc object, stripping leading and trailing whitespace."""
        spans = []
        start = 0

        for match in Vectorizer.delimiter_regex.finditer(doc.text):
            end = match.start()
            if start < end:
                # Adjust start and end indices to strip whitespace
                stripped_start, stripped_end = Vectorizer.strip_whitespace(doc.text, start, end)
                if stripped_start < stripped_end:  # Ensure there's still content left
                    span = doc.char_span(stripped_start, stripped_end, alignment_mode='expand')
                    if span is not None:
                        spans.append(span)
            start = match.start()

        if start < len(doc.text):
            # Adjust final span to strip whitespace
            stripped_start, stripped_end = Vectorizer.strip_whitespace(doc.text, start, len(doc.text))
            if stripped_start < stripped_end:  # Ensure there's still content left
                span = doc.char_span(stripped_start, stripped_end, alignment_mode='expand')
                if span is not None:
                    spans.append(span)

        # Set paragraph spans in the doc
        doc.spans['paragraphs'] = spans

        return doc

    @staticmethod
    def index_paragraphs(paragraphs: List[Span]) -> IntervalTree:
        parag_idx = IntervalTree()
        context_sents = []
        for i, paragraph in enumerate(paragraphs):
            parag_idx.add(paragraph.start, paragraph.end, i)
        return parag_idx

    @staticmethod
    def get_paragcontext(concept: Span, parag_context_top_n: int, parag_idx: IntervalTree) -> str:
        parag_ids = parag_idx.search(concept.start, concept.end)
        parag_id = min([d.data for d in parag_ids])
        parag = concept.doc.spans['paragraphs'][parag_id]
        parag_context = parag.text[:parag_context_top_n]
        return parag_context

    @staticmethod
    def index_sentence_windows(sents: List[Span], sent_window: int) -> Tuple[IntervalTree, List[List[Span]]]:
        context_sents = []
        sent_idx = IntervalTree()
        for i in range(0, len(sents) - sent_window + 1):
            begin_sent = sents[i]
            end_sent = sents[i + sent_window - 1]
            sent_idx.add(begin_sent.start, end_sent.end, len(context_sents))
            context_sents.append(sents[i:i + sent_window])
        return sent_idx, context_sents

    @staticmethod
    def format_concepts(doc: Doc, type_filter: Union[Set[str], Dict] = set(), has_paragraph_spans: bool = False) -> \
    List[Span]:
        concepts = []
        if (not has_paragraph_spans and len(doc.spans) > 0) or (has_paragraph_spans and len(doc.spans) > 1):
            for span_type in doc.spans.keys():
                if span_type == 'paragraphs':
                    continue
                if len(type_filter) == 0 or span_type in type_filter:
                    concepts.extend(doc.spans[span_type])
        else:
            concepts = [ent for ent in doc.ents if (len(type_filter) == 0 or ent.label in type_filter)]
        return concepts

    @staticmethod
    def to_data_dict(doc: Doc, sent_window: int = 1, type_filter: Union[Set[str], Dict] = set(),
                     default_label: str = "NEG", data_dict: dict = {'X': [], 'concept': [], 'y': []},
                     parag_context_top_n: int = 0) -> Dict:
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
        @param parag_context_top_n: use top n characters as paragraph context to generate paragcontext column.
        @return: a dictionary
        """
        sents = list(doc.sents)
        sent_idx, context_sents = Vectorizer.index_sentence_windows(sents, sent_window)
        concepts = Vectorizer.format_concepts(doc, type_filter, parag_context_top_n > 0)

        get_doc_name = 'doc_name' in data_dict
        doc_name = doc._.doc_name if get_doc_name else ''

        if isinstance(type_filter, Set):
            data_dict = Vectorizer.to_data_dict_on_types(concepts=concepts,
                                                         type_filter=type_filter,
                                                         default_label=default_label,
                                                         data_dict=data_dict,
                                                         sent_idx=sent_idx, context_sents=context_sents,
                                                         doc_name=doc_name, parag_context_top_n=parag_context_top_n)
        elif isinstance(type_filter, Dict):
            if len(type_filter) == 0:
                data_dict = Vectorizer.to_data_dict_on_types(concepts=concepts,
                                                             default_label=default_label,
                                                             data_dict=data_dict,
                                                             sent_idx=sent_idx, context_sents=context_sents,
                                                             doc_name=doc_name, parag_context_top_n=parag_context_top_n)
            else:
                data_dict = Vectorizer.to_data_dict_on_type_attr_values(concepts=concepts,
                                                                        type_filter=type_filter,
                                                                        default_label=default_label,
                                                                        data_dict=data_dict,
                                                                        sent_idx=sent_idx, context_sents=context_sents,
                                                                        doc_name=doc_name,
                                                                        parag_context_top_n=parag_context_top_n)
        else:
            raise TypeError(
                'The arg: "type_filter" needs to be either a set of concept names or a dictionary. Not a {}:\n\t{}'.format(
                    type(type_filter), str(type_filter)))
        return data_dict

    @staticmethod
    def to_seq_data_dict(doc: Doc, sent_window: int = 1, type_filter: Union[List[str], OrderedDict] = [],
                         output_labels: OrderedDict = OrderedDict(),
                         default_label: str = "O",
                         data_dict: OrderedDict = OrderedDict(),
                         max_tokens: int = 200,
                         pad_token: Union[str, None] = None,
                         sep_token: Union[str, None] = '[SEP]') -> Dict:
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
        @param output_labels: the ordered list of output label
        @param default_label: If there is no labeled concept in the context sentences, label it with this default_label
        @param data_dict: a dictionary to hold the output and pass on across documents, so that a corpus can be aggregated
        @param sent_idx: an IntervalTree built with all sentences in the doc
        @param context_sents: a 2-d list of sentences with predefined window size.
        @return: a dictionary
        """
        if len(type_filter) == 0:
            raise ValueError(
                'type_filter must be defined in Sequence Vectorizing, otherwise the output labels cannot be ordered consistantly.')
        sent_idx = IntervalTree()
        sents = list(doc.sents)
        context_sents = []
        for i in range(0, len(sents) - sent_window + 1):
            begin_sent = sents[i]
            end_sent = sents[i + sent_window - 1]
            sent_idx.add(begin_sent.start, end_sent.end, len(context_sents))
            context_sents.append(sents[i:i + sent_window])

        filter_type = False
        if isinstance(type_filter, list):
            filter_type = True
            type_filter = OrderedDict([(t, t) for i, t in enumerate(type_filter)])

        if len(output_labels) == 0:
            Vectorizer.get_output_labels(type_filter, output_labels)

        if len(data_dict) == 0:
            data_dict: dict = OrderedDict([(col, []) for col in (['X', 'tokens'] + list(output_labels.keys()) + ['y'])])

        concepts = []
        if len(doc.spans) > 0:
            for type in doc.spans.keys():
                if type in type_filter:
                    concepts.extend(doc.spans[type])
        else:
            concepts = [ent for ent in doc.ents if (ent.label in type_filter)]

        get_doc_name = 'doc_name' in data_dict
        doc_name = doc._.doc_name if get_doc_name else ''

        if filter_type:
            data_dict = Vectorizer.to_seq_data_dict_on_types(concepts=concepts,
                                                             type_filter=type_filter,
                                                             output_labels=output_labels,
                                                             default_label=default_label,
                                                             data_dict=data_dict,
                                                             sent_idx=sent_idx, context_sents=context_sents,
                                                             doc_name=doc_name, sep_token=sep_token)
        else:
            data_dict = Vectorizer.to_seq_data_dict_on_type_attr_values(concepts=concepts,
                                                                        type_filter=type_filter,
                                                                        output_labels=output_labels,
                                                                        default_label=default_label,
                                                                        data_dict=data_dict,
                                                                        sent_idx=sent_idx,
                                                                        context_sents=context_sents,
                                                                        doc_name=doc_name, sep_token=sep_token)

        return data_dict

    @staticmethod
    def get_output_labels(type_filter: Dict, output_labels: OrderedDict):
        for key, value in type_filter.items():
            if isinstance(value, str):
                output_labels[value] = len(output_labels)
            elif isinstance(value, Dict):
                Vectorizer.get_output_labels(value, output_labels)
            else:
                print('Unsupported value type in type_filter: {}'.format(type(value)))

    @staticmethod
    def to_seq_data_dict_on_types(concepts: List[Span], type_filter: OrderedDict = OrderedDict(),
                                  output_labels: OrderedDict = OrderedDict(),
                                  default_label: str = "O",
                                  data_dict: dict = {'X': [], 'tokens': [], 'labels': [], 'y': []},
                                  sent_idx: IntervalTree = None, context_sents: List[List[Span]] = [],
                                  doc_name: str = '', sep_token: Union[str, None] = '[SEP]') -> Dict:
        """
        Convert a SpaCy doc into a labeled data dictionary. Assuming the doc has been labeled based on concepts(snippets), Vectorizer
        extends the input to the concepts' context sentences (depends on the sent_window size), generate labeled context
        sentences data, and return a dictionary (with three keys: 'X'---the text of context sentences,'concepts'---
        the text of labeled concepts, 'y'---label)
        @param concepts: a list of concepts (in Span type)
        @param type_filter: a list of type names that need to be included to be vectorized
        @param default_label: The default label given to the tokens that do no have any annotations with given types (type_filter)
        @param max_tokens: Maximum tokens to limit the length of the vector
        @param pad_token: If None, the given sentence will be padded to the fixed length using given string. Otherwise, it
        will not be padded
        @param sep_token:  A special token separating two different sentences in the same input. If None, will not be inserted.
        @param data_dict: a dictionary to hold the output and pass on across documents, so that a corpus can be aggregated
        @param sent_idx: an IntervalTree built with all sentences in the doc
        @param context_sents: a 2-d list of sentences with predefined window size.
        @param doc_name: doc file name (for tracking purpose)
        @return: a dictionary
        """
        if sent_idx is None or len(context_sents) == 0:
            return data_dict
        get_doc_name = 'doc_name' in data_dict
        all_concept_idx = IntervalTree()
        for id, concept in enumerate(concepts):
            if concept.label_ not in type_filter:
                continue
            all_concept_idx.add(concept.start, concept.end, id)

        for i, context in enumerate(context_sents):
            if sep_token is None:
                sepped_context = [t for s in context for t in list(s)]
            else:
                sepped_context = [t for s in context for t in list(s) + [sep_token]][:-1]
            data_dict['X'].append(' '.join([str(s) for s in sepped_context]))
            data_dict['tokens'].append([str(t) for t in sepped_context])
            if sep_token is None:
                for label in output_labels.keys():
                    data_dict[label].append([default_label] * len(sepped_context))
            else:
                for label in output_labels.keys():
                    data_dict[label].append(
                        [default_label if t != sep_token else sep_token for t in data_dict['tokens'][-1]])

            overlapped_concepts = all_concept_idx.search(context[0].start, context[-1].end)

            if len(overlapped_concepts) > 0:
                sent_idx = IntervalTree()
                for j, s in enumerate(context):
                    sent_idx.add(s.start, s.end, j)

                for con_id in overlapped_concepts:
                    concept = concepts[con_id.data]
                    label = concept.label_
                    output_label = type_filter[label] if label in type_filter else default_label
                    sent_id = sent_idx.search(concept.start, concept.end)
                    if sent_id is None or len(sent_id) == 0:
                        print("Error: no overlapped sentence is found for concept: {}".format(concept))
                        continue
                    sent_id = sent_id[0].data
                    rel_start = concept.start - context[0].start
                    rel_end = concept.end - context[0].start

                    if sep_token is not None:
                        # because we inserted the sep_tokens, we need to add the offset as well
                        rel_start += sent_id
                        rel_end += sent_id

                    if rel_end > len(sepped_context):
                        rel_end = len(sepped_context)
                    # to_edit = data_dict['output_label'][-1]
                    # to_edit[rel_start:rel_end] = [output_label] * (
                    #         rel_end - rel_start)
                    data_dict[output_label][-1][rel_start:rel_end] = [output_label] * (
                            rel_end - rel_start)
            if get_doc_name:
                data_dict['doc_name'].append(doc_name)
        return data_dict

    @staticmethod
    def to_seq_data_dict_on_type_attr_values(concepts: List[Span], type_filter: OrderedDict = OrderedDict(),
                                             output_labels: OrderedDict = OrderedDict(),
                                             default_label: str = "O",
                                             data_dict: dict = {'X': [], 'tokens': [], 'labels': [], 'y': []},
                                             sent_idx: IntervalTree = None, context_sents: List[List[Span]] = [],
                                             doc_name: str = '', sep_token: Union[str, None] = '[SEP]') -> Dict:
        """
        Convert a SpaCy doc into a labeled data dictionary. Assuming the doc has been labeled based on concepts(snippets), Vectorizer
        extends the input to the concepts' context sentences (depends on the sent_window size), generate labeled context
        sentences data, and return a dictionary (with three keys: 'X'---the text of context sentences,'concepts'---
        the text of labeled concepts, 'y'---label)
        @param concepts: a list of concepts (in Span type)
        @param type_filter: a list of type names that need to be included to be vectorized
        @param default_label: The default label given to the tokens that do no have any annotations with given types (type_filter)
        @param max_tokens: Maximum tokens to limit the length of the vector
        @param pad_token: If None, the given sentence will be padded to the fixed length using given string. Otherwise, it
        will not be padded
        @param sep_token:  A special token separating two different sentences in the same input. If None, will not be inserted.
        @param data_dict: a dictionary to hold the output and pass on across documents, so that a corpus can be aggregated
        @param sent_idx: an IntervalTree built with all sentences in the doc
        @param context_sents: a 2-d list of sentences with predefined window size.
        @param doc_name: doc file name (for tracking purpose)
        @return: a dictionary
        """

        if sent_idx is None or len(context_sents) == 0:
            return data_dict
        get_doc_name = 'doc_name' in data_dict
        all_concept_idx = IntervalTree()
        for id, concept in enumerate(concepts):
            if concept.label_ not in type_filter:
                continue
            all_concept_idx.add(concept.start, concept.end, id)

        for i, context in enumerate(context_sents):
            if sep_token is None:
                sepped_context = [t for s in context for t in list(s)]
            else:
                sepped_context = [t for s in context for t in list(s) + [sep_token]][:-1]
            data_dict['X'].append(' '.join([str(s) for s in sepped_context]))
            data_dict['tokens'].append([str(t) for t in sepped_context])
            if sep_token is None:
                for label in output_labels.keys():
                    data_dict[label].append([default_label] * len(sepped_context))
            else:
                for label in output_labels.keys():
                    data_dict[label].append(
                        [default_label if t != sep_token else sep_token for t in data_dict['tokens'][-1]])

            overlapped_concepts = all_concept_idx.search(context[0].start, context[-1].end)

            if len(overlapped_concepts) > 0:
                sent_idx = IntervalTree()
                for j, s in enumerate(context):
                    sent_idx.add(s.start, s.end, j)

                for con_id in overlapped_concepts:
                    concept = concepts[con_id.data]
                    label = concept.label_
                    sent_id = sent_idx.search(concept.start, concept.end)
                    if sent_id is None or len(sent_id) == 0:
                        print("Error: no overlapped sentence is found for concept: {}".format(concept))
                        continue
                    sent_id = sent_id[0].data
                    rel_start = concept.start - context[0].start
                    rel_end = concept.end - context[0].start

                    if sep_token is not None:
                        # because we inserted the sep_tokens, we need to add the offset as well
                        rel_start += sent_id
                        rel_end += sent_id

                    if rel_end > len(sepped_context):
                        rel_end = len(sepped_context)
                    mapped_labels = Vectorizer.get_mapped_names(concept, type_filter)

                    for output_label in mapped_labels:
                        data_dict[output_label][-1][rel_start:rel_end] = [output_label] * (
                                rel_end - rel_start)
            if get_doc_name:
                data_dict['doc_name'].append(doc_name)

        # for i, context in enumerate(context_sents):
        #     data_dict['X'].append(' '.join([str(s) for s in context]))
        #     data_dict['tokens'].append([str(t) for t in context])
        #     if len(all_concept_idx.search(context[0].idx, context[-1].idx + context[-1].__len__())) == 0:
        #         data_dict['labels'].append([[default_label] * len(context) for l in type_filter])
        #     else:
        #         data_dict['labels'].append(
        #             [[Vectorizer.infer_token_label(l, concept_indexes[l].search(t.idx, t.idx + t.__len__()), concepts,
        #                                            type_filter,
        #                                            default_label) for t in
        #               context] for l in type_filter])
        #     if get_doc_name:
        #         data_dict['doc_name'].append(doc_name)
        return data_dict

    @staticmethod
    def to_data_dict_on_types(concepts: List[Span], type_filter: Set = set(),
                              default_label: str = "NEG", data_dict: dict = {'X': [], 'concept': [], 'y': []},
                              sent_idx: IntervalTree = None, context_sents: List[List[Span]] = None,
                              doc_name: str = '', parag_context_top_n: int = 0) -> Dict:
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
        @param parag_context_top_n: use top n characters as paragraph context to generate paragcontext column.
        @return: a dictionary
        """
        if sent_idx is None or context_sents is None:
            return data_dict

        if parag_context_top_n > 0 and len(context_sents) > 0:
            doc = context_sents[0][0].doc
            if 'paragraphs' not in doc.spans:
                Vectorizer.add_default_paragraphs(doc)
            parag_idx = Vectorizer.index_paragraphs(doc.spans['paragraphs'])

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
                    if parag_context_top_n > 0:
                        data_dict['paragcontext'].append(
                            Vectorizer.get_paragcontext(concept, parag_context_top_n, parag_idx))
                    if get_doc_name:
                        data_dict['doc_name'].append(doc_name)
        for i, context in enumerate(context_sents):
            if i not in labeled_sents_id:
                data_dict['X'].append(' '.join([str(s) for s in context]))
                data_dict['y'].append(default_label)
                data_dict['concept'].append('')
                if parag_context_top_n > 0:
                    data_dict['paragcontext'].append(
                        Vectorizer.get_paragcontext(context[0], parag_context_top_n, parag_idx))
                if get_doc_name:
                    data_dict['doc_name'].append(doc_name)
        return data_dict

    @staticmethod
    def to_data_dict_on_type_attr_values(concepts: List[Span], type_filter: Dict = dict(),
                                         default_label: str = "NEG",
                                         data_dict: dict = {'X': [], 'concept': [], 'y': []},
                                         sent_idx: IntervalTree = None, context_sents: List[List[Span]] = None,
                                         doc_name: str = '', parag_context_top_n: int = 0) -> Dict:
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
        @param parag_context_top_n: use top n characters as paragraph context to generate paragcontext column.
        @return: a dictionary
        """
        if sent_idx is None or context_sents is None:
            return data_dict
        get_doc_name = 'doc_name' in data_dict

        if parag_context_top_n > 0 and len(context_sents) > 0:
            doc = context_sents[0][0].doc
            if 'paragraphs' not in doc.spans:
                Vectorizer.add_default_paragraphs(doc)
            parag_idx = Vectorizer.index_paragraphs(doc.spans['paragraphs'])

        labeled_sents_id = set()
        for concept in concepts:
            conclusions = Vectorizer.get_mapped_names(concept=concept, type_filter=type_filter)
            # y is using conclusions label, so that we can stratefied sampling.
            # while the actual sequential labels are stored in "labels"
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
                            if parag_context_top_n > 0:
                                data_dict['paragcontext'].append(
                                    Vectorizer.get_paragcontext(concept, parag_context_top_n, parag_idx))
                            if get_doc_name:
                                data_dict['doc_name'].append(doc_name)
        # add unlabeled sentences as default label
        for i, context in enumerate(context_sents):
            if i not in labeled_sents_id:
                data_dict['X'].append(' '.join([str(s) for s in context]))
                data_dict['y'].append(default_label)
                data_dict['concept'].append('')
                if parag_context_top_n > 0:
                    data_dict['paragcontext'].append(
                        Vectorizer.get_paragcontext(context[0], parag_context_top_n, parag_idx))
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
            if hasattr(concept._, attr):
                value = getattr(concept._, attr)
            elif hasattr(concept._, 'ANNOT_'+attr):
                value = getattr(concept._, 'ANNOT_'+attr)
            else:
                continue
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
                                                       default_label=default_label, track_doc_name=track_doc_name)
        rows = []
        for i in range(0, len(data_dict['X'])):
            if track_doc_name:
                rows.append([data_dict['X'][i], data_dict['concept'][i], data_dict['y'][i], data_dict['doc_name'][i]])
            else:
                rows.append([data_dict['X'][i], data_dict['concept'][i], data_dict['y'][i]])
        sents_nparray = np.array(rows)
        return sents_nparray

    @staticmethod
    def docs_to_seq_data_dict(docs: List[Doc], sent_window: int = 1,
                              type_filter: Union[List[str], OrderedDict] = [],
                              default_label: str = "O",
                              max_tokens: int = 200,
                              pad_token: Union[str, None] = None,
                              sep_token: Union[str, None] = '[SEP]',
                              track_doc_name: bool = False):
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
        data_dict = {'X': [], 'tokens': [], 'labels': [], 'y': [], 'doc_name': []} if track_doc_name else {'X': [],
                                                                                                           'tokens': [],
                                                                                                           'labels': [],
                                                                                                           'y': []}
        output_labels = OrderedDict()
        if len(output_labels) == 0:
            Vectorizer.get_output_labels(type_filter, output_labels)

        data_dict: dict = OrderedDict([(col, []) for col in ['X', 'tokens'] + list(output_labels.keys()) + ['y']])

        if track_doc_name:
            data_dict['doc_name'] = []

        for doc in docs:
            Vectorizer.to_seq_data_dict(doc, sent_window=sent_window, type_filter=type_filter,
                                        output_labels=output_labels, default_label=default_label,
                                        data_dict=data_dict, max_tokens=max_tokens,
                                        pad_token=pad_token, sep_token=sep_token)
        return data_dict

    @staticmethod
    def docs_to_seq_df(docs: List[Doc], sent_window: int = 1, type_filter: Union[List[str], OrderedDict] = [],
                       output_labels: OrderedDict = OrderedDict(),
                       default_label: str = "O",
                       data_dict: dict = {'X': [], 'tokens': [], 'labels': [], 'y': []},
                       max_tokens: int = 200,
                       pad_token: Union[str, None] = None,
                       sep_token: Union[str, None] = '[SEP]', track_doc_name: bool = False) -> pd.DataFrame:
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
        data_dict = Vectorizer.docs_to_seq_data_dict(docs, sent_window=sent_window, type_filter=type_filter,
                                                     output_labels=output_labels, default_label=default_label,
                                                     data_dict=data_dict, max_tokens=max_tokens,
                                                     pad_token=pad_token, sep_token=sep_token)
        df = pd.DataFrame(data_dict)
        return df

    @staticmethod
    def docs_to_seq_nparray(docs: List[Doc], sent_window: int = 1, type_filter: Union[List[str], OrderedDict] = [],
                            output_labels: OrderedDict = OrderedDict(),
                            default_label: str = "O",
                            data_dict: dict = {'X': [], 'tokens': [], 'labels': [], 'y': []},
                            max_tokens: int = 200,
                            pad_token: Union[str, None] = None,
                            sep_token: Union[str, None] = '[SEP]', track_doc_name: bool = False):
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
        data_dict = Vectorizer.docs_to_seq_data_dict(docs, sent_window=sent_window, type_filter=type_filter,
                                                     output_labels=output_labels, default_label=default_label,
                                                     data_dict=data_dict, max_tokens=max_tokens,
                                                     pad_token=pad_token, sep_token=sep_token)

        rows = []
        keys = list(data_dict.keys())
        for i in range(0, len(data_dict['X'])):
            rows.append([data_dict[k][i] for k in keys])
        sents_nparray = np.array(rows)
        return sents_nparray
