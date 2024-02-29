import logging
from collections import OrderedDict, _OrderedDictItemsView
from pathlib import Path
from typing import Union, Tuple, Set
import re
from spacy.language import Language
from spacy.tokens.span import Span

from .base_reader import BaseDocReader, BaseDirReader


class BratDocReader(BaseDocReader):
    """ This is a subclass of BaseDocReader to read in eHOST format files and generate SpaCy Docs
    """

    def __init__(self, nlp: Language = None, support_overlap: bool = False,
                 log_level: int = logging.WARNING, encoding: str = None, doc_name_depth: int = 0,
                 schema_file: Union[str, Path] = '', store_anno_string: bool = False,
                 **kwargs):
        """

        @param nlp: Spacy Language model
        @param support_overlap: whether need to support overlapped annotations
        @param log_level: logging level configuration
        @param encoding: txt encoding
        @param doc_name_depth: depth of parent directories to add into doc_name
                default is 0: only use file name
                1: use 1 level parent directory name + file name
                -1: use full absolution path
                if you are dealing with multiple directories,this is helpful to
                locate the original files
        @param schema_file: initiate Span attributes using eHOST schema configuration file
        @param store_anno_string: whether read annotated string from annotations to double check parsed Span's correction
        @param kwargs:other parameters
        """
        self.schema_set = False
        self.attr_names = self.set_attributes(schema_file=schema_file, encoding=encoding)
        if store_anno_string:
            if not Span.has_extension("span_txt"):
                Span.set_extension("span_txt", default="")
        super().__init__(nlp=nlp, support_overlap=support_overlap,
                         log_level=log_level, encoding=encoding, doc_name_depth=doc_name_depth,
                         schema_file=schema_file, store_anno_string=store_anno_string, **kwargs)
        pass

    def set_attributes(self, schema_file: Union[str, Path] = '', encoding: str = None) -> Set:
        """


        The current version SpaCy doesn't differentiate attributes for different annotation types.
        Thus, any attributes extended here will be applied to all Spans.
        @param schema_file: initiate Span attributes using eHOST schema configuration file
        @param encoding: text encoding
        @return: a set of attribute names
        """
        schema_file = self.check_file_validity(schema_file, False)
        attr_names = set()
        attr_conf_start = False
        if schema_file is not None and schema_file.name.endswith("conf"):
            print('found annotation.conf file')
            for row in schema_file.read_text(encoding=encoding).split("\n"):
                if len(row.strip()) == 0 or row[0] == '#':
                    continue
                if row.startswith(r'[attributes]'):
                    attr_conf_start = True
                    continue
                elif row[0] == '[':
                    attr_conf_start = False
                if attr_conf_start:
                    # [attributes]
                    # Negation        Arg:<EVENT>
                    # Confidence        Arg:<EVENT>, Value:Possible|Likely|Certain
                    items=re.split('\s+', row)
                    name = items[0]
                    default_value = None
                    values=items[-1]
                    if values.startswith('Value'):
                      default_value=values.split(":")
                      if len(default_value>1):
                        default_value=default_value[1]
                    if name not in attr_names and not Span.has_extension(name):
                        Span.set_extension(name, default=default_value)
                        attr_names.add(name)
            self.schema_set = True
        return attr_names

    def infer_anno_file_path(self, txt_file: Path) -> Path:
        """
        From the path of the text file, infer the corresponding annotation file. Need to be implemented for each subclass,
        as different annotation format use different conventions to store annotation files and txt files
        @param txt_file: the Path of a txt file
        @return: the Path of the corresponding annotation file
        """
        txt_file_name = txt_file.name
        anno_file_name = txt_file_name[:-4] + '.ann'
        anno_file = Path(txt_file.parent, anno_file_name)
        self.check_file_validity(anno_file)
        return anno_file

    def parse_to_dicts(self, anno_str: str, sort_spans: bool = False) -> Tuple[
        _OrderedDictItemsView, OrderedDict, OrderedDict, OrderedDict]:
        """
        Parse annotations into a Tuple of OrderedDicts, must be implemented in subclasses
        @param anno: The annotation string (can be a file path or file content, depends on how get_anno_content is implemented)
        @param sort_spans: whether sort the parsed spans
        @return: A Tuple of following items:
             sorted_spans: a sorted OrderedDict Items ( spans[entity_id] = (start, end, span_text))
             classes: a OrderedDict to map a entity id to [entity label, [attr_ids]]
             attributes: a OrderedDict to map a attribute id to (attribute_name, attribute_value)
             relations: a OrderedDict to map a relation_id to (label, (relation_component_ids))
        """
        spans = OrderedDict()
        attributes = OrderedDict()
        classes = OrderedDict()
        relations = OrderedDict()
        for row in anno_str.split('\n'):
            if len(row.strip()) == 0 or row[0] == '#':
                continue
            if row[0] == 'T':
                entity_id, label, start, end, span_text = self.parse_annotation_tag(row)
                if self.store_anno_string:
                    spans[entity_id] = (start, end, span_text)
                else:
                    spans[entity_id] = (start, end)
                if entity_id not in classes:
                    classes[entity_id] = ['', []]
                classes[entity_id][0] = label
            elif row[0] == 'A':
                attr_id, entity_id, attr, value = self.parse_attribute_tag(row)
                attributes[attr_id] = (attr, value)
                if entity_id not in classes:
                    classes[entity_id] = ['', []]
                classes[entity_id][1].append(attr_id)
            elif row[0] == 'R' or row[0] == 'E':
                rel_id, label, components = self.parse_relation_tag(row)
                if rel_id is not None:
                    relations[rel_id] = (label, components)
        if sort_spans:
            spans = sorted(spans.items(), key=lambda x: x[1][0])
        else:
            spans = spans.items()
        return spans, classes, attributes, relations

    # T1	Gene_expression 447 457	expression
    # T9	Protein 546 557;565 570	complicated panic

    def parse_annotation_tag(self, row: str) -> Tuple:
        """

        @param row: a string of annotation content row that define an entity
        @return: a Tuple of (annotation_id, label, absolute start offset, absolute end offset, covered span text)
        """
        elements = row.split('\t')
        if len(elements) < 3:
            self.logger.warning("Entity annotation format error: " + row)
        id = elements[0]
        span_text = elements[2]
        anno = elements[1].split(' ')
        if len(anno) < 3:
            self.logger.warning("Entity annotation format error: " + row)
        label = anno[0]
        start = int(anno[1])
        end = int(anno[-1])
        return id, label, start, end, span_text

    # A1	Confidence E2 Possible
    # A6	Negation E5
    def parse_attribute_tag(self, row: str) -> Tuple:
        """

        @param row: a string of annotation content row that define an attribute
        @return: a Tuple of (attribute_id, corresponding entity id, attribute_name, attribute_value)
        """
        elements = row.split('\t')
        if len(elements) < 2:
            self.logger.warning("Attribute annotation format error: " + row)
        attr_id = elements[0]
        anno = elements[1].split(' ')
        if len(anno) < 2:
            self.logger.warning("Attribute annotation format error: " + row)
        entity_id = anno[1]
        attr = anno[0]
        if len(anno) == 2:
            value = True
        else:
            value = anno[2]
        return attr_id, entity_id, attr, value

    # R1	Part-of Arg1:T9 Arg2:T11
    # E1	Gene_expression:T1
    # E1	Gene_expression:T1 Theme:T2
    # E2	Negative_regulation:T3 Cause:E1 Theme:E3
    def parse_relation_tag(self, row: str) -> Tuple:
        """

        @param row:a string of annotation content row that define a relationship
        @return: a Tuple of (relation_id, (components ids contained in this relationship))
        """
        elements = row.strip().split('\t')
        rel_id = elements[0]
        anno = elements[1].split(' ')
        label = None
        if row[0] == 'R':
            label = anno[0]
            components = (anno[1].split(':')[1], anno[2].split(':')[1])
        elif row[0] == 'E':
            if len(anno) == 1:
                return (None, None, None)
            else:
                l = []
                c = []
                for comp in anno:
                    atom = comp.split(':')
                    l.append(atom[0])
                    c.append(atom[1])
                l.sort()
                label = '+'.join(l)
                components = tuple(c)
        return rel_id, label, components


class BratDirReader(BaseDirReader):
    def __init__(self, txt_extension: str = 'txt', recursive: bool = False,
                 nlp: Language = None, **kwargs):
        """

        @param txt_extension: the text file extension name (default is 'txt').
        @param recursive: whether read file recursively down to the subdirectories.
        @param nlp: a SpaCy language model
        @param kwargs:other parameters to initiate BratDocReader
        """
        super().__init__(txt_extension=txt_extension, recursive=recursive, nlp=nlp,
                         docReaderClass=BratDocReader, **kwargs)
        pass
