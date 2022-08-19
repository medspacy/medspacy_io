import logging
from collections import OrderedDict
from pathlib import Path
from typing import Union, Tuple, Set

from lxml import etree
from lxml.etree import Element, iterparse
from spacy.language import Language
from spacy.tokens.span import Span

from .base_reader import BaseDocReader, BaseDirReader


class EhostDocReader(BaseDocReader):
    """ This is a subclass of BaseDocReader to read in eHOST format files and generate SpaCy Docs
    """

    def __init__(self, nlp: Language = None, support_overlap: bool = False,
                 log_level: int = logging.WARNING, encoding: str = None, doc_name_depth: int = 0,
                 schema_file: Union[str, Path] = '', store_anno_string: bool = False,
                 use_adjudication: bool = False, **kwargs):
        """

        @param nlp: a SpaCy language model
        @param support_overlap: if the EhostDocReader need to support reading from overlapped annotations.
            Because SpaCy's Doc.ents does not allows overlapped Spans, to support overlapping, Spans need to be stored
            somewhere else----Doc._.concepts
        @param log_level: set the logger's logging level. TO debug, set to logging.DEBUG
        @param encoding: txt encoding
        @param doc_name_depth: depth of parent directories to add into doc_name
                default is 0: only use file name
                1: use 1 level parent directory name + file name
                -1: use full absolution path
                if you are dealing with multiple directories,this is helpful to
                locate the original files
        @param schema_file: initiate Span attributes using eHOST schema configuration file
        @param store_anno_string: whether read annotated string from annotations to double check parsed Span's correction
        @param use_adjudication: if read annotations from adjudication folder
        @param kwargs:other parameters
        """
        self.schema_set = False
        self.attr_names = self.set_attributes(schema_file=schema_file, encoding=encoding)
        if store_anno_string:
            if not Span.has_extension("span_txt"):
                Span.set_extension("span_txt", default="")
        super().__init__(nlp=nlp, support_overlap=support_overlap,
                         log_level=log_level, encoding=encoding, doc_name_depth=doc_name_depth,
                         schema_file=schema_file, store_anno_string=store_anno_string,
                         use_adjudication=use_adjudication, **kwargs)
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
        if schema_file is not None:
            root = etree.parse(str(schema_file.absolute()))
            for attr_def in root.iter("attributeDef"):
                name = attr_def[0].text.replace(' ', '_')
                default_value = attr_def[2].text
                if name not in attr_names and not Span.has_extension(name):
                    Span.set_extension(name, default=default_value)
                    attr_names.add(name)
            self.schema_set = True
        return attr_names

    def get_anno_content(self, txt_file: Path) -> str:
        """

        @param txt_file: a string or Path of a text file
        @return: a string of corresponding annotation file path string (because this function needs to be overwriten,
        the anno_content doesn't mean to retrieve the content in this class's function implementation)
        """
        anno_file = self.infer_anno_file_path(txt_file)
        self.check_file_validity(anno_file)
        return str(anno_file.absolute())

    def infer_anno_file_path(self, txt_file: Path) -> Path:
        """
        From the path of the text file, infer the corresponding annotation file. Need to be implemented for each subclass,
        as different annotation format use different conventions to store annotation files and txt files
        @param txt_file: the Path of a txt file
        @return: the Path of the corresponding annotation file
        """
        txt_file_name = txt_file.name
        anno_file_name = txt_file_name + '.knowtator.xml'
        if not self.use_adjudication:
            anno_file = Path(txt_file.parent.parent, 'saved', anno_file_name)
        else:
            anno_file = Path(txt_file.parent.parent, 'adjudication', anno_file_name)
        self.check_file_validity(anno_file)
        return anno_file

    def parse_to_dicts(self, xml_file: str, sort_spans: bool = False) -> Tuple[OrderedDict, dict, OrderedDict, OrderedDict]:
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
        iter = etree.iterparse(xml_file, events=('start',))
        # this doesn't seem elegant, but is said to be the fastest way
        spans = OrderedDict()
        classes = dict()
        attributes = OrderedDict()
        relations = OrderedDict()
        for event, ele in iter:
            if ele.tag == 'annotation':
                id, start, end, span_text = self.parse_annotation_tag(ele, iter)
                if self.store_anno_string:
                    spans[id] = (start, end, span_text)
                else:
                    spans[id] = (start, end)
            elif ele.tag == 'stringSlotMention':
                attr_id, attr, value = self.parse_attribute_tag(ele, iter)
                attributes[attr_id] = (attr, value)
            elif ele.tag == 'classMention':
                # <classMention id="EHOST_Instance_1">
                #   <hasSlotMention id="EHOST_Instance_8"/>
                #   <mentionClass id="Purulent">pain</mentionClass>
                # some annotations don't have "hasSlotMention" element
                # </classMention>

                # <classMention id="EHOST_Instance_61">
                #     <hasSlotMention id="EHOST_Instance_67" />
                #     <hasSlotMention id="EHOST_Instance_68" />
                #     <mentionClass id="Exclusions">presented</mentionClass>
                # </classMention>
                id = ele.get('id')
            elif ele.tag == 'mentionClass':
                class_tag = ele.get('id')
                if id not in classes:
                    classes[id] = ['', []]
                classes[id][0] = class_tag
            elif ele.tag == 'hasSlotMention':
                if id not in classes:
                    classes[id] = ['', []]
                classes[id][1].append(ele.get('id'))
            elif ele.tag == 'complexSlotMention':
                # <complexSlotMention id="EHOST_Instance_41">
                #     <mentionSlot id="Rel_A" />
                #     <complexSlotMentionValue value="EHOST_Instance_29" />
                # </complexSlotMention>
                rel_id, label, components = self.parse_relation_tag(ele, iter)
                if rel_id is not None:
                    relations[rel_id] = (label, components)
        if sort_spans:
            spans = sorted(spans.items(), key=lambda x: x[1][0])
        return spans, classes, attributes, relations

    # <annotation>
    #   <mention id="EHOST_Instance_1"/>
    #   <annotator id="eHOST_2010">sjl</annotator>
    #   <span start="232" end="236"/>
    #   <spannedText>pain</spannedText>
    #   <creationDate>Sun Apr 19 19:30:11 MDT 2020</creationDate>
    # </annotation>
    def parse_annotation_tag(self, ele: Element, iter: iterparse) -> Tuple:
        """

        @param ele:the current lxml element that start an entity defintition
        @param iter: lxml element iterator
        @return: a Tuple of (annotation_id, label, absolute start offset, absolute end offset, covered span text)
        """
        id = None
        start = -1
        end = -1
        span_text = None
        for i in range(0, 4):
            eve, child = iter.__next__()
            if child.tag == 'mention':
                id = child.get('id')
            elif child.tag == 'span':
                start = int(child.get('start'))
                end = int(child.get('end'))
            elif child.tag == 'spannedText':
                span_text = child.text
        return id, start, end, span_text

    # <stringSlotMention id="EHOST_Instance_8">
    #   <mentionSlot id="status"/>
    #   <stringSlotMentionValue value="present"/>
    # </stringSlotMention>
    def parse_attribute_tag(self, ele: Element, iter: iterparse) -> Tuple:
        """

        @param ele:the current lxml element that start an attribute defintition
        @param iter: lxml element iterator
        @return: a Tuple of (attribute_id, corresponding entity id, attribute_name, attribute_value)
        """
        id = ele.get('id')
        attr = ''
        value = ''
        for i in range(0, 2):
            eve, child = iter.__next__()
            if child.tag == 'mentionSlot':
                attr = child.get('id')
            else:
                value = child.get('value')
        return id, attr, value

    def parse_relation_tag(self, ele: Element, iter: iterparse) -> Tuple:
        """
        # TODO
        @param ele:the current lxml element that start a relationship defintition
        @param iter: lxml element iterator
        @return: a Tuple of (relation_id, (components ids contained in this relationship))
        """
        return (None, None, None)


class EhostDirReader(BaseDirReader):
    def __init__(self, txt_extension: str = 'txt', recursive: bool = False,
                 nlp: Language = None, **kwargs):
        """

        @param txt_extension: the text file extension name (default is 'txt').
        @param recursive: whether read file recursively down to the subdirectories.
        @param nlp: a SpaCy language model
        @param kwargs:other parameters to initiate EhostDocReader
        """
        super().__init__(txt_extension=txt_extension, recursive=recursive, nlp=nlp,
                         docReaderClass=EhostDocReader, **kwargs)
        pass
