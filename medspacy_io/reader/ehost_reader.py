from collections import OrderedDict
from pathlib import Path
from typing import Union, Tuple, Type

from lxml import etree
from lxml.etree import Element, iterparse
from spacy.language import Language
from spacy.tokens.span import Span

from .base_reader import BaseDocReader, BaseDirReader


class EhostDocReader(BaseDocReader):
    """ This is a subclass of BaseDocReader to read in eHOST format files and generate SpaCy Docs
    """

    def __init__(self, nlp: Language = None, support_overlap: bool = False, use_adjudication: bool = False,
                 schema_file: Union[str, Path] = ''):
        """

        :param nlp: a SpaCy language model
        :param use_adjudication: if read annotations from adjudication folder
        :param schema_file: initiate Span attributes using eHOST schema configuration file
        """
        self.schema_set = False
        self.set_attributes(schema_file=schema_file)
        super().__init__(nlp=nlp, support_overlap=support_overlap, use_adjudication=use_adjudication)
        pass

    def set_attributes(self, schema_file: Union[str, Path] = ''):
        """
        The current version SpaCy doesn't differentiate attributes for different annotation types.
        Thus, any attributes extended here will be applied to all Spans.
        :arg schema_file: initiate Span attributes using eHOST schema configuration file

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
        pass

    def get_anno_content(self, txt_file: Path) -> str:
        """
        :arg txt_file from the text file path, infer knowtator.xml file path
        To use lxml's iterparse, just keep the path string here instead of reading the content
        """
        anno_file = self.infer_anno_file_path(txt_file)
        self.check_file_validity(anno_file)
        return str(anno_file.absolute())

    def infer_anno_file_path(self, txt_file: Path) -> Path:
        txt_file_name = txt_file.name
        anno_file_name = txt_file_name + '.knowtator.xml'
        if not self.use_adjudication:
            anno_file = Path(txt_file.parent.parent, 'saved', anno_file_name)
        else:
            anno_file = Path(txt_file.parent.parent, 'adjudication', anno_file_name)
        self.check_file_validity(anno_file)
        return anno_file

    def process_without_overlaps(self, doc):
        """
        Take in a SpaCy doc, add eHOST annotation(Span)s to doc.ents.
        This function doesn't support overlapped annotations.
         :arg doc: a SpaCy doc
        """
        sorted_span, classes, attributes = self.parse_to_dicts(self.anno, sort_spans=True)
        existing_entities = list(doc.ents)
        new_entities = list()
        previous_token_offset = 0
        total = len(doc)
        for id, (start, end) in sorted_span:
            # because SpaCy uses token offset instead of char offset to define Spans, we need to match them,
            # binary search is used here to speed up
            token_start = self.find_start_token(start, previous_token_offset, total, doc)
            token_end = self.find_end_token(end, token_start, total, doc)
            if token_start < 0 or token_start >= total or token_end < 0 or token_end > total:
                raise ValueError(
                    "It is likely your annotations overlapped, which process_without_overlaps doesn't support parsing "
                    "those. You will need to initiate the EhostDocReader with 'support_overlap=True' in the arguements")
            if token_start >= 0 and token_end > 0:
                span = Span(doc, token_start, token_end, label=classes[id][0])
                for attr_id in classes[id][1]:
                    attr_name = attributes[attr_id][0]
                    attr_value = attributes[attr_id][1]
                    setattr(span._, attr_name, attr_value)
                new_entities.append(span)
                previous_token_offset = token_end
            else:
                raise OverflowError(
                    'The span of the annotation: {}[{}:{}] is out of document boundary.'.format(classes[id][0], start,
                                                                                                end))
            pass
        doc.ents = existing_entities + new_entities
        return doc

    def process_support_overlaps(self, doc):
        """
        Take in a SpaCy doc, add eHOST annotation(Span)s. This function supports adding overlapped annotations.
         :arg doc: a SpaCy doc
        """
        sorted_span, classes, attributes = self.parse_to_dicts(self.anno, sort_spans=True)
        existing_concepts: dict = doc._.concepts
        previous_token_offset = 0
        previous_abs_end = 0
        total = len(doc)
        for id, (start, end) in sorted_span:
            # because SpaCy uses token offset instead of char offset to define Spans, we need to match them,
            # binary search is used here to speed up
            if start < previous_abs_end:
                token_start = self.find_start_token(start, token_start - 1 if token_start > 0 else 0, total, doc)
            else:
                token_start = self.find_start_token(start, previous_token_offset, total, doc)
            token_end = self.find_end_token(end, token_start, total, doc)
            if token_start >= 0 and token_end > 0:
                span = Span(doc, token_start, token_end, label=classes[id][0])
                for attr_id in classes[id][1]:
                    attr_name = attributes[attr_id][0]
                    attr_value = attributes[attr_id][1]
                    setattr(span._, attr_name, attr_value)
                if classes[id][0] not in existing_concepts:
                    existing_concepts[classes[id][0]] = list()
                existing_concepts[classes[id][0]].append(span)
                previous_token_offset = token_end
                previous_abs_end = end
            else:
                raise OverflowError(
                    'The span of the annotation: {}[{}:{}] is out of document boundary.'.format(classes[id][0], start,
                                                                                                end))
            pass
        return doc

    def parse_to_dicts(self, xml_file: str, sort_spans: bool = False) -> Tuple[OrderedDict, OrderedDict]:
        iter = etree.iterparse(xml_file, events=('start',))
        # this doesn't seem elegant, but is said to be the fastest way
        spans = OrderedDict()
        classes = dict()
        attributes = OrderedDict()
        for event, ele in iter:
            if ele.tag == 'annotation':
                id, start, end, span_text = self.parse_annotation_tag(ele, iter)
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
        if sort_spans:
            spans = sorted(spans.items(), key=lambda x: x[1][0])
        return spans, classes, attributes

    # <annotation>
    #   <mention id="EHOST_Instance_1"/>
    #   <annotator id="eHOST_2010">sjl</annotator>
    #   <span start="232" end="236"/>
    #   <spannedText>pain</spannedText>
    #   <creationDate>Sun Apr 19 19:30:11 MDT 2020</creationDate>
    # </annotation>
    def parse_annotation_tag(self, ele: Element, iter: iterparse) -> Tuple:
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


class EhostDirReader(BaseDirReader):
    def __init__(self, txt_dir: Union[str, Path], txt_extension: str = 'txt',
                 nlp: Language = None, docReaderClass: Type = None,
                 recursive: bool = False, use_adjudication: bool = False,
                 schema_file: Union[str, Path] = ''):
        """
        :param txt_dir: the directory contains text files (can be annotation file, if the text content and annotation content are saved in the same file).
        :param txt_extension: the text file extension name (default is 'txt').
        :param nlp: a SpaCy language model
        :param docReaderClass: a DocReader class that can be initiated.
        :param recursive: whether read file recursively down to the subdirectories.
        :param use_adjudication: if read annotations from adjudication folder
        :param schema_file: initiate Span attributes using eHOST schema configuration file
        """
        super().__init__(txt_dir=txt_dir, txt_extension=txt_extension, nlp=nlp, docReaderClass=docReaderClass,
                         recursive=recursive, use_adjudication=use_adjudication, schema_file=schema_file)
        pass
