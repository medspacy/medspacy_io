import logging
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
                 schema_file: Union[str, Path] = '', store_anno_string: bool = False,
                 log_level: bool = logging.WARNING, encoding:str=None, **kwargs):
        """

        :param nlp: a SpaCy language model
        :param support_overlap: if the EhostDocReader need to support reading from overlapped annotations.
            Because SpaCy's Doc.ents does not allows overlapped Spans, to support overlapping, Spans need to be stored
            somewhere else----Doc._.concepts
        :param use_adjudication: if read annotations from adjudication folder
        :param schema_file: initiate Span attributes using eHOST schema configuration file
        :param store_anno_string: whether read annotated string from annotations to double check parsed Span's correction
        :param log_level: set the logger's logging level. TO debug, set to logging.DEBUG
        """
        self.schema_set = False
        self.set_attributes(schema_file=schema_file)
        if store_anno_string:
            if not Span.has_extension("span_txt"):
                Span.set_extension("span_txt", default="")
        super().__init__(nlp=nlp, support_overlap=support_overlap, use_adjudication=use_adjudication,
                         store_anno_string=store_anno_string, log_level=log_level,encoding=encoding, **kwargs)
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
        # token_left_bound = 0
        token_right_bound = len(doc) - 1
        token_start = -1
        token_end = -1
        for id, span_tuple in sorted_span:
            # because SpaCy uses token offset instead of char offset to define Spans, we need to match them,
            # binary search is used here to speed up
            if self.store_anno_string:
                start, end, span_txt = span_tuple
            else:
                start, end = span_tuple
            # because SpaCy uses token offset instead of char offset to define Spans, we need to match them,
            # binary search is used here to speed up
            if start < doc[0].idx:
                # If the annotation fall into a span that is before the 1st Spacy token, adjust the span to the 1st
                # token
                token_start = 0
                token_end = 1
            elif token_start >= token_right_bound:
                # If the annotation fall into a span that is after the last Spacy token, adjust the span to the last
                # token
                token_start = token_right_bound - 2
                token_end = token_right_bound - 1
            else:
                token_start = self.find_start_token(start, token_start, token_right_bound, doc)
                if end >= doc[-1].idx + doc[-1].__len__():
                    token_end = token_right_bound - 1
                else:
                    token_end = self.find_end_token(end, token_start, token_right_bound, doc)
            if token_start < 0 or token_start >= token_right_bound or token_end < 0 or token_end > token_right_bound:
                raise ValueError(
                    "It is likely your annotations overlapped, which process_without_overlaps doesn't support parsing "
                    "those. You will need to initiate the EhostDocReader with 'support_overlap=True' in the arguements")
            if token_start >= 0 and token_end > 0:
                span = Span(doc, token_start, token_end+1, label=classes[id][0])
                for attr_id in classes[id][1]:
                    attr_name = attributes[attr_id][0]
                    attr_value = attributes[attr_id][1]
                    setattr(span._, attr_name, attr_value)
                if self.store_anno_string and span_txt is not None:
                    setattr(span._, "span_txt", span_txt)
                new_entities.append(span)
                token_start = token_end
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
        # token_left_bound = 0
        previous_abs_end = 0
        token_right_bound = len(doc) - 1
        token_start = -1
        token_end = -1
        for id, span_tuple in sorted_span:
            # because SpaCy uses token offset instead of char offset to define Spans, we need to match them,
            # binary search is used here to speed up
            if self.store_anno_string:
                start, end, span_txt = span_tuple
            else:
                start, end = span_tuple
            if start < doc[0].idx:
                # If the annotation fall into a span that is before the 1st Spacy token, adjust the span to the 1st
                # token
                token_start = 0
                token_end = 1
            elif token_start >= token_right_bound:
                # If the annotation fall into a span that is after the last Spacy token, adjust the span to the last
                # token
                self.logger.debug("token_start {} >= token_right_bound {}".format(token_start,token_right_bound))
                token_start = token_right_bound
                token_end = token_right_bound+1
            else:
                # if start < previous_abs_end:
                #     self.logger.debug("To find {} between token_start - 1({}[{}]) and  token_right_bound({}[{}])"
                #                       .format(start, token_start-1, doc[token_start-1].idx,
                #                               token_right_bound, doc[token_right_bound].idx), )
                #     token_start = self.find_start_token(start, token_start - 1 if token_start > 0 else 0,
                #                                         token_right_bound, doc)
                #     self.logger.debug('\tfind token_start={}[{}]'.format(token_start, doc[token_start].idx))
                #
                # else:
                self.logger.debug("To find {} between token_start ({}[{}]) and  token_right_bound({}[{}])"
                                  .format(start, token_start , doc[token_start ].idx,
                                          token_right_bound, doc[token_right_bound].idx), )
                token_start = self.find_start_token(start, token_start, token_right_bound, doc)
                self.logger.debug("\tfind start token {}('{}')".format(token_start, doc[token_start]))
                if end >= doc[-1].idx + doc[-1].__len__():
                    self.logger.debug("end  ({}) >= doc[-1].idx ({}) + doc[-1].__len__() ({})".format(end, doc[-1].idx , doc[-1].__len__()))
                    token_end = token_right_bound+1
                else:
                    self.logger.debug("To find token_end starts from {} between token_start ({}[{}]) and  token_right_bound({}[{}])"
                                      .format(end, token_start, doc[token_start].idx,
                                              token_right_bound, doc[token_right_bound].idx))
                    token_end = self.find_end_token(end, token_start, token_right_bound, doc)
                    self.logger.debug("\tFind end token {}('{}')".format(token_end, doc[token_end]))
            if token_start >= 0 and token_end > 0:
                span = Span(doc, token_start, token_end, label=classes[id][0])
                if self.logger.isEnabledFor(logging.DEBUG):
                    import re
                    if re.sub('\s+',' ',span._.span_txt) != re.sub('\s+',' ',str(span)):
                        self.logger.debug('{}[{}:{}]\n\t{}<>\n\t{}<>'.format(classes[id][0],token_start,token_end,re.sub('\s+',' ',span._.span_txt), re.sub('\s+',' ',str(span))))
                for attr_id in classes[id][1]:
                    attr_name = attributes[attr_id][0]
                    attr_value = attributes[attr_id][1]
                    setattr(span._, attr_name, attr_value)
                if self.store_anno_string and span_txt is not None:
                    setattr(span._, "span_txt", span_txt)
                if classes[id][0] not in existing_concepts:
                    existing_concepts[classes[id][0]] = list()
                existing_concepts[classes[id][0]].append(span)
                # token_start = token_end
                previous_abs_end = token_start

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
                 nlp: Language = None, docReaderClass: Type = None, support_overlap: bool = False,
                 recursive: bool = False, use_adjudication: bool = False,
                 schema_file: Union[str, Path] = '', **kwargs):
        """
        :param txt_dir: the directory contains text files (can be annotation file, if the text content and annotation
        content are saved in the same file). :param txt_extension: the text file extension name (default is 'txt').
        :param nlp: a SpaCy language model :param docReaderClass: a DocReader class that can be initiated. :param
        recursive: whether read file recursively down to the subdirectories. :param use_adjudication: if read
        annotations from adjudication folder :param schema_file: initiate Span attributes using eHOST schema
        configuration file
        """
        super().__init__(txt_dir=txt_dir, support_overlap=support_overlap, txt_extension=txt_extension, nlp=nlp,
                         docReaderClass=docReaderClass, recursive=recursive, use_adjudication=use_adjudication,
                         schema_file=schema_file, **kwargs)
        pass
