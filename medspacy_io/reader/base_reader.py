import logging
from collections import OrderedDict, _OrderedDictItemsView
from pathlib import Path
from typing import List, Union, Type, Tuple

from spacy.language import Language
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span


class BaseDocReader(object):
    """
    A base class for document reader, define interfaces for subclasses to inherent from
    """

    def __init__(self, nlp: Language = None, support_overlap: bool = False,
                 log_level: int = logging.WARNING, encoding: str = None, doc_name_depth: int = 0,
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
        @param kwargs:other parameters
        """
        for param_name, value in kwargs.items():
            setattr(self, param_name, value)
        if nlp is None:
            raise NameError('parameter "nlp" need to be defined')
        self.nlp = nlp
        self.encoding = encoding
        self.doc_name_depth = doc_name_depth
        self.support_overlap = support_overlap
        self.set_logger(log_level)
        if not Span.has_extension('annotation_id'):
            Span.set_extension('annotation_id', default='')
        if not Doc.has_extension('relations'):
            Doc.set_extension('relations', default=[])
        if not Doc.has_extension('doc_name'):
            Doc.set_extension('doc_name', default='')
        pass

    def set_logger(self, level: int = logging.DEBUG):
        self.logger = logging.getLogger(__name__)
        self.logger = logging.getLogger()
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(level)
        pass

    def get_txt_content(self, txt_file: Path) -> str:
        """
        May need to be overwritten, if the txt_file is not a txt file, e.g. an xmi file
        @param txt_file: the path of a text file
        @return: the content of text file
        """
        return Path(txt_file).read_text(encoding=self.encoding)

    def get_anno_content(self, txt_file: Path) -> str:
        """
        Either this method or infer_anno_file_path must be overwritten, as different annotation format
        use different conventions to store annotation files and text files (different in relative paths, or use a
        single file to store both of them)
        @param txt_file: the path of a text file
        @return: the content of the corresponding annotation file (for xml files, need to overwrite this function
        to return the file path string instead)
        """
        anno_file = self.infer_anno_file_path(txt_file)
        return Path(anno_file).read_text(encoding=self.encoding)

    def infer_anno_file_path(self, txt_file: Path) -> Path:
        """
        From the path of the text file, infer the corresponding annotation file. Need to be implemented for each subclass,
        as different annotation format use different conventions to store annotation files and txt files
        @param txt_file: the Path of a txt file
        @return: the Path of the corresponding annotation file
        """
        return None

    def check_file_validity(self, file: Union[str, Path], raise_error: bool = True) -> Path:
        """

        @param file: a string or Path of target file path
        @param raise_error: pause the program if an error is encountered
        @return: Path of the file
        """
        if file is None or (isinstance(file, str) and len(file) == 0):
            if raise_error:
                raise NameError(file)
            else:
                return None
        if isinstance(file, str):
            file = Path(file)
        if not file.exists() or not file.is_file():
            if raise_error:
                raise FileNotFoundError(file.absolute())
            else:
                return None
        return file

    def read(self, txt_file: Union[str, Path]) -> Doc:
        """
        Read annotations and return a Spacy Doc, need to be implemented in subclasses

        @param txt_file: the text file path in an annotation project, where the corresponding
         annotation file path can be inferred through method: infer_anno_file_path
        """
        txt, anno = self.get_contents(txt_file=txt_file)
        doc = self.nlp(txt)
        doc._.doc_name = self.get_doc_name(txt_file, self.doc_name_depth)
        if self.support_overlap:
            if not doc.has_extension("concepts"):
                doc.set_extension("concepts", default=OrderedDict())
        return self.process(doc, anno)

    def get_doc_name(self, txt_file: Union[str, Path], doc_name_depth: int = 0) -> str:
        """
        @param txt_file: the path of txt file (annotation file's location will be inferred )
        @return: a string to put in the Doc._.doc_name
        """
        txt_file_path = txt_file if isinstance(txt_file, Path) else Path(txt_file)
        base_name = txt_file_path.name
        if doc_name_depth == -1:
            base_name = str(txt_file_path.absolute())
        elif doc_name_depth > 0:
            base_name = str(txt_file_path.absolute() \
                            .relative_to(txt_file_path.absolute().parents[doc_name_depth]))
        return base_name

    def process(self, doc: Doc, anno: str) -> Doc:
        """

        @param doc: An initiated SpaCy Doc
        @param anno: The annotation string (can be a file path or file content, depends on how get_anno_content is implemented)
        @return: Annotation-added SpaCy Doc
        """
        sorted_span, classes, attributes, relations = self.parse_to_dicts(anno, sort_spans=True)
        #print("PARSE_TO_DIC attributes:", attributes)
        if self.support_overlap:
            return self.process_support_overlaps(doc, sorted_span, classes, attributes, relations)
        else:
            return self.process_without_overlaps(doc, sorted_span, classes, attributes, relations)

    def parse_to_dicts(self, anno: str, sort_spans: bool = False) -> Tuple[
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
        return (None, None, None, None,)

    def process_without_overlaps(self, doc: Doc, sorted_spans: _OrderedDictItemsView, classes: OrderedDict,
                                 attributes: OrderedDict,
                                 relations: OrderedDict) -> Doc:
        """:arg a SpaCy Doc, can be overwriten by the subclass as needed.
            This function will add spans to doc.ents (defined by SpaCy as default)
            which doesn't allow overlapped annotations.
            @param doc: initiated SpaCy Doc
            @param sorted_spans: a sorted OrderedDict Items ( spans[entity_id] = (start, end, span_text))
            @param classes: a OrderedDict to map a entity id to [entity label, [attr_ids]]
            @param attributes: a OrderedDict to map a attribute id to (attribute_name, attribute_value)
            @param relations: a OrderedDict to map a relation_id to (label, (relation_component_ids))
            @return: annotated Doc
        """
        existing_entities = list(doc.ents)
        new_entities = list()
        # token_left_bound = 0
        token_right_bound = len(doc) - 1
        token_start = -1
        token_end = -1
        for id, span_tuple in sorted_spans:
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
                token_start = token_right_bound - 1
                token_end = token_right_bound
            else:
                token_start = self.find_start_token(start, token_start, token_right_bound, doc)
                if end >= doc[-1].idx + doc[-1].__len__():
                    token_end = token_right_bound + 1
                else:
                    token_end = self.find_end_token(end, token_start, token_right_bound, doc)
            if token_start < 0 or token_start >= token_right_bound or token_end < 0 or token_end > token_right_bound:
                raise ValueError(
                    "It is likely your annotations overlapped, which process_without_overlaps doesn't support parsing "
                    "those. You will need to initiate the EhostDocReader with 'support_overlap=True' in the arguements")
            if token_start >= 0 and token_end > 0:
                span = Span(doc, token_start, token_end, label=classes[id][0])
                span._.annotation_id = id
                for attr_id in classes[id][1]:
                    if attr_id not in attributes:
                        continue
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
        doc.ents = existing_entities + new_entities
        rels = []
        #print(relations)
        #print(classes) rel_source, (rel_s, rel_name, rel_target)
        for rel_source, (rel_s, rel_name, rel_target) in relations.items():
            source_span = None
            target_span = None
            for ent in doc.ents:
                for cls in classes[ent._.annotation_id][1]:
                    if cls == rel_source:
                        source_span = ent
                        break
                if ent._.annotation_id == rel_target:
                    target_span = ent
            rels.append((source_span,target_span,rel_name))
        doc._.relations = rels
         
        return doc

    def process_support_overlaps(self, doc: Doc, sorted_spans: _OrderedDictItemsView, classes: OrderedDict,
                                 attributes: OrderedDict,
                                 relations: OrderedDict) -> Doc:
        """:arg a SpaCy Doc, can be overwriten by the subclass as needed.
            This function will add spans to doc._.concepts (defined in 'read' function above,
            which allows overlapped annotations.
            @param doc: initiated SpaCy Doc
            @param sorted_spans: a sorted OrderedDict Items ( spans[entity_id] = (start, end, span_text))
            @param classes: a OrderedDict to map a entity id to [entity label, [attr_ids]]
            @param attributes: a OrderedDict to map a attribute id to (attribute_name, attribute_value)
            @param relations: a OrderedDict to map a relation_id to (label, (relation_component_ids))
            @return: annotated Doc
        """
        #print("ATTRIBUTES Ordered Dic:", attributes, "ATTRIBUTES name list:", attributes.keys())
        existing_concepts: dict = doc._.concepts
        # token_left_bound = 0
        previous_abs_end = 0
        token_right_bound = len(doc) - 1
        token_start = -1
        token_end = -1
        for id, span_tuple in sorted_spans:
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
                self.logger.debug("token_start {} >= token_right_bound {}".format(token_start, token_right_bound))
                token_start = token_right_bound
                token_end = token_right_bound + 1
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
                                  .format(start, token_start, doc[token_start].idx,
                                          token_right_bound, doc[token_right_bound].idx), )
                token_start = self.find_start_token(start, token_start, token_right_bound, doc)
                self.logger.debug("\tfind start token {}('{}')".format(token_start, doc[token_start]))
                if end >= doc[-1].idx + doc[-1].__len__():
                    self.logger.debug("end  ({}) >= doc[-1].idx ({}) + doc[-1].__len__() ({})".format(end, doc[-1].idx,
                                                                                                      doc[
                                                                                                          -1].__len__()))
                    token_end = token_right_bound + 1
                else:
                    self.logger.debug(
                        "To find token_end starts from {} between token_start ({}[{}]) and  token_right_bound({}[{}])"
                            .format(end, token_start, doc[token_start].idx,
                                    token_right_bound, doc[token_right_bound].idx))
                    token_end = self.find_end_token(end, token_start, token_right_bound, doc)
                    self.logger.debug("\tFind end token {}('{}')".format(token_end, doc[token_end]))
            if token_start >= 0 and token_end > 0:
                span = Span(doc, token_start, token_end, label=classes[id][0])
                span._.annotation_id = id #This is added, otherwise relation cannot be referred
                if self.logger.isEnabledFor(logging.DEBUG):
                    import re
                    if re.sub('\s+', ' ', span._.span_txt) != re.sub('\s+', ' ', str(span)):
                        self.logger.debug('{}[{}:{}]\n\t{}<>\n\t{}<>'.format(classes[id][0], token_start, token_end,
                                                                             re.sub('\s+', ' ', span._.span_txt),
                                                                             re.sub('\s+', ' ', str(span))))

                for attr_id in classes[id][1]:

                    if attr_id not in attributes:
                        continue
                    attr_name = attributes[attr_id][0]
                    attr_value = attributes[attr_id][1]
                    print("THE ATTRIBUTES FOR:", classes[id][0], " IS ", attributes[attr_id][0], attributes[attr_id][1])
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
            #pass
        rels = []
        for rel_source, (rel_s, rel_name, rel_target) in relations.items():
            print("RELATION:",rel_source,rel_s, rel_name,rel_target)
            source_span = None
            target_span = None
            for cls in doc._.concepts.keys():
                for sp in doc._.concepts[cls]:
                    #print("---ALL EHOST ID:", sp._.annotation_id, sp.text)
                    if sp._.annotation_id == rel_s:
                        print("FIND THE SOURCE:", sp._.annotation_id, rel_s)
                        source_span = sp
                        #break
                    if sp._.annotation_id == rel_target:
                        target_span = sp
            if (source_span is not None) and (target_span is not None):
                rels.append((source_span, target_span, rel_name))
        doc._.relations = rels

        return doc

    def get_contents(self, txt_file: Union[str, Path]) -> Tuple[str, str]:
        """

        @param txt_file: a string or Path of a text file
        @return: a string of text file content and corresponding annotation file content (for xml file, return annotation file path instead)
        """
        txt_file = self.check_file_validity(txt_file)
        txt = ''
        anno = ''
        if txt_file is not None:
            txt = self.get_txt_content(txt_file)
            anno = self.get_anno_content(txt_file)
        return txt, anno

    def find_start_token(self, start: int, token_left_bound: int,
                         token_right_bound: int, doc: Doc) -> int:
        """
        Use binary search to find the token-based offset of an annotation (Span) start

        :arg start: the start character offset of input span
        :arg token_left_bound: the left boundary (token offset) of the search window
        :arg token_right_bound: the right boundary (token offset) of the search window
        :arg doc: the input SpaCy Doc
        """
        if token_right_bound > token_left_bound:
            mid = int(token_left_bound + (token_right_bound - token_left_bound) / 2)
            self.logger.debug(
                "Find abs offset {} between {}[{}:{}] and {}[{}:{}], mid={}"
                    .format(start, token_left_bound, doc[token_left_bound].idx,
                            doc[token_left_bound].idx + doc[token_left_bound].__len__(),
                            token_right_bound, doc[token_right_bound].idx,
                            doc[token_right_bound].idx + doc[token_right_bound].__len__(), mid))
            if doc[mid].idx <= start < doc[mid].idx + len(doc[mid]):
                self.logger.debug(
                    "return mid={} when doc[{}]({}) < {} < doc[{}].idx+len(doc[{}])({})"
                        .format(mid, mid, doc[mid].idx, start, mid, mid, doc[mid].idx + len(doc[mid])))
                return mid
            elif mid > 1 and doc[mid].idx > start >= doc[mid - 1].idx + len(doc[mid - 1]):
                # sometime, manually created annotation can start outside SpaCy tokens, so adjustment is needed here.
                self.logger.debug(
                    "return mid={} when start {} between token {} and token []".format(mid, start, mid, mid - 1))
                return mid
            elif doc[mid].idx > start:
                if mid > 0:
                    return self.find_start_token(start, token_left_bound, mid - 1, doc)
                else:
                    return -1
            else:
                if mid < len(doc) - 1:
                    return self.find_start_token(start, mid + 1, token_right_bound, doc)
                else:
                    return -1;

        elif token_right_bound == token_left_bound:
            return token_left_bound
        else:
            return -1

    def find_end_token(self, end: int, token_left_bound: int, token_right_bound: int, doc: Doc):
        """
        Assume most of the annotations are short (a few tokens), it will more efficient to not use binary search here.
        :arg end: the end character offset of input span
        :arg token_left_bound: the start token offset of the input span
        :arg token_right_bound: the end boundary of tokens--check if span'end are invalid
        :arg doc: SpaCy Doc
        """
        for i in range(token_left_bound, token_right_bound + 1):
            if end <= doc[i].idx + doc[i].__len__():
                return i + 1
        return -1


class BaseDirReader:
    """
    A base class for directory reader, define interfaces for subclasses to inherent from
    """

    def __init__(self, txt_extension: str = 'txt', recursive: bool = False,
                 nlp: Language = None,
                 docReaderClass: Type = None, **kwargs):
        """
        @param txt_extension: the text file extension name (default is 'txt').
        @param recursive: whether read file recursively down to the subdirectories.
        @param nlp: a SpaCy language model.
        @param docReaderClass: a DocReader class that can be initiated.
        @param kwargs: other parameters that need to pass on to this DirReader or docReaderClass above
        """
        self.txt_extension = txt_extension
        self.recursive = recursive
        if docReaderClass is None or not issubclass(docReaderClass, BaseDocReader):
            raise NameError('docReaderClass must not be None, and must be a subclass of "BaseDocReader."')
        if nlp is None:
            raise NameError('parameter "nlp" need to be defined')
        self.nlp = nlp
        self.kwargs = kwargs
        self.reader = docReaderClass(nlp=self.nlp, **self.kwargs)
        pass

    def check_dir_validity(self, dir: Union[str, Path], raise_error: bool = True) -> Path:
        """
        Check if the given directory is valid
        @param dir: a string or Path of given directory
        @param raise_error: whether raise an error if there is
        @return: the Path of the given directory
        """
        if dir is None or (isinstance(dir, str) and len(dir) == 0):
            if raise_error:
                raise NameError(dir)
            else:
                return None
        if isinstance(dir, str):
            dir = Path(dir)
        if not dir.exists() or not dir.is_dir():
            if raise_error:
                raise NotADirectoryError(dir.absolute())
            else:
                return None
        return dir

    def read(self, txt_dir: Union[str, Path]) -> List[Doc]:
        """
        Read text files and annotation files, return a list of SpaCy Doc, need to be implemented in subclasses
        @param txt_dir: the directory contains text files (can be annotation file, if the text content and annotation content are saved in the same file).
        @return: a list of SpaCy Docs
        """
        txt_dir = self.check_dir_validity(txt_dir)
        if txt_dir is None:
            return []
        txt_files = self.list_files(txt_dir, self.txt_extension, self.recursive)
        docs = []
        for txt_file in txt_files:
            try:
                doc = self.reader.read(txt_file)
                docs.append(doc)
            except:
                raise IOError('An error occured while parsing annotation for document: {}'.format(txt_file.absolute()))
            pass
        return docs

    def list_files(self, input_dir_path: Path, file_extension: str, recursive: bool = False) -> List[Path]:
        if recursive:
            files = list(input_dir_path.rglob('*.' + file_extension))
        else:
            files = list(input_dir_path.glob('*.' + file_extension))
        return files
