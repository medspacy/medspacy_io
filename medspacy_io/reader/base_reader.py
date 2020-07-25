import logging
from collections import OrderedDict
from pathlib import Path
from typing import List, Union, Type

from spacy.language import Language
from spacy.tokens.doc import Doc


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
        self.txt = None
        self.anno = None
        self.encoding = encoding
        self.doc_name_depth = doc_name_depth
        self.support_overlap = support_overlap
        self.set_logger(log_level)
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
        :param txt_file: the path of a text file
        :return: the content of text file
        """
        return Path(txt_file).read_text(encoding=self.encoding)

    def get_anno_content(self, txt_file: Path) -> str:
        """
        Either this method or infer_anno_file_path must be overwritten, as different annotation format
        use different conventions to store annotation files and text files (different in relative paths, or use a
        single file to store both of them)
        :param txt_file: the path of a text file
        :return: the content of text file
        """
        anno_file = self.infer_anno_file_path(txt_file)
        self.check_file_validity(anno_file)
        return Path(anno_file).read_text(encoding=self.encoding)

    def infer_anno_file_path(self, txt_file: Path) -> Path:
        """
        From the path of the text file, infer the corresponding annotation file. Need to be implemented for each subclass,
        as different annotation format use different conventions to store annotation files and txt files
        """
        return None

    def check_file_validity(self, file: Union[str, Path], raise_error: bool = True) -> Path:
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

        :param txt_file: the text file path in an annotation project, where the corresponding
         annotation file path can be inferred through method: infer_anno_file_path
        """
        self.get_contents(txt_file=txt_file)
        doc = self.nlp(self.txt)
        doc._.doc_name = self.get_doc_name(txt_file, self.doc_name_depth)
        if self.support_overlap:
            if not doc.has_extension("concepts"):
                doc.set_extension("concepts", default=OrderedDict())
        return self.process(doc)

    def get_doc_name(self, txt_file: Union[str, Path], doc_name_depth: int = 0) -> str:
        """
        @param txt_file: the path of txt file (annotation file's location will be inferred )
        @return: a string to put in the doc_name
        """
        txt_file_path = txt_file if isinstance(txt_file, Path) else Path(txt_file)
        base_name = txt_file_path.name
        if doc_name_depth == -1:
            base_name = str(txt_file_path.absolute())
        elif doc_name_depth > 0:
            base_name = str(txt_file_path.absolute() \
                            .relative_to(txt_file_path.absolute().parents[doc_name_depth]))
        return base_name

    def process(self, doc):
        """:arg a SpaCy Doc, must be implemented in the subclass."""
        if self.support_overlap:
            return self.process_support_overlaps(doc)
        else:
            return self.process_without_overlaps(doc)

    def process_without_overlaps(self, doc):
        """:arg a SpaCy Doc, can be implemented in the subclass as needed.
            This function will add spans to doc.ents (defined by SpaCy as default)
            which doesn't allow overlapped annotations.
        """
        return doc

    def process_support_overlaps(self, doc):
        """:arg a SpaCy Doc, can be implemented in the subclass as needed.
            This function will add spans to doc._.concepts (defined in 'read' function above,
            which allows overlapped annotations.
        """
        return doc

    def get_contents(self, txt_file: Union[str, Path]):
        txt_file = self.check_file_validity(txt_file)
        if txt_file is not None:
            self.txt = self.get_txt_content(txt_file)
            self.anno = self.get_anno_content(txt_file)
        pass

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

    def __init__(self, txt_extension: str = 'txt',
                 nlp: Language = None,
                 docReaderClass: Type = None, recursive: bool = False, **kwargs):
        """
        :param txt_extension: the text file extension name (default is 'txt').
        :param nlp: a SpaCy language model.
        :param docReaderClass: a DocReader class that can be initiated.
        :param recursive: whether read file recursively down to the subdirectories.
        :param kwargs: other parameters that need to pass on to this DirReader or docReaderClass above
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
        :param txt_dir: the directory contains text files (can be annotation file, if the text content and annotation content are saved in the same file).
        :return: a list of SpaCy Docs
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
