from collections import OrderedDict
from pathlib import Path
from typing import List, Union, Type, Tuple

from spacy.language import Language
from spacy.syntax.nn_parser import Parser
from spacy.tokens.doc import Doc


class BaseDocReader(object):
    """
    A base class for document reader, define interfaces for subclasses to inherent from
    """

    def __init__(self, nlp: Language = None, support_overlap: bool = False, **kwargs):
        """

        :param nlp: a SpaCy language model
        :param kwargs: other parameters
        """
        for param_name, value in kwargs.items():
            setattr(self, param_name, value)
        if nlp is None:
            raise NameError('parameter "nlp" need to be defined')
        self.nlp = nlp
        self.txt = None
        self.anno = None
        self.support_overlap = support_overlap
        pass

    def get_txt_content(self, txt_file: Path) -> str:
        """
        May need to be overwritten, if the txt_file is not a txt file, e.g. an xmi file
        :param txt_file: the path of a text file
        :return: the content of text file
        """
        return Path(txt_file).read_text(encoding='UTF8')

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
        return Path(anno_file).read_text(encoding='UTF8')

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
        if self.support_overlap:
            doc.set_extension("concepts", default=OrderedDict(), force=True)
        return self.process(doc)

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

    def find_start_token(self, start: int, left_token_offset: int,
                         right_token_offset: int, doc: Doc) -> int:
        """
        Use binary search to find the token-based offset of an annotation (Span) start

        :arg start: the start character offset of input span
        :arg left_token_offset: the left boundary (token offset) of the search window
        :arg right_token_offset: the right boundary (token offset) of the search window
        :arg doc: the input SpaCy Doc
        """
        if right_token_offset >= left_token_offset:
            mid = int(left_token_offset + (right_token_offset - left_token_offset) / 2)
            if doc[mid].idx <= start and start < doc[mid].idx + len(doc[mid]):
                return mid
            elif doc[mid].idx > start:
                return self.find_start_token(start, left_token_offset, mid - 1, doc)
            else:
                return self.find_start_token(start, mid + 1, right_token_offset, doc)
        else:
            return -1

    def find_end_token(self, end: int, token_start: int, total: int, doc: Doc):
        """
        Assume most of the annotations are short (a few tokens), it will more efficient to not use binary search here.
        :arg end: the end character offset of input span
        :arg token_start: the start token offset of the input span
        :arg total: the total number of tokens--check if span'end are invalid
        :arg doc: SpaCy Doc
        """
        for i in range(token_start, total):
            if end <= doc[i].idx + len(doc[i]):
                return i + 1
        return -1


class BaseDirReader:
    """
    A base class for directory reader, define interfaces for subclasses to inherent from
    """

    def __init__(self, txt_dir: Union[str, Path], txt_extension: str = 'txt',
                 nlp: Language = None,
                 docReaderClass: Type = None, recursive: bool = False, **kwargs):
        """

        :param txt_dir: the directory contains text files (can be annotation file, if the text content and annotation content are saved in the same file).
        :param txt_extension: the text file extension name (default is 'txt').
        :param nlp: a SpaCy language model.
        :param docReaderClass: a DocReader class that can be initiated.
        :param recursive: whether read file recursively down to the subdirectories.
        :param kwargs: other parameters that need to pass on to this DirReader or docReaderClass above
        """
        self.txt_dir = self.check_dir_validity(txt_dir)
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
                raise NotADirectoryError(dir)
            else:
                return None
        return dir

    def read(self) -> List[Doc]:
        """
        Read text files and annotation files, return a list of SpaCy Doc, need to be implemented in subclasses
        """
        txt_files = self.list_files(self.txt_dir, self.txt_extension)
        docs = []
        for txt_file in txt_files:
            try:
                doc = self.reader.read(txt_file)
                docs.append(doc)
            except:
                raise IOError('An error occured while parsing annotation for document: {}'.format(txt_file.absolute()))
            pass
        return docs

    def list_files(self, input_dir_path: Path, file_extension: str) -> List[Path]:
        if self.recursive:
            files = list(input_dir_path.rglob('*.' + file_extension))
        else:
            files = list(input_dir_path.glob('*.' + file_extension))
        return files
