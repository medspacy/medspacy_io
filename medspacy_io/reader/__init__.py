# This package contains a list of annotation readers, which take in annotation files (with or without document text files) and construct a spacy doc
from .base_reader import BaseDocReader, BaseDirReader
from .ehost_reader import EhostDocReader, EhostDirReader
