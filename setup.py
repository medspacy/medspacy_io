from setuptools import setup
from codecs import open
from os import path
from setuptools.extension import Extension

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


def get_version():
    """Load the version from version.py, without importing it.

    This function assumes that the last line in the file contains a variable defining the
    version string with single quotes.

    """
    try:
        with open('medspacy/version.py', 'r') as f:
            return f.read().split('\n')[0].split('=')[-1].replace('\'', '').strip()
    except IOError:
        return "0.0.0a1"


dir_path = path.dirname(path.realpath(__file__))
include_dirs = [dir_path + "/medspacy", dir_path]
extensions = [
    Extension(
        include_dirs=include_dirs,
    ),
]

setup(
    name='medspacy-io',
    packages=['medspacy-io'],  # this must be the same as the name above
    version=get_version(),
    description='A collection of modules to facilitate reading text from various sources and writing to various sources.',
    author="medSpaCy",
    author_email="medspacy.dev@gmail.com",
    url='https://github.com/medspacy/read_write',  # use the URL to the github repo
    keywords=['medspacy', 'reader', 'writer', 'ehost', 'brat', 'xmi', 'io', 'reader', 'writer', 'nlp', 'annotation'],
    long_description=long_description,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Development Status :: 3 - Alpha',
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    license='MIT License',
    zip_safe=False,
    install_requires=[
        'lxml>=4.4.0', 'spacy>=2.2.2'
    ],
    setup_requires=[
        'lxml>=4.4.0', 'spacy>=2.2.2'
    ],
    test_suite='nose.collector',
    tests_require='nose',
    data_files=[('test_data', ['tests/data/ehost_test_corpus/config/projectschema.xml',
                               'tests/data/ehost_test_corpus/corpus/doc1.txt',
                               'tests/data/ehost_test_corpus/corpus/doc2.txt',
                               'tests/data/ehost_test_corpus/saved/doc1.txt.knowtator.xml',
                               'tests/data/ehost_test_corpus/saved/doc2.txt.knowtator.xml'])],
)
