from setuptools import setup
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='anthroscore-eacl',
    __version__ = "0.0.1",


    url='https://github.com/myracheng/anthroscore',
    author='Myra Cheng',
    author_email='myra@cs.stanford.edu',
description = "Package to compute AnthroScore, a computational linguistic measure of anthropomorphism in text",
    packages=['src'],
    license='BSD 2-clause',
    install_requires=['regex',
                      'spacy==3.7.2',
                      'numpy',
                      'scipy',
                      'transformers',
                      'pandas',
                      'torch',
                     ],
	long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[     
        'Programming Language :: Python :: 3.11',
    ],
)
