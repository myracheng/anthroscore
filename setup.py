from setuptools import setup

setup(
    name='anthroscore',
    __version__ = "0.0.0",


    url='https://github.com/myracheng/anthroscore',
    author='Myra Cheng',
    author_email='myra@cs.stanford.edu',

    packages=['anthroscore'],
    
    license='BSD 2-clause',
    install_requires=['regex',
                      'spacy==3.4.4',
                      'numpy',
                      'scipy',
                      'transformers==4.25.1',
                      'pandas',
                      'torch==2.0.1',
                      'en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.1/en_core_web_sm-3.4.1-py3-none-any.whl'
                      ],

    classifiers=[     
        'Programming Language :: Python :: 3.11',
    ],
)