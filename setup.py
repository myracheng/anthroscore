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
                      'spacy==3.7.2',
                      'numpy',
                      'scipy',
                      'transformers',
                      'pandas',
                      'torch',
                      'en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl#sha256=86cc141f63942d4b2c5fcee06630fd6f904788d2f0ab005cce45aadb8fb73889'
                     ],

    classifiers=[     
        'Programming Language :: Python :: 3.11',
    ],
)
