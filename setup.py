from setuptools import setup

setup(
    name='anthroscore-eacl',
    __version__ = "0.0.0",


    url='https://github.com/myracheng/anthroscore',
    author='Myra Cheng',
    author_email='myra@cs.stanford.edu',

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

    classifiers=[     
        'Programming Language :: Python :: 3.11',
    ],
)
