import os

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="naivenlp",
    version="0.0.2",
    description="NLP toolkit, including tokenization, sequence tagging, etc.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/luozhouyang/naivenlp",
    author="ZhouYang Luo",
    author_email="zhouyang.luo@gmail.com",
    packages=setuptools.find_packages(),
    # include_package_data=True,
    package_data={
        "naivenlp": [
            "tokenizers/data/dict.txt",
            "tokenizers/data/prob_emit.p",
            "tokenizers/data/prob_start.p",
            "tokenizers/data/prob_trans.p",
        ]
    },
    install_requires=[
        "jieba",
    ],
    extras_require={

    },
    license="Apache Software License",
    classifiers=(
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    )
)
