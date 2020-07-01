# naivenlp

![Python package](https://github.com/luozhouyang/naivenlp/workflows/Python%20package/badge.svg)
[![PyPI version](https://badge.fury.io/py/naivenlp.svg)](https://badge.fury.io/py/naivenlp)
[![Python](https://img.shields.io/pypi/pyversions/naivenlp.svg?style=plastic)](https://badge.fury.io/py/naivenlp)


A naive toolkit for NLP.

## Tokenizers

A tokenizer is used to tokenize text. It can converts tokens to ids, and convert ids to tokens.

Here are some vocab-based tokenizers, which means theses tokenizers need an vocabulary.

* `VocabBasedTokenizer`, base class for vocab-based tokenizers.
* `JiebaTokenizer`, an wrapper for original [fsxjy/jieba](https://github.com/fxsjy/jieba)
* `BasicTokenizer` and `WordpieceTokenizer`, from [google-research/bert](https://github.com/google-research/bert)
* `LanguageModelTokenizer`, a tokenizer for language models. `Transformer`, `BERT` for example.


