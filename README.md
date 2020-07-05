# naivenlp

![Python package](https://github.com/luozhouyang/naivenlp/workflows/Python%20package/badge.svg)
[![PyPI version](https://badge.fury.io/py/naivenlp.svg)](https://badge.fury.io/py/naivenlp)
[![Python](https://img.shields.io/pypi/pyversions/naivenlp.svg?style=plastic)](https://badge.fury.io/py/naivenlp)


NLP常用工具包。



主要包含以下模块：

- [naivenlp](#naivenlp)
  - [Tokenizers](#tokenizers)
    - [JiebaTokenizer的使用](#jiebatokenizer的使用)
    - [CustomTokenizer的使用](#customtokenizer的使用)
    - [BasicTokenizer的使用](#basictokenizer的使用)
    - [WordpieceTokenizer的使用](#wordpiecetokenizer的使用)
    - [TransformerTokenizer的使用](#transformertokenizer的使用)
    - [BertTokenizer的使用](#berttokenizer的使用)
  - [Correctors](#correctors)
  - [Similarity](#similarity)


## Tokenizers

`Tokenizer`的作用是**分词**， 同时具有把词语映射到ID的功能。

`naivenlp.tokenizers`模块包含以下`Tokenizer`实现：

* `JiebaTokenizer`，继承自`VocabBasedTokenizer`，分词使用`jieba`
* `CustomTokenizer`，继承自`VocabBasedTokenizer`，基于词典文件的`Tokenizer`，包装`tokenize_fn`自定义函数来实现各种自定义的`Tokenizer`
* `TransformerTokenizer`，继承自`VocabBasedTokenizer`，用于`Transformer`模型分词
* `BertTokenizer`，继承自`VocabBasedTokenizer`，用于`BERT`模型分词



### JiebaTokenizer的使用

分词过程使用`jieba`。

```python
from naivenlp.tokenizers import JiebaTokenizer

tokenizer = JiebaTokenizer(
    vocab_file='vocab.txt',
    pad_token='[PAD]',
    unk_token='[UNK]',
    bos_token='[BOS]',
    eos_token='[EOS]',
)

tokenizer.tokenize('hello world!', mode=0, hmm=True)

tokenizer.encode('hello world!', add_bos=False, add_eos=False)

```

### CustomTokenizer的使用

方便用户自定义分词过程。

以使用`baidu/lac`来分词为例。

```bash
pip install lac
```

```python
from naivenlp.tokenizers import CustomTokenizer

from LAC import LAC

lac = LAC(mode='seg')

def lac_tokenize(text, **kwargs):
    return lac.run(text)


tokenizer = CustomTokenizer(
    vocab_file='vocab.txt',
    tokenize_fn=lac_tokenize,
    pad_token='[PAD]',
    unk_token='[UNK]',
    bos_token='[BOS]',
    eos_token='[EOS]',
)

tokenizer.tokenize('hello world!')

tokenizer.encode('hello world!', add_bos=False, add_eos=False)

```

### BasicTokenizer的使用

这个分词器的使用很简单。不需要词典。它会根据空格来分词。它有以下功能：

* 按照空格和特殊字符分词
* 根据设置，决定是否**大小写转换**
* 根据设置，切分汉字，按照字的粒度分词

```python
from naivenlp.tokenizers import BasicTokenizer

tokenizer = BasicTokenizer(do_lower_case=True, tokenize_chinese_chars=True)

tokenizer.tokenize('hello world, 你好世界')

```


### WordpieceTokenizer的使用

`Wordpiece`是一种分词算法，具体请自己查询相关文档。

`WordpieceTokenizer`需要传入一个词典map。

```python
from naivenlp.tokenizers import WordpieceTokenizer

tokenizer = WordpieceTokenizer(vocab=vocab, unk_token='[UNK]')

tokenizer.tokenize('hello world, 你好世界')
```


### TransformerTokenizer的使用

```python
from naivenlp.tokenizers import TransformerTokenizer


tokenizer = TransformerTokenizer(vocab_file='vocab.txt')

tokenizer.tokenize('Hello World, 你好世界')

tokenizer.encode('Hello World, 你好世界', add_bos=False, add_eos=False)

```

### BertTokenizer的使用

```python
from naivenlp.tokenizers import BertTokenizer


tokenizer = BertTokenizer(vocab_file='vocab.txt', cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]')

tokenizer.tokenize('Hello World, 你好世界')

tokenizer.encode('Hello World, 你好世界', add_bos=False, add_eos=False)

```


## Correctors



## Similarity

多种字符串相似度的度量。是对[luozhouyang/python-string-similarity](https://github.com/luozhouyang/python-string-similarity)的包装。

```python
import naivenlp

a = 'ACCTTTDEX'
b = 'CGGTTEEXX'

naivenlp.cosine_distance(a, b)

naivenlp.cosine_similarity(a, b)

naivenlp.jaccard_distance(a, b)

naivenlp.jaccard_similarity(a, b)

naivenlp.levenshtein_distance(a, b)

naivenlp.levenshtein_distance_normalized(a, b)

naivenlp.levenshtein_similarity(a, b)

naivenlp.weighted_levenshtein_distance(a, b)

naivenlp.damerau_distance(a, b)

naivenlp.lcs_distance(a, b)

naivenlp.lcs_length(a, b)

naivenlp.sorense_dice_distance(a, b)

naivenlp.sorense_dice_similarity(a, b)

naivenlp.optimal_string_alignment_distance(a, b)

```