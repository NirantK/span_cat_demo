import numpy as np
import spacy
from matplotlib.pyplot import axis
from numpy.testing import assert_equal
from spacy import registry
from spacy.tokens import Doc, DocBin, Span
from spacy.training import Example
from spacy.util import fix_random_seed, registry
from thinc.api import (Config, Model, Ops, get_current_ops, set_dropout_rate,
                       to_numpy)
from thinc.types import Ragged

from functions import build_entity_suggester, build_ngram_suggester
import pytest
from typing import Any, Callable, List, Optional
SPAN_KEY = "sc"


def test_ngram_suggester(en_tokenizer):
    # test different n-gram lengths
    for size in [1, 2, 3]:
        ngram_suggester = registry.misc.get("ngram_suggester.v2")(sizes=[size])
        docs = [
            en_tokenizer(text)
            for text in [
                "a",
                "a b",
                "a b c",
                "a b c d",
                "a b c d e",
                "a " * 100,
            ]
        ]
        ngrams = ngram_suggester(docs)
        assert ngrams.dataXd.ndim == 2
        # span sizes are correct
        for s in ngrams.data:
            assert s[1] - s[0] == size
        # spans are within docs
        offset = 0
        for i, doc in enumerate(docs):
            sz = ngrams.lengths[i]
            spans = ngrams.dataXd[offset : offset + sz]
            spans_set = set()
            for span in spans:
                assert 0 <= span[0] < len(doc)
                assert 0 < span[1] <= len(doc)
                spans_set.add((span[0], span[1]))
            # spans are unique
            assert spans.shape[0] == len(spans_set)
            offset += ngrams.lengths[i]
        # the number of spans is correct
        assert_equal(ngrams.lengths, [max(0, len(doc) - (size - 1)) for doc in docs])

    # test 1-3-gram suggestions
    ngram_suggester = registry.misc.get("ngram_suggester.v2")(sizes=[1, 2, 3])
    docs = [en_tokenizer(text) for text in ["a", "a b", "a b c", "a b c d", "a b c d e"]]
    ngrams = ngram_suggester(docs)
    assert_equal(ngrams.lengths, [1, 3, 6, 9, 12])
    assert_equal(
        ngrams.data,
        [
            # doc 0
            [0, 1],
            # doc 1
            [0, 1],
            [1, 2],
            [0, 2],
            # doc 2
            [0, 1],
            [1, 2],
            [2, 3],
            [0, 2],
            [1, 3],
            [0, 3],
            # doc 3
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [0, 2],
            [1, 3],
            [2, 4],
            [0, 3],
            [1, 4],
            # doc 4
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [0, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [0, 3],
            [1, 4],
            [2, 5],
        ],
    )

    # test some empty docs
    ngram_suggester = registry.misc.get("ngram_suggester.v2")(sizes=[1])
    docs = [en_tokenizer(text) for text in ["", "a", ""]]
    ngrams = ngram_suggester(docs)
    assert_equal(ngrams.lengths, [len(doc) for doc in docs])

    # test all empty docs
    ngram_suggester = registry.misc.get("ngram_suggester.v2")(sizes=[1])
    docs = [en_tokenizer(text) for text in ["", "", ""]]
    ngrams = ngram_suggester(docs)
    assert_equal(ngrams.lengths, [len(doc) for doc in docs])


# def test_spans_ops_missing():
#     """Should raise a Value Error when ops is None"""
#     sentence = ""

#     with pytest.raises(ValueError):
#         from_spans()
