import numpy as np
import spacy
from numpy.testing import assert_equal
from spacy import registry
from spacy.tokens import Doc, DocBin, Span
from spacy.training import Example
from spacy.util import fix_random_seed, registry
from thinc.api import (Config, Model, Ops, get_current_ops, set_dropout_rate,
                       to_numpy)
from thinc.types import Ragged

from functions import build_ngram_suggester, build_entity_suggester, from_indices, from_spans
import pytest
from typing import Any, Callable, List, Optional
SPAN_KEY = "sc"

TEST_DATA = [{
    "sentence": "I am Abdul from Mumbai",
},{
    "sentence": "I used to work for Google Inc."
}]

def test_from_indices(nlp):
    
    # does it raise value error when the lengths do not match
    with pytest.raises(ValueError):
        indices = [
            [2, 3],
            [7, 10],
            [4, 5]
        ]
        lengths = [2, 2]
        from_indices(indices, lengths)

    # does it raise value error when indices is empty
    with pytest.raises(ValueError):
        indices = []
        lengths = []
        from_indices(indices, lengths)
    
    # does it raise value error when any index is None
    with pytest.raises(AttributeError):
        indices = [None, [1, 2], [3, 4]]
        lengths = [1, 2]
        from_indices(indices, lengths)
    
    with pytest.raises(AttributeError):
        indices = [[None, 2], [1, 2], [3, 4]]
        lengths = [1, 2]
        from_indices(indices, lengths)

    # check if an error is raised with indices are not 2d array
    with pytest.raises(ValueError):
        indices = [[1], [3], [5]]
        lengths = [1, 2]
        from_indices(indices, lengths)

def test_from_spans():
    # does it raise an error with span is not in doc
    ops = get_current_ops()
    blank_nlp = spacy.blank("en")

    # Test with valid spans and docs
    docs = [blank_nlp(element["sentence"]) for element in TEST_DATA]
    span_groups = [
        [Span(docs[0], 4, 5), Span(docs[0], 2, 3)],
        [Span(docs[1], 3, 5)]
    ]

    _ = from_spans(span_groups, docs, ops)

    # Test with invalid span groups
    with pytest.raises(AttributeError):
        span_groups = [
            [None, Span(docs[0], 2, 3)],
            [Span(docs[1], 3, 5)]
        ]
        _ = from_spans(span_groups, docs, ops)

# def test_entity_suggester(en_tokenizer):
#     suggester = registry.misc.get("entity_suggester.v1")()
#     docs = [en_tokenizer(element["sentence"]) for element in TEST_DATA]
#     suggestions = suggester(docs)
#     assert suggestions.dataXd.ndim == 2
#     offset = 0
    
#     expected_indices = [[2, 3], [4, 5], [5, 7]]
#     expected_lengths = [2, 1]
   
#     for i, doc in enumerate(docs):
#         sz = suggestions.lengths[i]
#         spans = suggestions.dataXd[offset:offset + sz]
#         spans_set = set()

#         expected_set = expected_indices[offset: offset + sz]

#         for j, span in enumerate(spans):
#             assert_equal(span, expected_set[j])

#         for span in spans:
#             assert 0 <= span[0] < len(doc)
#             assert 0 < span[1] <= len(doc)
#             spans_set.add((span[0], span[1]))
#         # unique spans
#         assert spans.shape[0] == len(spans_set)
#         assert expected_lengths[i] == sz
#         offset += sz
    
    
#     # check if the number of spans is correct
#     # TODO

#     # check if the spans are correct for this specific suggester
#     # test some docs with entities
#     # TODO

#     # test some empty docs
#     # TODO
    
#     # test some docs with no entities
#     # TODO

#     # test all empty docs
#     # TODO

#     # test all docs with no entities
#     # TODO

def test_ngram_suggester_v3(en_tokenizer):
    # test different n-gram lengths
    for size in [1, 2, 3]:
        ngram_suggester = registry.misc.get("ngram_suggester.v3")(sizes=[size])
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
    ngram_suggester = registry.misc.get("ngram_suggester.v3")(sizes=[1, 2, 3])
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
    ngram_suggester = registry.misc.get("ngram_suggester.v3")(sizes=[1])
    docs = [en_tokenizer(text) for text in ["", "a", ""]]
    ngrams = ngram_suggester(docs)
    assert_equal(ngrams.lengths, [len(doc) for doc in docs])

    # test all empty docs
    ngram_suggester = registry.misc.get("ngram_suggester.v3")(sizes=[1])
    docs = [en_tokenizer(text) for text in ["", "", ""]]
    ngrams = ngram_suggester(docs)
    assert_equal(ngrams.lengths, [len(doc) for doc in docs])

def test_ngram_suggester_v2(en_tokenizer):
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
