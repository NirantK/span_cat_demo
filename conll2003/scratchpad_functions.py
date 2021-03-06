import random

import numpy as np
import spacy
from matplotlib.pyplot import axis
from spacy import registry
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, DocBin, Span
from thinc.api import (Config, Model, Ops, get_current_ops, set_dropout_rate,
                       to_numpy)
from thinc.types import Ragged

from pathlib import Path
from typing import Any, Callable, List, Optional

@registry.misc("entity_ngram_suggester.v1")
def build_entity_ngram_suggester(model: str = "en_core_web_sm", sizes: List[int] = [1]) -> Callable[[List[Doc], List[str]], Ragged]:
    """
    Suggester which uses the spaCy Entity Recognizer to suggest spans.
    """
    nlp = spacy.load(model)

    def entity_suggester(docs: List[Doc], *, ops: Optional[Ops] = None) -> Ragged:
        """
        Suggests spans for each entity in the given docs.
        """
        span_groups = []

        if ops is None:
            ops = get_current_ops()

        for doc in docs:
            doc_spans = []
            new_doc = nlp(doc.text)
            
            token_count = len(doc)
            for size in sizes:
                for i in range(token_count - size + 1):
                    doc_spans.append(doc[i: i+size])
            
            # for ent in new_doc.ents:
            #     span = doc.char_span(ent.start_char, ent.end_char)
            #     if span is not None and span not in doc_spans:
            #         doc_spans.append(span)
            

            assert len(doc_spans) > 0

            if len(doc_spans) > 0:
                span_groups.append(doc_spans)
            else:
                span_groups.append([])

        assert len(span_groups) == len(docs)
        return from_spans(span_groups, docs, ops)

    return entity_suggester


@registry.misc("entity_suggester.v1")
def build_entity_suggester(model: str = "en_core_web_sm") -> Callable[[List[Doc], List[str]], Ragged]:
    """
    Suggester which uses the spaCy Entity Recognizer to suggest spans.
    """
    nlp = spacy.load(model)

    def entity_suggester(docs: List[Doc], *, ops: Optional[Ops] = None) -> Ragged:
        """
        Suggests spans for each entity in the given docs.
        """
        span_groups = []

        if ops is None:
            ops = get_current_ops()

        for doc in docs:
            doc_spans = []
            new_doc = nlp(doc.text)
            for ent in new_doc.ents:
                span = doc.char_span(ent.start_char, ent.end_char)
                if span is not None:
                    doc_spans.append(span)
            
            if len(doc_spans) > 0:
                span_groups.append(doc_spans)
            else:
                span_groups.append([])
        assert len(span_groups) == len(docs)
        return from_spans(span_groups, docs, ops)

    return entity_suggester

@registry.misc("ngram_suggester.v2")
def build_ngram_suggester(sizes: List[int]) -> Callable[[List[Doc]], Ragged]:
    """Suggest all spans of the given lengths. Spans are returned as a ragged
    array of integers. The array has two columns, indicating the start and end
    position."""

    def ngram_suggester(docs: List[Doc], *, ops: Optional[Ops] = None) -> Ragged:
        if ops is None:
            ops = get_current_ops()
        spans, lengths = [], []

        for doc in docs:
            doc_spans = []
            starts = ops.xp.arange(len(doc), dtype="i")
            starts = starts.reshape((-1, 1))
            length = 0
            for size in sizes:
                if size <= len(doc):
                    starts_size = starts[: len(doc) - (size - 1)]
                    ngrams = ops.xp.hstack((starts_size, starts_size + size))
                    doc_spans.extend([element for element in ngrams])
                    length += len(ngrams)
            lengths.append(length)
            spans.extend(ops.xp.array(doc_spans))

        if len(spans) > 0:
            output = Ragged(ops.xp.array(spans), ops.asarray(lengths, dtype="i"))
        else:
            output = Ragged(ops.xp.zeros((0,0)), ops.asarray(lengths, dtype="i"))

        assert output.dataXd.ndim == 2
        return output

    return ngram_suggester





def intersect2D(a, b, ops: Ops):
  """
  Find row intersection between 2D numpy arrays, a and b.
  Returns another numpy array with shared rows
  """
  a, b = to_numpy(a), to_numpy(b)
  return ops.asarray([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])

def union2d(a, b, ops:Ops):
    merged = ops.xp.concatenate([a, b], axis=0)
    merged = to_numpy(merged)
    # print(len(merged))
    merged = ops.xp.vstack({tuple(row) for row in merged})
    span_set = set()
    for row in to_numpy(merged):
        span_set.add((row[0], row[1]))
    assert len(span_set) == len(merged)
    return ops.asarray(merged)

def merge_unique_ragged(one: Ragged, two:Ragged, *, ops: Optional[Ops] = None) -> Ragged:
    """Merge two Ragged objects into one Ragged object"""
    if ops is None:
        ops = get_current_ops()

    lengths = []
    one_offset, two_offset = 0, 0 
    for one_l, two_l in zip(one.lengths, two.lengths):
        suggest_one = one.dataXd[one_offset: one_offset+one_l]
        one_offset += one_l
        assert len(suggest_one) == one_l
        suggest_two = two.dataXd[two_offset: two_offset+two_l]
        two_offset += two_l
        assert len(suggest_two) == two_l
        assert suggest_one.ndim == suggest_two.ndim == 2
        common = union2d(suggest_one, suggest_two, ops)
        length = len(common)
        lengths.append(length)

    output = Ragged(data=common, lengths=ops.asarray(lengths, dtype="i"))
    return output

@registry.misc("nounchunk_ngram_suggester.v1")
def build_nounchunk_ngram_suggester(sizes: List[int]) -> Callable[[List[Doc]], Ragged]:
    """
    Suggest all spans of the given lengths. 
    Spans are returned as a ragged array of integers. 
    """

    nlp = spacy.load("en_core_web_sm")

    def nounchunk_ngram_suggester(
        docs: List[Doc], *, ops: Optional[Ops] = None
    ) -> Ragged:
        if ops is None:
            ops = get_current_ops()

        spans, lengths = [], []

        for doc in docs:
            new_doc = nlp(doc.text)
            doc_spans, length = [], 0
            for chunk in new_doc.noun_chunks:
                char_start, char_end = chunk.start_char, chunk.end_char
                span = doc.char_span(char_start, char_end)
                if span is not None:
                    assert (span.end - span.start) > 0
                    element = ops.xp.asarray([span.start, span.end])
                    assert len(element) == 2
                    doc_spans.append(element)
                    length += 1

            lengths.append(length)
            assert length == len(doc_spans)
            if len(doc_spans) > 0:
                spans.extend(ops.xp.array(doc_spans))
            else:
                spans.extend(ops.xp.array(ops.xp.zeros((0, 0))))
        
        ngram_suggester = build_ngram_suggester(sizes = sizes)
        ngrams = ngram_suggester(docs = docs, ops = ops)

        nounchunk_spans = Ragged(data=ops.asarray(spans), lengths=ops.asarray(lengths, dtype="i"))
        nounchunk_ngram_spans = merge_unique_ragged(ngrams, nounchunk_spans, ops = ops)
        return ngrams

    return nounchunk_ngram_suggester


# @registry.misc("train_ngram_suggester.v1")
# def train_ngram_suggester(
#     sizes: List[int], train_corpus: Path
# ) -> Callable[[List[Doc]], Ragged]:
#     """Suggest all spans of the given lengths. Spans are returned as a ragged
#     array of integers. The array has two columns, indicating the start and end
#     position."""

#     # Prepare matcher
#     nlp = spacy.blank("en")
#     docbin = DocBin().from_disk(train_corpus)
#     train_docs = list(docbin.get_docs(nlp.vocab))
#     patterns = set()
#     for doc in train_docs:
#         for ent in doc.ents:
#             patterns.add(nlp.make_doc(ent.text))

#     matcher = PhraseMatcher(nlp.vocab)
#     matcher.add("ENT", list(patterns))

#     def ngram_suggester(docs: List[Doc], *, ops: Optional[Ops] = None) -> Ragged:
#         if ops is None:
#             ops = get_current_ops()
#         spans = []
#         lengths = []

#         for doc in docs:
#             starts = ops.xp.arange(len(doc), dtype="i")
#             starts = starts.reshape((-1, 1))
#             length = 0
#             for size in sizes:
#                 if size <= len(doc):
#                     starts_size = starts[: len(doc) - (size - 1)]
#                     ngrams = ops.xp.hstack((starts_size, starts_size + size))
#                     spans.extend([element for element in ngrams])
#                     length += len(ngrams)
#                 # if spans:
#                 # assert spans[-1].ndim == 2, spans[-1].shape

#             matches = matcher(doc, as_spans=True)
#             for span in matches:
#                 element = ops.xp.hstack((span.start, span.end))
#                 spans.append(element)
#                 length += 1

#             lengths.append(length)

#         if len(spans) > 0:
#             spans = ops.xp.asarray(spans)
#             output = Ragged(ops.xp.vstack(spans), ops.asarray(lengths, dtype="i"))
#         else:
#             output = Ragged(ops.xp.zeros((0, 0)), ops.asarray(lengths, dtype="i"))

#         assert output.dataXd.ndim == 2
#         return output

#     return ngram_suggester

# @registry.misc("random_suggester.v1")
# def build_random_suggester(sizes: List) -> Callable:
#     """
#     Suggests random spans for each doc.

#     Args:
#         sizes (List):

#     Returns:
#         Callable:
#     """
#     random.seed(37)

#     def random_suggester(docs: List[Doc], *, ops: Optional[Ops] = None) -> Ragged:
#         """
#         Suggests 2 spans at random from each doc of a random size from sizes
#         """
#         if ops is None:
#             ops = get_current_ops()

#         spans, lengths = [], []
#         for doc in docs:
#             length = 0
#             token_count = len(doc)
#             doc_spans = []
#             while len(doc_spans) < 2:
#                 start = random.choice(range(token_count - 2))
#                 end = start + 1
#                 element = ops.xp.array([start, end])
#                 doc_spans.append(element)
#                 length += 1

#             if len(doc_spans) > 0 and len(doc_spans) == length:
#                 spans.extend(doc_spans)
#                 lengths.append(length)

#         assert len(spans[-1]) == 2
#         spans = ops.xp.array(spans)
#         assert spans.ndim == 2
#         output = Ragged(spans, ops.xp.array(lengths, dtype="i"))
#         assert output.dataXd.ndim == 2
#         return output

#     return random_suggester