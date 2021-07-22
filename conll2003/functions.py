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


def from_indices(indices: List[Any], lengths: List, ops: Optional[Ops] = None) -> Ragged:
    """
    Make Ragged spans for training and inference.
    Used as a data type convertor in suggester for Span Categorizer

    Args:
        indices (List[Any]): Python list with np, ops.xp, or List elements
        lengths (List): Number of span elements in each doc

    Raises:
        ValueError: In case of input values not meeting expected data ranges
        TypeError: Invalid datatype as input

    Returns:
        Ragged: the Thinc datatype for training
    """
    if ops is None:
        ops =  get_current_ops()
    
    if not isinstance(indices, list): 
        raise TypeError(f"Expected list, got {type(indices)}")

    # check if sum of lengths is same as length of indices, if not raise ValueError
    if not np.allclose(np.sum(lengths), len(indices)):
        raise ValueError("lengths of indices and sum of lengths do not match.")

    if len(indices) == 0:
        raise ValueError(
            f"There were no (start, end) pairs found. Check if indices input is empty"
        )

    # check if any element is None, if yes raise ValueError
    if any(x is None for x in indices):
        raise AttributeError(
            f"Got a None instead of (start, end) pair integer values. Check if indices input is correct"
        )
    
    for pair in indices:
        if any(x is None for x in pair):
            raise AttributeError(
            f"There was a None in one of the (start, end) pairs. Check if indices input is correct"
        )


    indices = ops.asarray(indices)
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError(
                f"Expected indices to be a 2d matrix, with each row being a [start, end] pair"
            )

    output = Ragged(data = indices, lengths = ops.asarray(lengths, dtype="i"))
    assert output.dataXd.ndim == 2
    return output


def from_spans(
    span_groups: List[List], docs: List[Doc], ops: Optional[Ops] = None
) -> Ragged:
    """
    Convert a list of spans into a Ragged object
    """
    if ops is None:
        raise ValueError(f"ops cannot be None")
    indices = []
    lengths = []
    for doc, spans in zip(docs, span_groups):
        for span in spans:
            indices.append(ops.xp.array([span.start, span.end]))
        if len(spans) > 0:
            lengths.append(len(spans))
        else:
            indices.append(ops.xp.zeros((0, 0)))
            lengths.append(0)
    return from_indices(indices, lengths, ops=ops)

@registry.misc("ngram_suggester.v3")
def build_ngram_suggester_from_spans(sizes: List[int]) -> Callable[[List[Doc]], Ragged]:
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