from matplotlib.pyplot import axis
import spacy
from spacy import registry
from spacy.tokens import Doc, DocBin, Span
from spacy.matcher import PhraseMatcher
from typing import List, Callable, Optional, Any
from thinc.types import Ragged
from thinc.api import Config, Model, get_current_ops, set_dropout_rate, Ops, to_numpy
from pathlib import Path
import numpy as np

def from_indices(indices: List[Any], lengths:List, ops: Optional[Ops] = None)->Ragged:
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
    # check if sum of lengths is same as length of indices, if not raise ValueError
    if not np.allclose(np.sum(lengths), len(indices)):
        raise ValueError("Sum of lengths of indices and lengths do not match.")
    
    if len(indices) == 0:
        raise ValueError(f"There were no (start, end) pairs found. Check if indices input is empty")
    
    # check if any element is None, if yes raise ValueError
    if any(x is None for x in indices):
        raise ValueError(f"There were (start, end) pairs with None values. Check if indices input is correct")
    
    if type(indices) == type([]) and type(indices[-1]) == type(ops.xp.array([1])):
        indices = ops.xp.array(indices) 
        if indices.ndim != 2:
            raise ValueError(f"Expected indices to be a 2d matrix, with each row being a [start, end] pair")

    output = Ragged(indices, ops.asarray(lengths, dtype="i"))
    assert output.dataXd.ndim == 2
    return output

def from_spans(span_groups: List[List], docs: List[Doc], ops: Optional[Ops] = None)->Ragged:
    """
    Convert a list of spans into a Ragged object
    """
    if ops is None:
        raise ValueError(f"ops cannot be None")
    indices = []
    lengths = []
    for doc, spans in zip(docs, span_groups):
        for span in spans:
            start, end  = span.start, span.end
            assert start < end 
            assert end <= len(doc)
            indices.append(ops.xp.array([start, end]))
            
        lengths.append(len(spans))
    return from_indices(indices, lengths, ops=ops)

@registry.misc("entity_suggester.v1")
def build_entity_suggester(model:str ="en_core_web_sm", make_diff_doc: bool = True) -> Callable[[List[Doc], List[str]], Ragged]:
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
                char_start, char_end = ent.start_char, ent.end_char
                span = doc.char_span(char_start, char_end)
                if span is not None:
                    doc_spans.append(span)

            span_groups.append(doc_spans)
    
        return from_spans(span_groups, docs, ops)

    return entity_suggester

import random

@registry.misc("random_suggester.v1")
def build_random_spam_suggester(sizes: List)->Callable:
    """
    Suggests random spans for each doc.

    Args:
        sizes (List): 

    Returns:
        Callable:
    """
    random.seed(37)

    def random_suggester(docs: List[Doc], *, ops: Optional[Ops] = None) -> Ragged:
        """
        Suggests 2 spans at random from each doc of a random size from sizes
        """
        if ops is None:
            ops = get_current_ops()

        spans, lengths = [], []
        for doc in docs: 
            length = 0
            token_count = len(doc)
            doc_spans = []
            while len(doc_spans) < 2 :
                start = random.choice(range(token_count-2))
                end = start + 1
                element = ops.xp.array([start, end])
                doc_spans.append(element)
                length += 1
            
            if len(doc_spans) > 0 and len(doc_spans) == length:
                spans.extend(doc_spans)
                lengths.append(length)

        assert len(spans[-1]) == 2
        spans = ops.xp.array(spans)
        assert spans.ndim == 2
        output = Ragged(spans, ops.xp.array(lengths, dtype="i"))
        assert output.dataXd.ndim == 2
        return output

    return random_suggester

@registry.misc("nounchunk_ngram_suggester.v1")
def build_nounchunk_ngram_suggester(sizes: List[int], train_corpus: Path) -> Callable[[List[Doc]], Ragged]:
    """Suggest all spans of the given lengths. Spans are returned as a ragged
    array of integers. The array has two columns, indicating the start and end
    position."""
    
    nlp = spacy.load("en_core_web_sm")

    def nounchunk_ngram_suggester(docs: List[Doc], *, ops: Optional[Ops] = None) -> Ragged:
        if ops is None:
            ops = get_current_ops()  

        spans = []
        lengths = []

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

            new_doc = nlp(doc.text)       
            for chunk in new_doc.noun_chunks:
                char_start, char_end = chunk.start_char, chunk.end_char
                span = doc.char_span(char_start, char_end)
                if span is not None:
                    assert (span.end - span.start) > 0
                    element = ops.xp.asarray([span.start, span.end])
                    assert len(element) == 2
                    doc_spans.append(element)
                    length += 1
            
            assert length == len(doc_spans)
            lengths.append(length)
            if length == 0:
                raise ValueError("Length is 0")
            spans.extend(ops.xp.array(doc_spans))
        
        # spans = list(set(spans))
        if len(spans) > 0:
            spans = ops.xp.array(spans)
            assert len(spans) == sum(lengths)
            # print(type(spans), len(spans), len(lengths))
            assert spans.ndim == 2
            output = Ragged(spans, ops.asarray(lengths, dtype="i"))
        else:
            output = Ragged(ops.xp.zeros((0, 0)), ops.asarray(lengths, dtype="i"))

        assert output.dataXd.ndim == 2
        return output

    return nounchunk_ngram_suggester



@registry.misc("train_ngram_suggester.v1")
def train_ngram_suggester(sizes: List[int], train_corpus: Path) -> Callable[[List[Doc]], Ragged]:
    """Suggest all spans of the given lengths. Spans are returned as a ragged
    array of integers. The array has two columns, indicating the start and end
    position."""
    
    # Prepare matcher
    nlp = spacy.blank("en") 
    docbin = DocBin().from_disk(train_corpus)
    train_docs = list(docbin.get_docs(nlp.vocab))
    patterns = set()
    for doc in train_docs:
        for ent in doc.ents:
            patterns.add(nlp.make_doc(ent.text))
            
    matcher = PhraseMatcher(nlp.vocab)
    matcher.add("ENT", list(patterns))


    def ngram_suggester(docs: List[Doc], *, ops: Optional[Ops] = None) -> Ragged:
        if ops is None:
            ops = get_current_ops()
        spans = []
        lengths = []

        for doc in docs:
            starts = ops.xp.arange(len(doc), dtype="i")
            starts = starts.reshape((-1, 1))
            length = 0
            for size in sizes:
                if size <= len(doc):
                    starts_size = starts[: len(doc) - (size - 1)]
                    ngrams = ops.xp.hstack((starts_size, starts_size + size))
                    spans.extend([element for element in ngrams])
                    length += len(ngrams)
                # if spans:
                    # assert spans[-1].ndim == 2, spans[-1].shape
            

            matches = matcher(doc, as_spans=True)
            for span in matches:
                element = ops.xp.hstack((span.start, span.end))
                spans.append(element)
                length+=1
            
            lengths.append(length)
        
        if len(spans) > 0:
            spans = ops.xp.asarray(spans)
            output = Ragged(ops.xp.vstack(spans), ops.asarray(lengths, dtype="i"))
        else:
            output = Ragged(ops.xp.zeros((0, 0)), ops.asarray(lengths, dtype="i"))

        assert output.dataXd.ndim == 2
        return output

    return ngram_suggester
