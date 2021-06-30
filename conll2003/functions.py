from matplotlib.pyplot import axis
import spacy
from spacy import registry
from spacy.tokens import Doc, DocBin, Span
from spacy.matcher import PhraseMatcher
from typing import List, Callable, Optional
from thinc.types import Ragged
from thinc.api import Config, Model, get_current_ops, set_dropout_rate, Ops, to_numpy
from pathlib import Path
import numpy as np

@registry.misc("train_ngram_suggester.v1")
def build_ngram_suggester(sizes: List[int], train_corpus: Path) -> Callable[[List[Doc]], Ragged]:
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
                    spans.append(ops.xp.hstack((starts_size, starts_size + size)))
                    length += spans[-1].shape[0]
                if spans:
                    assert spans[-1].ndim == 2, spans[-1].shape
            lengths.append(length)

        flatten = lambda t: [item for sublist in t for item in sublist]

        spans = flatten(spans)
        # print(len(spans), spans[-1], spans[-1].shape, type(spans[-1]))

        for idx, doc in enumerate(docs):
            matches = matcher(doc, as_spans=True)
            for span in matches:
                element = ops.xp.hstack((span.start, span.end))
                # print(type(element), element.shape)
                spans.append(element)
                lengths[idx]+=1

        if len(spans) > 0:
            output = Ragged(ops.xp.vstack(spans), ops.asarray(lengths, dtype="i"))
        else:
            output = Ragged(ops.xp.zeros((0, 0)), ops.asarray(lengths, dtype="i"))

        assert output.dataXd.ndim == 2
        return output

    return ngram_suggester

@registry.misc("nounchunk_ngram_suggester.v1")
def build_ngram_suggester(sizes: List[int], train_corpus: Path) -> Callable[[List[Doc]], Ragged]:
    """Suggest all spans of the given lengths. Spans are returned as a ragged
    array of integers. The array has two columns, indicating the start and end
    position."""

    def ngram_suggester(docs: List[Doc], *, ops: Optional[Ops] = None) -> Ragged:
        if ops is None:
            ops = get_current_ops()
        spans, noun_spans = [], []
        nlp = spacy.load("en_core_web_sm")
        lengths, noun_lengths = [], []
        for doc in docs:
            # starts = ops.xp.arange(len(doc), dtype="i")
            # starts = starts.reshape((-1, 1))
            # length = 0
            # for size in sizes:
            #     if size <= len(doc):
            #         starts_size = starts[:len(doc) - (size - 1)]
            #         spans.append(ops.xp.hstack((starts_size, starts_size + size)))
            #         length += spans[-1].shape[0]
            #     if spans:
            #         assert spans[-1].ndim == 2, spans[-1].shape
            
            new_doc = nlp(doc.text)
            noun_length = 0
            for chunk in new_doc.noun_chunks:
                char_start, char_end = chunk.start_char, chunk.end_char
                span = doc.char_span(char_start, char_end)
                if span is not None:
                    # start, end = span.start, span.end
                    noun_spans.append([span.start, span.end])
                    noun_length += 1
            noun_lengths.append(noun_length)
            # lengths.append(length)

        if len(spans) > 0:
            # element = ops.xp.vstack(spans)
            output = Ragged(ops.xp.vstack(noun_spans), ops.asarray(noun_lengths, dtype="i"))
        else:
            output = Ragged(ops.xp.zeros((0,0)), ops.asarray(lengths, dtype="i"))

        assert output.dataXd.ndim == 2
        return output

    return ngram_suggester