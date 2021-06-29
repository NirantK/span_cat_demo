import spacy
from spacy import registry
from spacy.tokens import Doc, DocBin
from typing import List, Callable, Optional
from thinc.types import Ragged
from thinc.api import Config, Model, get_current_ops, set_dropout_rate, Ops
from pathlib import Path
import numpy as np

@registry.misc("ngram_suggester.v2")
def build_ngram_suggester(sizes: List[int]) -> Callable[[List[Doc]], Ragged]:
    """Suggest all spans of the given lengths. Spans are returned as a ragged
    array of integers. The array has two columns, indicating the start and end
    position."""

    def ngram_suggester(docs: List[Doc], *, ops: Optional[Ops] = None) -> Ragged:
        if ops is None:
            ops = get_current_ops()
        spans = []
        nlp = spacy.load("en_core_web_sm")
        lengths = []
        for doc in docs:
            starts = ops.xp.arange(len(doc), dtype="i")
            starts = starts.reshape((-1, 1))
            length = 0
            for size in sizes:
                if size <= len(doc):
                    starts_size = starts[:len(doc) - (size - 1)]
                    spans.append(ops.xp.hstack((starts_size, starts_size + size)))
                    length += spans[-1].shape[0]
                if spans:
                    assert spans[-1].ndim == 2, spans[-1].shape
            
            new_doc = nlp(doc.text)
            try:
                assert len(new_doc) == len(doc)
            except AssertionError as ae:
                raise AssertionError(f"Found new doc with {len(new_doc)} tokens while blank doc has {len(doc)} tokens.\n The original sentence: {doc.text}")
            
            for chunk in doc.noun_chunks:
                start, end = chunk.start, chunk.end
                spans.append(ops.xp.hstack((start, end)))
                length += 1
            
            lengths.append(length)

        if len(spans) > 0:
            output = Ragged(ops.xp.vstack(spans), ops.asarray(lengths, dtype="i"))
        else:
            output = Ragged(ops.xp.zeros((0,0)), ops.asarray(lengths, dtype="i"))

        assert output.dataXd.ndim == 2
        return output

    return ngram_suggester