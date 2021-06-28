from spacy import registry
from spacy.tokens import Doc
from typing import List, Callable, Optional
from thinc.types import Ragged
from thinc.api import Config, Model, get_current_ops, set_dropout_rate, Ops


@registry.misc("ngram_suggester.v2")
def build_ngram_suggester(sizes: List[int]) -> Callable[[List[Doc]], Ragged]:
    """Suggest all spans of the given lengths. Spans are returned as a ragged
    array of integers. The array has two columns, indicating the start and end
    position."""

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
                    starts_size = starts[:len(doc) - (size - 1)]
                    spans.append(ops.xp.hstack((starts_size, starts_size + size)))
                    length += spans[-1].shape[0]
                if spans:
                    assert spans[-1].ndim == 2, spans[-1].shape
            lengths.append(length)

        if len(spans) > 0:
            output = Ragged(ops.xp.vstack(spans), ops.asarray(lengths, dtype="i"))
        else:
            output = Ragged(ops.xp.zeros((0,0)), ops.asarray(lengths, dtype="i"))

        assert output.dataXd.ndim == 2
        return output

    return ngram_suggester