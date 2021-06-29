import spacy
from spacy import registry
from spacy.tokens import Doc, DocBin, Span
from typing import List, Callable, Optional
from thinc.types import Ragged
from thinc.api import Config, Model, get_current_ops, set_dropout_rate, Ops
from pathlib import Path
import numpy as np

@registry.misc("train_ngram_suggester.v1")
def build_ngram_suggester(sizes: List[int], train_corpus: Path) -> Callable[[List[Doc]], Ragged]:
    """Suggest all spans of the given lengths. Spans are returned as a ragged
    array of integers. The array has two columns, indicating the start and end
    position."""

    def ngram_suggester(docs: List[Doc], *, ops: Optional[Ops] = None) -> Ragged:
        if ops is None:
            ops = get_current_ops()
        spans = []
        lengths = []
    
        # I want to load the train docs and get the plain text for all their entities. 
        # Once I've the plain text, I can use a matcher to find the start, end token pair in the target doc
        assert train_corpus.exists()

        nlp = spacy.blank("en") 
        docbin = DocBin().from_disk(train_corpus)
        train_docs = list(docbin.get_docs(nlp.vocab))

        train_ents_vocab = set()
        for doc in train_docs:
            for ent in doc.ents:
                train_ents_vocab.add(ent.text)
                
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

            for ent in train_ents_vocab:
                start = doc.text.find(ent)
                if start == -1:
                    continue
                end = start + len(ent)
                span = doc.char_span(start, end)
                if span is not None:
                    spans.append(ops.xp.hstack((span.start, span.end)))
                    length +=1
            
            lengths.append(length)
        
        # spans = np.array(list(set(spans)))
        if len(spans) > 0:
            output = Ragged(ops.xp.vstack(spans), ops.asarray(lengths, dtype="i"))
        else:
            output = Ragged(ops.xp.zeros((0,0)), ops.asarray(lengths, dtype="i"))

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
            
            # try:
                # assert len(new_doc) == len(doc)
            # except AssertionError as ae:
                # print(f"Found new doc with {len(new_doc)} tokens while blank doc has {len(doc)} tokens.\n The original sentence: {doc.text}")
            
            new_doc = nlp(doc.text)
            for chunk in new_doc.noun_chunks:
                char_start, char_end = chunk.start_char, chunk.end_char
                span = doc.char_span(char_start, char_end)
                if span is not None:
                    # start, end = span.start, span.end
                    spans.append(ops.xp.hstack((span.start, span.end)))
                    length += 1
            lengths.append(length)

        if len(spans) > 0:
            output = Ragged(ops.xp.vstack(spans), ops.asarray(lengths, dtype="i"))
        else:
            output = Ragged(ops.xp.zeros((0,0)), ops.asarray(lengths, dtype="i"))

        assert output.dataXd.ndim == 2
        return output

    return ngram_suggester