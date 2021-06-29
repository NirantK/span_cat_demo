from pathlib import Path
import typer
import spacy
from spacy.tokens import DocBin


def main(loc: Path, lang: str, spans_key: str):
    """
    Set the NER data into the doc.spans, under a given key.
    The SpanCategorizer component uses the doc.spans, so that it can work with
    overlapping or nested annotations, which can't be represented on the
    per-token level.
    """
    nlp = spacy.blank("en")
    docbin = DocBin().from_disk(loc)
    docs = list(docbin.get_docs(nlp.vocab))
    for doc in docs:
        doc.spans[spans_key] = list(doc.ents)
    DocBin(docs=docs).to_disk(loc)


if __name__ == "__main__":
    typer.run(main)