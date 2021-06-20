"""Convert entity annotation from HuggingFace datasets IOB format to spaCy v3 .spacy format."""
import typer
from datasets import load_dataset

import spacy
from pathlib import Path
from spacy.tokens import DocBin, Span
from typing import List


def download():
    dataset = load_dataset("conll2003")
    return dataset


def retokenize(tokens: List) -> str:
    return " ".join(tokens)


def convert(
    lang: str, output_path: Path, split: str, spans_key: str = "sc", count: int = None
):
    nlp = spacy.blank(lang)
    dataset = load_dataset("conll2003")
    splits = list(dataset.keys())
    assert split in splits
    data = dataset[split]
    tags_key = data.features["ner_tags"].feature.names
    db = DocBin()
    docs = []
    for i, element in enumerate(data):
        if count is not None and i > count:
            break
        sentence = retokenize(element["tokens"])
        doc = nlp(sentence)
        ner_tags = [tags_key[k] for k in element["ner_tags"]]
        char_spans, char_len = [], 0
        substring = ""
        j, prev_begin = 0, False
        for j, tag in enumerate(ner_tags):
            token = element["tokens"][j]
            if "B-" in tag:
                if prev_begin:
                    char_spans.append({"start": start, "end": char_len, "label": label})
                    substring = ""
                substring += token
                label = tag[2:]
                start = char_len
                prev_begin = True
            if "I-" in tag:
                substring += " "
                substring += token
                prev_begin = False
            if "O" == tag and len(substring) > 0:
                char_spans.append({"start": start, "end": char_len, "label": label})
                substring = ""
                prev_begin = False
            char_len += len(token)
            char_len += 1  # for the space

        if len(substring) > 0:  # jugaad for when the last tag is "I-"
            char_spans.append(
                {
                    "start": start,
                    "end": char_len,
                    "label": label,
                }
            )
        for span in char_spans:
            start, end, label = span["start"], span["end"], span["label"]
            new_ent = doc.char_span(start, end - 1, label=label)
            doc.set_ents([new_ent], default="unmodified")

        doc.spans[spans_key] = list(doc.ents)
        docs.append(doc)
        db.add(doc)
    db.to_disk(output_path)


if __name__ == "__main__":
    typer.run(download)
    typer.run(convert)
