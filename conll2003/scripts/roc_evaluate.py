from pathlib import Path
from thinc.model import serialize_attr
import typer
import spacy
from spacy.tokens import DocBin
import json
from typing import Optional, List, Dict, Tuple
from wasabi import Printer
from pathlib import Path
import re
import srsly
from thinc.api import fix_random_seed
from thinc.types import Ragged, Ints1d
from spacy.training import Corpus
from spacy.tokens import Doc
# from spacy._util import app, Arg, Opt, setup_gpu, import_code
from spacy.scorer import Scorer
from spacy import util
from spacy import displacy
import numpy as np
import json

from thinc.api import to_numpy

def _ensure_cpu(spans: Ragged, lengths: Ints1d) -> Tuple[np.ndarray]:
    return to_numpy(spans.dataXd), to_numpy(spans.lengths), to_numpy(lengths)

def evaluate(
    model: str,
    data_path: Path,
    output: Optional[Path],
    use_gpu: int = -1,
    gold_preproc: bool = False,
    displacy_path: Optional[Path] = None,
    displacy_limit: int = 25,
    silent: bool = True,
    spans_key="sc",
) -> Scorer:
    msg = Printer(no_print=silent, pretty=not silent)
    fix_random_seed()
    data_path = util.ensure_path(data_path)
    output_path = util.ensure_path(output)
    displacy_path = util.ensure_path(displacy_path)
    if not data_path.exists():
        msg.fail("Evaluation data not found", data_path, exits=1)
    if displacy_path and not displacy_path.exists():
        msg.fail("Visualization output directory not found", displacy_path, exits=1)
    corpus = Corpus(data_path, gold_preproc=gold_preproc)
    nlp = util.load_model(model)
    dev_dataset = list(corpus(nlp))
    scores = nlp.evaluate(dev_dataset)
    data = {}
    pred_indices, pred_scores = scores["indices"], scores["scores"]
    pred_spans, pred_span_lengths, pred_scores = _ensure_cpu(pred_indices, pred_scores)
    pred_spans = pred_spans.tolist()
    pred_span_lengths = pred_span_lengths.tolist()
    pred_scores = pred_scores.tolist()
    data["spans"] = str(pred_spans)
    data["span_lengths"] = str(pred_span_lengths)
    data["scores"] = str(pred_scores)
    
    if "morph_per_feat" in scores:
        if scores["morph_per_feat"]:
            print_prf_per_type(msg, scores["morph_per_feat"], "MORPH", "feat")
            data["morph_per_feat"] = scores["morph_per_feat"]
    if "dep_las_per_type" in scores:
        if scores["dep_las_per_type"]:
            print_prf_per_type(msg, scores["dep_las_per_type"], "LAS", "type")
            data["dep_las_per_type"] = scores["dep_las_per_type"]
    if "ents_per_type" in scores:
        if scores["ents_per_type"]:
            print_prf_per_type(msg, scores["ents_per_type"], "NER", "type")
            data["ents_per_type"] = scores["ents_per_type"]
    if f"spans_{spans_key}_per_type" in scores:
        if scores[f"spans_{spans_key}_per_type"]:
            print_prf_per_type(msg, scores[f"spans_{spans_key}_per_type"], "SPANS", "type")
            data[f"spans_{spans_key}_per_type"] = scores[f"spans_{spans_key}_per_type"]
    if "cats_f_per_type" in scores:
        if scores["cats_f_per_type"]:
            print_prf_per_type(msg, scores["cats_f_per_type"], "Textcat F", "label")
            data["cats_f_per_type"] = scores["cats_f_per_type"]
    if "cats_auc_per_type" in scores:
        if scores["cats_auc_per_type"]:
            print_textcats_auc_per_cat(msg, scores["cats_auc_per_type"])
            data["cats_auc_per_type"] = scores["cats_auc_per_type"]
    
    if output_path is not None:
        srsly.write_json(output_path, data)
        # srsly.write_json(output_path, predictions)
        msg.good(f"Saved results to {output_path}")
    return data


def render_parses(
    docs: List[Doc],
    output_path: Path,
    model_name: str = "",
    limit: int = 250,
    deps: bool = True,
    ents: bool = True,
):
    docs[0].user_data["title"] = model_name
    if ents:
        html = displacy.render(docs[:limit], style="ent", page=True)
        with (output_path / "entities.html").open("w", encoding="utf8") as file_:
            file_.write(html)
    if deps:
        html = displacy.render(
            docs[:limit], style="dep", page=True, options={"compact": True}
        )
        with (output_path / "parses.html").open("w", encoding="utf8") as file_:
            file_.write(html)


def print_prf_per_type(
    msg: Printer, scores: Dict[str, Dict[str, float]], name: str, type: str
) -> None:
    data = []
    for key, value in scores.items():
        row = [key]
        for k in ("p", "r", "f"):
            v = value[k]
            row.append(f"{v * 100:.2f}" if isinstance(v, (int, float)) else v)
        data.append(row)
    msg.table(
        data,
        header=("", "P", "R", "F"),
        aligns=("l", "r", "r", "r"),
        title=f"{name} (per {type})",
    )


def print_textcats_auc_per_cat(
    msg: Printer, scores: Dict[str, Dict[str, float]]
) -> None:
    msg.table(
        [
            (k, f"{v:.2f}" if isinstance(v, (float, int)) else v)
            for k, v in scores.items()
        ],
        header=("", "ROC AUC"),
        aligns=("l", "r"),
        title="Textcat ROC AUC (per label)",
    )


# def main(loc: Path, lang: str, spans_key: str):
#     """
#     Set the NER data into the doc.spans, under a given key.
#     The SpanCategorizer component uses the doc.spans, so that it can work with
#     overlapping or nested annotations, which can't be represented on the
#     per-token level.
#     """
#     nlp = spacy.blank(lang)
#     docbin = DocBin().from_disk(loc)
#     docs = list(docbin.get_docs(nlp.vocab))
#     for doc in docs:
#         doc.spans[spans_key] = list(doc.ents)
#     DocBin(docs=docs).to_disk(loc)


if __name__ == "__main__":
    typer.run(evaluate)