import fire
from pathlib import Path
import json

from datasets import load_dataset


def convert_to_BIO(data, output_path):
    ner_tags = data.features["ner_tags"].feature.names
    new_data = []
    for i, element in enumerate(len(data)):
        d = data[i]
        text = []
        entities = []
        for token, tag in zip(d["tokens"], d["ner_tags"]):
            text.append(token)
            if not (ner_tags[tag]=='O'):
                entities.append((token, ner_tags[tag]))
        new_data.append([
                         " ".join(text),
                         entities
        ])
        output = []
        for d in new_data:
            ent_map = []
            for ent in d[1]:
                text = d[0]
                token = ent[0]
                label = ent[1]
                start = text.find(token)
                end = start + len(token)
                ent_map.append(
                    [start, 
                    end, 
                    label]
                )
            output.append(
                [
                 text,
                 {
                     "entities" : ent_map
                 }
                ]
            )
    json.dump(output, output_path.open("w"), indent=2)


def download(output_path):
    dataset = load_dataset("conll2003")
    data_dir = Path("./assets")
    name = output_path[:-5]
    if name=="dev":
        name = "validation"
    output_path = data_dir/output_path
    data = dataset[name]
    d = convert_to_BIO(data, output_path)



if __name__ == '__main__':
  fire.Fire(download)