import fire
from datasets import load_dataset

def download() -> None:
    dataset = load_dataset("conll2003")
    # return dataset

if __name__ == "__main__":
    fire.Fire(download)
