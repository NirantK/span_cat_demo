import pytest
import spacy
from spacy.util import get_lang_class

@pytest.fixture(scope="session")
def en_vocab():
    return get_lang_class("en")().vocab


@pytest.fixture(scope="session")
def nlp():
    return spacy.load("en_core_web_sm")

@pytest.fixture(scope="session")
def en_parser(en_vocab):
    nlp = get_lang_class("en")(en_vocab)
    return nlp.create_pipe("parser")


@pytest.fixture(scope="session")
def en_tokenizer():
    return get_lang_class("en")().tokenizer