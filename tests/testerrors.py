from spelling import errors
import pytest

def test_ngram_generator():
    text = "123"
    expected = ["--1","-12","123"]
    for expectation, truth in zip(expected, errors.ngram_generator(text, 3)):
        assert expectation == truth

def test_tokenize():
    text = "This isn't a sentence!"
    expected = ["this", "isn't", "a", "sentence"]
    for expectation, truth in zip(expected, errors.tokenize(text)):
        assert expectation == truth

@pytest.mark.parametrize("correct,incorrect", [
    ("test","ttest"),
    ("test","tst"),
    ("test","tset"),
    ("test","trst"),
])
def test_edit_functions_reproduce_insertion(correct,incorrect):
    edit, index = errors.get_edit_function(incorrect, correct)
    assert edit(correct, index) == incorrect
