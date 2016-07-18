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

@pytest.mark.parametrize("from_word,to_word", [
    ("test","ttest"),
    ("test","tst"),
    ("test","tset"),
    ("test","trst"),
])
def test_edit_functions_reproduce_insertion(from_word,to_word):
    edit, index = errors.get_edit_function(from_word, to_word)
    assert edit.function(from_word, index) == to_word

def test_edit_functions_near_end():
    from_word, to_word = "mandate","mandatey"
    edit, index = errors.get_edit_function(from_word, to_word)
    assert edit.function(from_word, index) == to_word
    assert edit.name == "--->y"

def test_edit_functions_transpose_at_end():
    from_word, to_word = "mandate","mandaet"
    edit, index = errors.get_edit_function(from_word, to_word)
    assert edit.function(from_word, index) == to_word
    assert edit.name == "t<->e"

def test_edit_functions_transpose_at_start():
    from_word, to_word = "mandate","amndate"
    edit, index = errors.get_edit_function(from_word, to_word)
    assert edit.function(from_word, index) == to_word
    assert edit.name == "m<->a"
