import pytest

from arc25.ngram import ngram_stats

@pytest.mark.parametrize("tokens,n,expected_unique", [
    ([1, 2, 3, 4, 5], 2, 4),
    ([1, 1, 1, 1], 2, 1),
])
def test_unique_ngrams(tokens, n, expected_unique):
    stats = ngram_stats(tokens, n)
    assert stats["unique_ngrams"] == expected_unique

@pytest.mark.parametrize("tokens,n,expected_count", [
    ([1, 2, 3, 4, 5], 2, 1),
    ([1, 1, 1, 1], 2, 3),
])
def test_most_repeated_ngram_count(tokens, n, expected_count):
    stats = ngram_stats(tokens, n)
    assert stats["most_repeated_ngram_count"] == expected_count