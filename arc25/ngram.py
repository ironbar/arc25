from collections import Counter
from itertools import islice


def ngram_iter(tokens, n):
    iters = [islice(tokens, i, None) for i in range(n)]
    return zip(*iters)

def ngram_stats(tokens, n):
    counts = Counter(ngram_iter(tokens, n))
    total = max(len(tokens) - n + 1, 0)
    unique = len(counts)
    most_repeated_ngram_count = counts.most_common(1)[0][1] if counts else 0
    return {
        "n": n,
        "total_ngrams": total,
        "unique_ngrams": unique,
        "unique_ngram_ratio": (unique / total) if total else 0.0,
        "most_repeated_ngram_count": most_repeated_ngram_count,
        "most_repeated_ngram_frequency": (most_repeated_ngram_count / total) if total else 0.0,
    }
