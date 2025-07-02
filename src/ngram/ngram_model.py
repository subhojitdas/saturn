from collections import Counter

from nltk.corpus import brown

from nltk.util import ngrams

# nltk.download('punkt')
# nltk.download('brown')

sentences = brown.sents()

tok_sents = [[w.lower() for w in sent] for sent in sentences]
# print(tok_sents)

unigram_counts = Counter()
bigram_counts = Counter()
trigram_counts = Counter()

for sent in tok_sents:
    sent = ['<start>'] + sent + ['<end>']
    unigrams = ngrams(sent, 1)
    bigrams = ngrams(sent, 2)
    trigrams = ngrams(sent, 3)

    unigram_counts.update(unigrams)
    bigram_counts.update(bigrams)
    trigram_counts.update(trigrams)

total_unigrams = sum(unigram_counts.values())
total_bigrams = sum(bigram_counts.values())
total_trigrams = sum(trigram_counts.values())


def backoff_probability(w1, w2, w3, alpha=1.0):
    trigram = (w1, w2, w3)
    bigram = (w2, w3)
    if trigram in trigram_counts:
        context_total = sum(v for k, v in trigram_counts.items() if k[:2] == (w1, w2))
        return trigram_counts[trigram] / context_total if context_total > 0 else 0
    elif bigram in bigram_counts:
        return alpha * (bigram_counts[bigram] / bigram_counts[(w2,)]) if (w2,) in unigram_counts else 0
    else:
        return alpha * (unigram_counts.get((w3,), 0) / total_unigrams)


def interpolated_probability(w1, w2, w3, lambdas=(0.5, 0.3, 0.2)):
    lambda3, lambda2, lambda1 = lambdas
    trigram = (w1, w2, w3)
    bigram = (w2, w3)
    context_trigram = sum(v for k, v in trigram_counts.items() if k[:2] == (w1, w2))
    p_tri = trigram_counts[trigram] / context_trigram if context_trigram > 0 else 0

    context_bigram = unigram_counts.get((w2,), 0)
    p_bi = bigram_counts[bigram] / context_bigram if context_bigram > 0 else 0
    p_uni = unigram_counts.get((w3,), 0) / total_unigrams

    return lambda3 * p_tri + lambda2 * p_bi + lambda1 * p_uni

w1, w2, w3 = "united", "states", "of"

print(f"Back-off estimate P({w3} | {w1}, {w2}):     {backoff_probability(w1, w2, w3):.6f}")
print(f"Interpolated estimate P({w3} | {w1}, {w2}): {interpolated_probability(w1, w2, w3):.6f}")