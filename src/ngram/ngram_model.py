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
        return trigram_counts[trigram] / context_total
    elif bigram in bigram_counts:
        return alpha * (bigram_counts[bigram] / bigram_counts[(w2,)]) if (w2,) in unigram_counts else 0
    else:
        return alpha * (unigram_counts.get((w3,), 0) / total_unigrams)

w1, w2, w3 = "can", "i", "tell"

print(f"Back-off estimate P({w3} | {w1}, {w2}):     {backoff_probability(w1, w2, w3):.6f}")