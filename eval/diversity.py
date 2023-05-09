from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import nltk


def diversity_score(sents):
    candidate_toks = list(map(lambda x: [y.lower() for y in nltk.word_tokenize(x) if len(y.strip()) > 0], sents))
    diversities = []
    for i in range(len(sents)):
        references = [c for j, c in enumerate(candidate_toks) if j != i]
        self_bleu = sentence_bleu(references, hypothesis=candidate_toks[i])
        diversities.append(1 - self_bleu)
    return np.mean(diversities)
