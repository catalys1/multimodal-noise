from functools import partial

import numpy as np
import tqdm


def generate_dataset(vocab_size, num_sentences, sentence_length_sampler=None, rng=None):
    '''Generate a "nested parentheses" dataset, according to the algorithm described in

        Ri R, Tsuruoka Y, "Pretraining with Artificial Language: Studying Transferable Knowledge in Language Models"
    
    Args:
        vocab_size (int): number of unique tokens in the dataset
        num_sentences (int): number of sentences to generate
        sentence_length_sampler (Callable): function that returns sentence lengths (number of tokens) sampled from
            some distribution. If None, a uniform distribution over the range (10, 50) is used.
        rng (np.random.Generator): (optional) random number generator, can be used for reproducibility. Note that if
            you want full reproducibility, it must be accounted for in the sentence_length_sampler as well.

    Returns:
        Sentences: a list of sentences, where each sentence is a list of tokens obeying the rules of the nested
            parentheses language.
        Pairs: a list of the token pairs (opening-paren token, closing-paren token)
    '''
    rng = rng or np.random.default_rng()

    if sentence_length_sampler is None:
        sentence_length_sampler = partial(uniform_sampler, 10, 50, rng)  # chosen arbitrarily

    word_vecs = rng.normal(size=(vocab_size // 2, 10))
    pairs = rng.choice(vocab_size, vocab_size, False, shuffle=False).reshape(2, -1).T.tolist()

    sentences = []
    for _ in tqdm.trange(num_sentences):
        cs = rng.normal(size=(10,))
        probs = np.exp(word_vecs @ cs)
        probs /= probs.sum()
        slen = sentence_length_sampler()
        idx = rng.choice(len(pairs), slen // 2, replace=True, p=probs)
        s = [pairs[k] for k in idx]
        s = stack_based_grammar(s)
        sentences.append(s)
    return sentences, pairs


def stack_based_grammar(pairs, rng, p=0.4):
    '''Stack-based grammar for generating nested parentheses sentence from a list of token-pairs.
    '''
    s = []
    stack = []
    for i in range(len(pairs)):
        v = rng.rand()
        if len(stack) == 0 or v < p:
            h, t = pairs[i]
            s.append(h)
            stack.append(t)
        else:
            s.append(stack.pop())
    for i in range(len(stack), 0, -1):
        s.append(stack[i - 1])
    return s


def uniform_sampler(min, max, rng):
    return rng.random_integers(min, max)
