
import numpy as np
import tqdm


def generate_dataset(vocab_size, num_sentences, sentence_length_sampler, seed=987654321):
    rng = np.random.default_rng(seed=seed)
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


def stack_based_grammar(pairs, p=0.4):
    s = []
    stack = []
    for i in range(len(pairs)):
        v = np.random.rand()
        if len(stack) == 0 or v < p:
            h, t = pairs[i]
            s.append(h)
            stack.append(t)
        else:
            s.append(stack.pop())
    for i in range(len(stack), 0, -1):
        s.append(stack[i - 1])
    return s


def uniform_sampler(min, max):
    return np.random.random_integers(min, max)
