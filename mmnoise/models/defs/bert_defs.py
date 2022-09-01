__all__ = [
    'bert_tiny',
]


def bert_tiny():
    return dict(
        hidden_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        max_position_embeddings = 512,
        intermediate_size = 512,
        hidden_act = "gelu",
        initializer_range = 0.02,
        hidden_dropout_prob = 0.1,
        attention_probs_dropout_prob = 0.1,
        type_vocab_size = 2,
    )