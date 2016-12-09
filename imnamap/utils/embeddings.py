import numpy as np
import tensorflow as tf


def load_embeddings(filename, token2id, shape):
    embeddings = np.zeros(shape=shape, dtype=np.float32)
    found_embeddings = set()

    with open(filename) as in_file:
        for line in in_file:
            row = line.split(" ")
            token = row[0]

            if token in token2id:
                embedding = row[1:]
                embeddings[token2id[token]] = np.array(embedding)
                found_embeddings.add(token)

    missing_embeddings = set(token2id.keys()).difference(found_embeddings)

    for token in missing_embeddings:
        embeddings[token2id[token]] = np.random.normal(scale=0.05, size=shape[1])

    return embeddings


def embedding_initializer(token2id, filename=None):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        if filename:
            embeddings = load_embeddings(filename, token2id, shape)
        else:
            embeddings = np.random.normal(scale=0.05, size=shape)

        return embeddings

    return _initializer
