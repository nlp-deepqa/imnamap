import csv
import pickle
import re
from collections import Counter

import numpy as np
import tensorflow as tf
from elasticsearch import Elasticsearch
from imnamap.models.imnamap import build_imnamap_model

from imnamap.memory.searcher import ElasticSearcher
from imnamap.utils.embeddings import embedding_initializer
from imnamap.utils.metrics import evaluate_hits
from imnamap.utils.preprocessing import preprocess_question, ids2tokens, pad_sequences, tokens2ids, parse_model_filename
from imnamap.utils.progress import Progbar

flags = tf.app.flags
flags.DEFINE_string("test", "../datasets/movie_dialog/task1_qa/task1_qa_test.txt.pkl", "Test data filename")
flags.DEFINE_string("index", "../datasets/movie_dialog/task1_qa/index.pkl", "Corpus index data filename")
flags.DEFINE_string("model", "model.ckpt", "Model filename")
flags.DEFINE_string("results", "results.tsv", "Results filename")
flags.DEFINE_string("es_address", "http://localhost:9200", "Elasticsearch server address")
flags.DEFINE_string("es_index", "movie_kb", "Elasticsearch index")
flags.DEFINE_integer("top_results", 1, "Number of results in top-k")
FLAGS = flags.FLAGS

if __name__ == "__main__":
    print("-- Loading test data from {}".format(FLAGS.test))
    with open(FLAGS.test, mode="rb") as in_file:
        test = pickle.load(in_file)

    print("-- Loading index data from {}".format(FLAGS.index))
    with open(FLAGS.index, mode="rb") as in_file:
        index = pickle.load(in_file)

    # Get dataset information
    num_tokens = len(index["token2id"])
    num_examples = len(test["dialogs_questions"])
    max_question_len = test["max_question_len"]
    max_answer_len = test["max_answer_len"]
    es_client = Elasticsearch({FLAGS.es_address})
    searcher = ElasticSearcher(es_client, FLAGS.es_index)
    max_doc_len = index["max_doc_len"]
    model_parameters = parse_model_filename(FLAGS.model)
    batch_size = model_parameters["batch_size"]
    num_batches = num_examples // batch_size
    print("Number of tokens: %d" % num_tokens)
    print("Number of examples: %d" % num_examples)
    print("Maximum question len: %d" % max_question_len)
    print("Maximum answer len: %d" % max_answer_len)
    print("Maximum document len: %d" % max_doc_len)

    print("-- Loading model: {}".format(FLAGS.model))
    with tf.Session() as sess:
        question_input = tf.placeholder(dtype=tf.int32, shape=(None, max_question_len))
        documents_input = tf.placeholder(dtype=tf.int32, shape=(None, None, max_doc_len))
        batch_size_input = tf.placeholder(dtype=tf.int32)
        frequencies_input = tf.placeholder(dtype=tf.float32, shape=(None, num_tokens))
        target_input = tf.placeholder(dtype=tf.float32, shape=(None, num_tokens))
        dropout_gate_input = tf.placeholder(dtype=tf.float32)
        dropout_dense_input = tf.placeholder(dtype=tf.float32)

        net = build_imnamap_model(
            question_input,
            documents_input,
            batch_size_input,
            frequencies_input,
            dropout_gate_input,
            dropout_dense_input,
            num_tokens,
            model_parameters["embedding_size"],
            embedding_initializer(index["token2id"]),
            model_parameters["gru_output_size"],
            model_parameters["inf_gru_output_size"],
            model_parameters["hidden_layer_size"],
            max_doc_len,
            model_parameters["top_docs"],
            model_parameters["num_hops"]
        )

        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.model)
        print("-- Model restored")

        batches_indexes = np.arange(num_examples)
        num_batch = 1

        with open(FLAGS.results, mode="w") as out_file:
            writer = csv.writer(out_file, delimiter="\t")
            progress = Progbar(num_batches)
            hits = np.zeros(num_batches)

            for start_idx in range(0, num_examples - batch_size + 1, batch_size):
                batch_indexes = batches_indexes[start_idx:start_idx + batch_size]
                batch_questions = test["dialogs_questions"][batch_indexes, :]
                batch_answers = test["dialogs_answers"][batch_indexes, :]
                k_hot_answers = np.zeros((batch_size, num_tokens), dtype="float32")
                for i, answer in enumerate(batch_answers):
                    for token_id in answer:
                        if token_id != index["token2id"]["#pad#"]:
                            k_hot_answers[i][token_id] = 1

                batch_docs = []
                for question in batch_questions:
                    question_docs = searcher.search(
                        preprocess_question(
                            [re.escape(token) for token in
                             ids2tokens(question, index["id2token"], index["token2id"]["#pad#"])]),
                        model_parameters["top_docs"]
                    )
                    batch_docs.append(
                        pad_sequences(
                            [tokens2ids(doc, index["token2id"]) for doc in question_docs],
                            maxlen=max_doc_len,
                            padding="post")
                    )
                batch_docs = np.array(
                    [np.pad(doc, [(0, model_parameters["top_docs"] - doc.shape[0]), (0, 0)], "constant")
                     for doc in batch_docs]
                )

                frequencies = np.ones((batch_size, num_tokens), dtype="float32")
                for i, docs in enumerate(batch_docs):
                    counter = Counter([token_id for doc in docs for token_id in doc])
                    for token_id, count in counter.items():
                        frequencies[i, token_id] = count

                predictions = sess.run(
                    tf.sigmoid(net[0]), {
                        question_input: batch_questions,
                        documents_input: batch_docs,
                        batch_size_input: batch_size,
                        frequencies_input: frequencies,
                        dropout_gate_input: 1.0,
                        dropout_dense_input: 1.0
                    }
                )

                top_k_input = tf.placeholder(dtype=tf.float32, shape=(None, num_tokens))
                top_k_values, top_k_indices = sess.run(
                    tf.nn.top_k(predictions, FLAGS.top_results), {
                        top_k_input: predictions
                    }
                )

                for i in range(batch_size):
                    question = ids2tokens(batch_questions[i], index["id2token"], index["token2id"]["#pad#"])
                    predicted_answers = ids2tokens(top_k_indices[i], index["id2token"])
                    correct_answers = ids2tokens(batch_answers[i], index["id2token"], index["token2id"]["#pad#"])
                    writer.writerow([" ".join(question),
                                     " ".join(predicted_answers),
                                     " ".join([str(p) for p in top_k_values[i]]),
                                     " ".join(correct_answers)])

                hits[num_batch - 1] = evaluate_hits(top_k_indices, batch_answers)
                progress.update(num_batch)
                num_batch += 1
            print("Average HITS@{}: {}".format(FLAGS.top_results, np.mean(hits)))
