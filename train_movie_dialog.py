import logging
import os
import pickle
import re
import sys
from collections import Counter

import numpy as np
import tensorflow as tf
from elasticsearch import Elasticsearch

from imnamap.memory.searcher import ElasticSearcher
from imnamap.models.imnamap import build_imnamap_model
from imnamap.utils.embeddings import embedding_initializer
from imnamap.utils.metrics import evaluate_hits
from imnamap.utils.preprocessing import preprocess_question, ids2tokens, tokens2ids, pad_sequences
from imnamap.utils.progress import Progbar

flags = tf.app.flags
flags.DEFINE_string("train", "../datasets/movie_dialog/task1_qa/task1_qa_train.txt.pkl", "Training data filename")
flags.DEFINE_string("valid", "../datasets/movie_dialog/task1_qa/task1_qa_dev.txt.pkl", "Validation data filename")
flags.DEFINE_string("index", "../datasets/movie_dialog/task1_qa/index.pkl", "Corpus index data filename")
flags.DEFINE_string("embeddings", None, "Pretrained word embeddings filename")
flags.DEFINE_string("model_dir", "../models/movie_dialog/task1_qa", "Model path")
flags.DEFINE_string("model_name", "iana", "Model name")
flags.DEFINE_string("es_address", "http://localhost:9200", "Elasticsearch server address")
flags.DEFINE_string("es_index", "movie_kb", "Elasticsearch index")
flags.DEFINE_integer("top_docs", 30, "Number of retrieved documents for each question")
flags.DEFINE_integer("num_hops", 3, "Number of hops")
flags.DEFINE_integer("num_epochs", 100, "Number of epochs")
flags.DEFINE_integer("embedding_size", 50, "Word embedding size")
flags.DEFINE_integer("gru_output_size", 128, "Question and documents GRU output size")
flags.DEFINE_integer("inf_gru_output_size", 128, "Inference GRU output size")
flags.DEFINE_integer("hidden_layer_size", 4096, "Last hidden layer size")
flags.DEFINE_string("optim_method", "adam", "Optimization method")
flags.DEFINE_integer("batch_size", 128, "Batch size")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
flags.DEFINE_float("embedding_l2_reg", 0.0001, "Word embedding L2 regularization coefficient")
flags.DEFINE_float("l2_max_norm", 5.0, "Upper bound clipping value for gradients")
flags.DEFINE_float("dropout_gate_prob", 0.2, "Dropout keep probability")
flags.DEFINE_float("dropout_dense_prob", 0.5, "Dropout keep probability")
flags.DEFINE_integer("num_no_improv", 5, "Number of times that no improvements observed on validation set")
flags.DEFINE_integer("top_results", 1, "Cutoff for HITS metrics evaluation")
flags.DEFINE_string("starter_checkpoint", None, "Checkpoint to restore before starting the training")
FLAGS = flags.FLAGS


def get_optimizer(optim_method):
    if optim_method == "adam":
        return tf.train.AdamOptimizer
    else:
        raise ValueError("Invalid optimization method!")


def evaluate_dataset_hits(sess,
                          net,
                          question_input,
                          documents_input,
                          batch_size_input,
                          frequencies_input,
                          dropout_gate_input,
                          dropout_dense_input,
                          dataset,
                          index,
                          batch_size,
                          searcher,
                          top_docs,
                          max_doc_len,
                          top_results):
    num_tokens = len(index["token2id"])
    num_examples = len(dataset["dialogs_questions"])
    batches_indexes = np.arange(num_examples)
    num_batches = num_examples // batch_size
    progress = Progbar(num_batches)
    hits = np.zeros(num_batches)
    num_batch = 1

    for start_idx in range(0, num_examples - batch_size + 1, batch_size):
        batch_indexes = batches_indexes[start_idx:start_idx + batch_size]
        batch_questions = dataset["dialogs_questions"][batch_indexes, :]
        batch_answers = dataset["dialogs_answers"][batch_indexes, :]
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
                top_docs
            )
            batch_docs.append(
                pad_sequences(
                    [tokens2ids(doc, index["token2id"]) for doc in question_docs],
                    maxlen=max_doc_len,
                    padding="post")
            )
        batch_docs = np.array(
            [np.pad(doc, [(0, top_docs - doc.shape[0]), (0, 0)], "constant")
             for doc in batch_docs]
        )

        frequencies = np.ones((batch_size, num_tokens), dtype="float32")
        for i, docs in enumerate(batch_docs):
            counter = Counter([token_id for doc in docs for token_id in doc])
            for token_id, count in counter.items():
                frequencies[i, token_id] = count

        top_k_values, top_k_indices = sess.run(
            tf.nn.top_k(tf.sigmoid(net[0]), top_results), {
                question_input: batch_questions,
                documents_input: batch_docs,
                batch_size_input: batch_size,
                frequencies_input: frequencies,
                dropout_gate_input: 1.0,
                dropout_dense_input: 1.0
            }
        )

        hits[num_batch - 1] = evaluate_hits(top_k_indices, batch_answers)
        progress.update(num_batch)
        num_batch += 1
    return np.mean(hits)


def main(_):
    os.makedirs(os.path.dirname(FLAGS.model_dir), exist_ok=True)
    # Compute model filename
    model_filename = "{model_name}__{top_docs}__{num_hops}__{num_epochs}" \
                     "__{embedding_size}__{gru_output_size}__{inf_gru_output_size}__" \
                     "{hidden_layer_size}__{optim_method}__{batch_size}__{learning_rate}__" \
                     "{embedding_l2_reg}__{l2_max_norm}__{dropout_gate_prob}__{dropout_dense_prob}" \
        .format(**FLAGS.__dict__["__flags"]) \
        .replace(".", "_")
    model_path = os.path.normpath(os.sep.join([FLAGS.model_dir, "{}.ckpt".format(model_filename)]))

    log_formatter = logging.Formatter("%(asctime)s %(message)s")
    root_logger = logging.getLogger(__name__)
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler("{}.log".format(model_path))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    if FLAGS.starter_checkpoint is not None:
        epoch_match = re.match(".*\.e(\d+).*", FLAGS.starter_checkpoint)
        if epoch_match:
            start_epoch = int(epoch_match.group(1))
        else:
            root_logger.fatal("Unable to parse epoch parameter from model file!")
            sys.exit(-1)
    else:
        start_epoch = 1

    root_logger.info("-- Loading training data from {}".format(FLAGS.train))
    with open(FLAGS.train, mode="rb") as in_file:
        train = pickle.load(in_file)

    root_logger.info("-- Loading validation data from {}".format(FLAGS.valid))
    with open(FLAGS.valid, mode="rb") as in_file:
        valid = pickle.load(in_file)

    root_logger.info("-- Loading index data from {}".format(FLAGS.index))
    with open(FLAGS.index, mode="rb") as in_file:
        index = pickle.load(in_file)

    # Get dataset information
    num_tokens = len(index["token2id"])
    num_examples = len(train["dialogs_questions"])
    max_question_len = train["max_question_len"]
    max_answer_len = train["max_answer_len"]
    es_client = Elasticsearch({FLAGS.es_address})
    searcher = ElasticSearcher(es_client, FLAGS.es_index)
    max_doc_len = index["max_doc_len"]
    num_batches = num_examples // FLAGS.batch_size
    root_logger.info("Number of tokens: %d" % num_tokens)
    root_logger.info("Number of examples: %d" % num_examples)
    root_logger.info("Maximum question len: %d" % max_question_len)
    root_logger.info("Maximum answer len: %d" % max_answer_len)
    root_logger.info("Maximum document len: %d" % max_doc_len)

    root_logger.info("-- Building model")
    with tf.Session() as sess:
        tf.set_random_seed(12345)
        np.random.seed(12345)
        question_input = tf.placeholder(dtype=tf.int32, shape=(None, max_question_len))
        documents_input = tf.placeholder(dtype=tf.int32, shape=(None, None, max_doc_len))
        batch_size_input = tf.placeholder(dtype=tf.int32)
        frequencies_input = tf.placeholder(dtype=tf.float32, shape=(None, num_tokens))
        target_input = tf.placeholder(dtype=tf.float32, shape=(None, num_tokens))
        dropout_gate_input = tf.placeholder(dtype=tf.float32)
        dropout_dense_input = tf.placeholder(dtype=tf.float32)

        if FLAGS.embeddings is not None:
            root_logger.info("-- Loading pretrained word embeddings from {}".format(FLAGS.embeddings))
            emb_initializer = embedding_initializer(index["token2id"], FLAGS.embeddings)
        else:
            emb_initializer = embedding_initializer(index["token2id"])

        net = build_imnamap_model(
            question_input,
            documents_input,
            batch_size_input,
            frequencies_input,
            dropout_gate_input,
            dropout_dense_input,
            num_tokens,
            FLAGS.embedding_size,
            emb_initializer,
            FLAGS.gru_output_size,
            FLAGS.inf_gru_output_size,
            FLAGS.hidden_layer_size,
            max_doc_len,
            FLAGS.top_docs,
            FLAGS.num_hops
        )
        loss_function = tf.reduce_mean(tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(net[0], target_input), 1)
        )

        with tf.variable_scope("embeddings", reuse=True):
            loss_function += FLAGS.embedding_l2_reg * tf.nn.l2_loss(tf.get_variable("embedding_matrix"))
        loss_summary_op = tf.scalar_summary("train_loss", loss_function)
        optim = get_optimizer(FLAGS.optim_method)(FLAGS.learning_rate)
        gvs = optim.compute_gradients(loss_function)
        clipped_gvs = [(tf.clip_by_norm(grad, FLAGS.l2_max_norm), var) for grad, var in gvs]
        train_step = optim.apply_gradients(clipped_gvs)
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        if FLAGS.starter_checkpoint:
            root_logger.info("-- Restored starter checkpoint from file {}".format(FLAGS.starter_checkpoint))
            saver.restore(sess, FLAGS.starter_checkpoint)
        summary_writer = tf.train.SummaryWriter(FLAGS.model_dir + "/logs", sess.graph)

        best_hits = 0
        best_epoch = 0
        no_improv_counter = 0
        best_model_path = None

        for e in range(start_epoch, FLAGS.num_epochs + 1):
            root_logger.info("==> online epoch # {0}".format(e))
            progress = Progbar(num_batches)
            batches_indexes = np.arange(num_examples)
            np.random.shuffle(batches_indexes)
            num_batch = 1
            epoch_loss = 0

            for start_idx in range(0, num_examples - FLAGS.batch_size + 1, FLAGS.batch_size):
                batch_indexes = batches_indexes[start_idx:start_idx + FLAGS.batch_size]
                batch_questions = train["dialogs_questions"][batch_indexes, :]
                batch_answers = train["dialogs_answers"][batch_indexes, :]
                k_hot_answers = np.zeros((FLAGS.batch_size, num_tokens), dtype="float32")
                for i, answer in enumerate(batch_answers):
                    for token_id in answer:
                        if token_id != index["token2id"]["#pad#"]:
                            k_hot_answers[i][token_id] = 1

                batch_docs = []
                for question in batch_questions:
                    question_docs = searcher.search(preprocess_question(
                        [re.escape(token) for token in
                         ids2tokens(question, index["id2token"], index["token2id"]["#pad#"])]),
                        FLAGS.top_docs
                    )
                    batch_docs.append(pad_sequences(
                        [tokens2ids(doc, index["token2id"]) for doc in question_docs],
                        maxlen=max_doc_len,
                        padding="post")
                    )
                batch_docs = np.array(
                    [np.pad(doc, [(0, FLAGS.top_docs - doc.shape[0]), (0, 0)], "constant")
                     for doc in batch_docs]
                )

                frequencies = np.ones((FLAGS.batch_size, num_tokens), dtype="float32")
                for i, docs in enumerate(batch_docs):
                    counter = Counter([token_id for doc in docs for token_id in doc])
                    for token_id, count in counter.items():
                        frequencies[i, token_id] = count

                loss, _, loss_summary = sess.run(
                    [loss_function, train_step, loss_summary_op], {
                        question_input: batch_questions,
                        documents_input: batch_docs,
                        batch_size_input: FLAGS.batch_size,
                        target_input: k_hot_answers,
                        frequencies_input: frequencies,
                        dropout_gate_input: FLAGS.dropout_gate_prob,
                        dropout_dense_input: FLAGS.dropout_dense_prob
                    }
                )

                summary_writer.add_summary(loss_summary, global_step=e)
                progress.update(num_batch, [("Loss", loss)])
                epoch_loss += loss
                num_batch += 1
            root_logger.info("Current epoch loss: {}".format(epoch_loss / num_batches))

            root_logger.info("-- Evaluating HITS@1 on validation set")
            current_hits = evaluate_dataset_hits(sess,
                                                 net,
                                                 question_input,
                                                 documents_input,
                                                 batch_size_input,
                                                 frequencies_input,
                                                 dropout_gate_input,
                                                 dropout_dense_input,
                                                 valid,
                                                 index,
                                                 FLAGS.batch_size,
                                                 searcher,
                                                 FLAGS.top_docs,
                                                 max_doc_len,
                                                 FLAGS.top_results)
            root_logger.info("Current HITS@1 on validation set: {}".format(current_hits))
            valid_hits_summary = tf.Summary(value=[tf.Summary.Value(tag="valid_hits", simple_value=current_hits)])
            summary_writer.add_summary(valid_hits_summary, global_step=e)

            if current_hits > best_hits:
                best_hits = current_hits
                best_epoch = e
                no_improv_counter = 0
                if e > 1:
                    os.remove(best_model_path)
                save_path = saver.save(sess, "{}.e{}".format(model_path, e))
                best_model_path = save_path
                root_logger.info("Model saved in file: {}".format(save_path))
            else:
                no_improv_counter += 1

            if no_improv_counter == FLAGS.num_no_improv:
                root_logger.info("-- Terminating training due to early stopping")
                root_logger.info("-- Best HITS@{} {} at epoch {}".format(FLAGS.top_results, best_hits, best_epoch))
                exit(-1)


if __name__ == "__main__":
    tf.app.run()
