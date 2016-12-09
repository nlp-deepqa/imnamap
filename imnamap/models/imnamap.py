import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell


def batch_vm(x, m):
    [input_size, output_size] = m.get_shape().as_list()
    input_shape = tf.shape(x)
    batch_rank = input_shape.get_shape()[0].value - 1
    batch_shape = input_shape[:batch_rank]
    output_shape = tf.concat(0, [batch_shape, [output_size]])
    x = tf.reshape(x, [-1, input_size])
    y = tf.matmul(x, m)
    y = tf.reshape(y, output_shape)
    return y


def build_imnamap_model(question_input,
                        documents_input,
                        batch_size_input,
                        token_counts,
                        dropout_gate_prob,
                        dropout_dense_prob,
                        num_tokens,
                        embedding_size,
                        embedding_initializer,
                        gru_output_size,
                        inf_gru_output_size,
                        hidden_layer_size,
                        max_doc_len,
                        num_docs,
                        num_iana_hops):
    with tf.variable_scope("embeddings"):
        embedding_matrix = tf.get_variable(
            "embedding_matrix",
            shape=(num_tokens, embedding_size),
            initializer=embedding_initializer
        )
    bigru_embedding_size = 2 * gru_output_size

    # Question lookup table
    question_lt = tf.nn.embedding_lookup(embedding_matrix, question_input)

    sequence_length = tf.reduce_sum(
        tf.cast(tf.not_equal(question_input, tf.zeros_like(question_input, dtype=tf.int32)), tf.int64),
        1
    )
    question_bigru_states = tf.nn.bidirectional_dynamic_rnn(
        tf.nn.rnn_cell.GRUCell(gru_output_size),
        tf.nn.rnn_cell.GRUCell(gru_output_size),
        question_lt,
        sequence_length=sequence_length,
        dtype=tf.float32,
        scope="question_bigru"
    )[0]

    # Question BiGRU
    question_bigru = tf.concat(2, question_bigru_states)

    # Document lookup table
    batched_documents_input = tf.reshape(documents_input, (-1, max_doc_len))
    batched_document_lt = tf.nn.embedding_lookup(embedding_matrix, batched_documents_input)

    sequence_length = tf.reduce_sum(
        tf.cast(tf.not_equal(batched_documents_input, tf.zeros_like(batched_documents_input, dtype=tf.int32)),
                tf.int64),
        1
    )

    docs_bigru_states = tf.nn.bidirectional_dynamic_rnn(
        GRUCell(gru_output_size),
        GRUCell(gru_output_size),
        batched_document_lt,
        sequence_length=sequence_length,
        scope="documents_bigru",
        dtype=tf.float32
    )[0]

    # Documents BiGRU
    documents_bigru = tf.reshape(
        tf.concat(2, docs_bigru_states),
        (batch_size_input, num_docs * max_doc_len, bigru_embedding_size)
    )

    def iana_inference_steps(question_bigru, documents_bigru, num_hops):
        # Inference GRU
        hid_init = tf.Variable(tf.random_normal((1, inf_gru_output_size), stddev=0.05))
        inference_state = tf.matmul(tf.ones((batch_size_input, 1)), hid_init)

        # Question attentive reader parameters
        Aq = tf.Variable(tf.random_normal(
            (inf_gru_output_size, bigru_embedding_size),
            stddev=0.05)
        )
        aq = tf.Variable(tf.zeros((bigru_embedding_size,)))

        # Document attentive reader parameters
        Ad = tf.Variable(tf.random_normal(
            (inf_gru_output_size + bigru_embedding_size, bigru_embedding_size),
            stddev=0.05)
        )
        ad = tf.Variable(tf.zeros((bigru_embedding_size,)))

        # Define attentive reader
        def attentive_reader(input, state, weight, bias):
            linear_product = batch_vm(state, weight) + bias
            attention_weights = tf.batch_matmul(input, tf.expand_dims(linear_product, 2))
            attention_weights = tf.nn.softmax(tf.squeeze(attention_weights, [2]))
            tiled_attention_weights = tf.tile(tf.expand_dims(attention_weights, 2), [1, 1, bigru_embedding_size])
            return attention_weights, tf.reduce_sum(tf.mul(input, tiled_attention_weights), 1)

        att_q_weights = []
        att_d_weights = []
        attentive_question = None

        with tf.variable_scope("inference_hop_scope"):

            for num_hop in range(num_hops):
                # Compute attentive question
                curr_att_q_weights, attentive_question = attentive_reader(question_bigru, inference_state, Aq, aq)
                att_q_weights.append(curr_att_q_weights)

                # Compute attentive document
                merged_inf_q = tf.concat(1, [inference_state, attentive_question])
                curr_att_d_weights, attentive_document = attentive_reader(documents_bigru, merged_inf_q, Ad, ad)
                att_d_weights.append(curr_att_d_weights)

                # Compute search gates input
                merged_r_input = tf.concat(1, [inference_state,
                                               attentive_question,
                                               attentive_document,
                                               tf.mul(attentive_question, attentive_document)])

                # Compute question search gate
                with tf.variable_scope("question_gate_scope"):
                    if num_hop > 0:
                        tf.get_variable_scope().reuse_variables()

                    question_gate = tf.nn.dropout(tf.contrib.layers.fully_connected(
                        tf.contrib.layers.fully_connected(
                            merged_r_input,
                            bigru_embedding_size,
                            weights_initializer=tf.random_normal_initializer(stddev=0.05)
                        ),
                        bigru_embedding_size,
                        weights_initializer=tf.random_normal_initializer(stddev=0.05)
                    ), dropout_gate_prob)
                    gated_question = tf.mul(attentive_question, question_gate)

                # Compute document search gate
                with tf.variable_scope("document_gate_scope"):
                    if num_hop > 0:
                        tf.get_variable_scope().reuse_variables()

                    document_gate = tf.nn.dropout(tf.contrib.layers.fully_connected(
                        tf.contrib.layers.fully_connected(
                            merged_r_input,
                            bigru_embedding_size,
                            weights_initializer=tf.random_normal_initializer(stddev=0.05)
                        ),
                        bigru_embedding_size,
                        activation_fn=tf.nn.sigmoid,
                        weights_initializer=tf.random_normal_initializer(stddev=0.05)
                    ), dropout_gate_prob)
                    gated_document = tf.mul(attentive_document, document_gate)

                # Compute inference GRU
                with tf.variable_scope("inference_GRU_scope"):
                    if num_hop > 0:
                        tf.get_variable_scope().reuse_variables()

                    inference_cell = GRUCell(inf_gru_output_size)
                    inference_input = tf.concat(1, [gated_question, gated_document])
                    _, inference_state = inference_cell(inference_input, inference_state)

            return att_q_weights, att_d_weights, attentive_question

    # Document attention weights
    att_q_weights, att_d_weights, attentive_question = iana_inference_steps(
        question_bigru,
        documents_bigru,
        num_iana_hops
    )

    # Reshaped document attention weights
    att_d_weights = [
        tf.reshape(
            tf.reshape(weights, (batch_size_input, num_docs, max_doc_len)),
            (batch_size_input, -1)
        ) for weights in att_d_weights
        ]
    last_att_d_weights = att_d_weights[-1]

    # Reshaped document inputs
    reshaped_document_input = tf.reshape(documents_input, (batch_size_input, -1))
    concat_weights_docs = tf.concat(1, [last_att_d_weights, tf.to_float(reshaped_document_input)])

    def get_tokens_scores(x):
        curr_att_d_weights, curr_docs_input = tf.split(0, 2, x)
        return tf.unsorted_segment_sum(curr_att_d_weights, tf.to_int32(curr_docs_input), num_tokens)

    tokens_scores = tf.map_fn(get_tokens_scores, concat_weights_docs, dtype=tf.float32)
    tokens_scores = tf.div(tokens_scores, token_counts)

    return tf.contrib.layers.fully_connected(
        tf.nn.dropout(tf.contrib.layers.fully_connected(
            tokens_scores,
            hidden_layer_size,
            weights_initializer=tf.random_normal_initializer()
        ), dropout_dense_prob),
        num_tokens,
        activation_fn=None,
        weights_initializer=tf.random_normal_initializer(stddev=0.05)
    ), att_q_weights, att_d_weights, attentive_question
