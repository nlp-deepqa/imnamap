import os
import re
import string

import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords


# Function taken from Keras library (https://github.com/fchollet/keras/blob/master/keras/preprocessing/sequence.py)
def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check "trunc" has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def preprocess_text(text, language="english", lower=True):
    return [token.lower() if lower else token for token in word_tokenize(text, language)]


def preprocess_question(query, language="english", lower=True):
    return [token.lower() if lower else token for token in query
            if token not in stopwords.words(language) and token not in string.punctuation]


def tokens2ids(tokens, token2id):
    return [token2id[token] for token in tokens]


def ids2tokens(ids, id2token, padding_value=None):
    if padding_value is not None:
        return [id2token[id] for id in ids if id != padding_value]
    return [id2token[id] for id in ids]


def multireplace(string, replacements):
    # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
    # 'hey ABC' and not 'hey ABc'
    substrs = sorted(replacements, key=len, reverse=True)

    # Create a big OR regex that matches any of the substrings to replace
    regexp = re.compile('|'.join(map(re.escape, substrs)))

    # For each match, look up the new string in the replacements
    return regexp.sub(lambda match: replacements[match.group(0)], string)


def parse_model_filename(filename):
    """
    Model filename is a list of the following parameters separated by "__":
        model_name
        top_docs
        num_hops
        num_epochs
        embedding_size
        gru_output_size
        inf_gru_output_size
        hidden_layer_size
        optim_method
        batch_size
        learning_rate (with decimal separator replaced by "_")
        embedding_l2_reg (with decimal separator replaced by "_")
        l2_max_norm (with decimal separator replaced by "_")
        dropout_gate_prob (with decimal separator replaced by "_")
        dropout_dense_prob (with decimal separator replaced by "_")
    """
    regex = "(?P<model_name>[a-zA-Z0-9]+)__" \
            "(?P<top_docs>\d+)__" \
            "(?P<num_hops>\d+)__" \
            "(?P<num_epochs>\d+)__" \
            "(?P<embedding_size>\d+)__" \
            "(?P<gru_output_size>\d+)__" \
            "(?P<inf_gru_output_size>\d+)__" \
            "(?P<hidden_layer_size>\d+)__" \
            "(?P<optim_method>[a-zA-Z0-9]+)__" \
            "(?P<batch_size>\d+)__" \
            "(?P<learning_rate>\d+_\d+)__" \
            "(?P<embedding_l2_reg>\d+_\d+)__" \
            "(?P<l2_max_norm>\d+_\d+)__" \
            "(?P<dropout_gate_prob>\d+_\d+)__" \
            "(?P<dropout_dense_prob>\d+_\d+).*"
    matches = re.match(regex, os.path.basename(filename))
    if matches:
        matches_dict = matches.groupdict()
        matches_dict["top_docs"] = int(matches_dict["top_docs"])
        matches_dict["num_hops"] = int(matches_dict["num_hops"])
        matches_dict["num_epochs"] = int(matches_dict["num_epochs"])
        matches_dict["embedding_size"] = int(matches_dict["embedding_size"])
        matches_dict["gru_output_size"] = int(matches_dict["gru_output_size"])
        matches_dict["inf_gru_output_size"] = int(matches_dict["inf_gru_output_size"])
        matches_dict["hidden_layer_size"] = int(matches_dict["hidden_layer_size"])
        matches_dict["batch_size"] = int(matches_dict["batch_size"])
        matches_dict["learning_rate"] = float(matches_dict["learning_rate"].replace("_", "."))
        matches_dict["embedding_l2_reg"] = float(matches_dict["embedding_l2_reg"].replace("_", "."))
        matches_dict["l2_max_norm"] = float(matches_dict["l2_max_norm"].replace("_", "."))
        matches_dict["dropout_gate_prob"] = float(matches_dict["dropout_gate_prob"].replace("_", "."))
        matches_dict["dropout_dense_prob"] = float(matches_dict["dropout_dense_prob"].replace("_", "."))

        return matches_dict
    else:
        raise ValueError("Invalid model filename!")
