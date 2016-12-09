import pickle
import sys

from imnamap.datasets.movie_dialog import load_dialog_data_er, vectorize_dialog_data
from imnamap.utils.preprocessing import pad_sequences

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Invalid number of parameters!")
        sys.exit(-1)

    train_filename = sys.argv[1]
    valid_filename = sys.argv[2]
    test_filename = sys.argv[3]
    kb_index_filename = sys.argv[4]
    index_filename = sys.argv[5]
    entities_filename = sys.argv[6]

    token2id = dict()
    id2token = dict()

    token2id["#pad#"] = 0
    id2token[0] = "#pad#"

    with open(kb_index_filename, mode="rb") as in_file:
        kb_index = pickle.load(in_file)

    for token, token_id in kb_index["token2id"].items():
        if token not in token2id:
            num_tokens = len(token2id)
            token2id[token] = num_tokens
            id2token[num_tokens] = token

    print("-- Loading entities data")
    with open(entities_filename, mode="rb") as in_file:
        entities = pickle.load(in_file)
    print("-- Loading training data")
    train = load_dialog_data_er(train_filename, token2id, id2token, entities)
    print("-- Loading validation data")
    valid = load_dialog_data_er(valid_filename, token2id, id2token, entities)
    print("-- Loading test data")
    test = load_dialog_data_er(test_filename, token2id, id2token, entities)

    vec_train = vectorize_dialog_data(train, token2id)
    vec_valid = vectorize_dialog_data(valid, token2id)
    vec_test = vectorize_dialog_data(test, token2id)

    max_question_len = max([
        vec_train["max_question_len"],
        vec_valid["max_question_len"],
        vec_test["max_question_len"]
    ])
    vec_train["max_question_len"] = max_question_len
    vec_valid["max_question_len"] = max_question_len
    vec_test["max_question_len"] = max_question_len

    max_answer_len = max([
        vec_train["max_answer_len"],
        vec_valid["max_answer_len"],
        vec_test["max_answer_len"]
    ]) + 1
    vec_train["max_answer_len"] = max_answer_len
    vec_valid["max_answer_len"] = max_answer_len
    vec_test["max_answer_len"] = max_answer_len

    print("-- Padding training data")
    vec_train["dialogs_questions"] = pad_sequences(vec_train["dialogs_questions"],
                                                   maxlen=max_question_len,
                                                   padding="post")

    vec_train["dialogs_answers"] = pad_sequences(vec_train["dialogs_answers"],
                                                 maxlen=max_answer_len,
                                                 padding="post")

    print("-- Padding validation data")
    vec_valid["dialogs_questions"] = pad_sequences(vec_valid["dialogs_questions"],
                                                   maxlen=max_question_len,
                                                   padding="post")

    vec_valid["dialogs_answers"] = pad_sequences(vec_valid["dialogs_answers"],
                                                 maxlen=max_answer_len,
                                                 padding="post")

    print("-- Padding test data")
    vec_test["dialogs_questions"] = pad_sequences(vec_test["dialogs_questions"],
                                                  maxlen=max_question_len,
                                                  padding="post")

    vec_test["dialogs_answers"] = pad_sequences(vec_test["dialogs_answers"],
                                                maxlen=max_answer_len,
                                                padding="post")

    print("-- Writing training data")
    with open(train_filename + ".pkl", mode="wb") as out_file:
        pickle.dump(vec_train, out_file)

    print("-- Writing validation data")
    with open(valid_filename + ".pkl", mode="wb") as out_file:
        pickle.dump(vec_valid, out_file)

    print("-- Writing test data")
    with open(test_filename + ".pkl", mode="wb") as out_file:
        pickle.dump(vec_test, out_file)

    print("-- Writing index data")
    with open(index_filename, mode="wb") as out_file:
        pickle.dump({"token2id": token2id, "id2token": id2token, "max_doc_len": kb_index["max_doc_len"]}, out_file)
