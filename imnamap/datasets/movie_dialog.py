import string

from imnamap.utils.preprocessing import preprocess_text, tokens2ids, multireplace


def load_dialog_data(filename, token2id, id2token):
    dialog_id = 0
    dialogs_ids = []
    dialogs_questions = []
    dialogs_answers = []
    num_lines = 1

    def build_ids(tokens):
        for token in tokens:
            if token not in token2id:
                num_tokens = len(token2id) + 1
                token2id[token] = num_tokens
                id2token[num_tokens] = token

    with open(filename) as in_file:
        for line in in_file:
            print("Line number {}".format(num_lines))
            line = line.lower()
            turn_id_question, answer = line.split("\t")
            turn_id, question = turn_id_question.split(" ", 1)

            question_tokens = [token for token in preprocess_text(question, lower=False)]
            answer_tokens = [token for token in preprocess_text(answer, lower=False)]

            build_ids(question_tokens)
            build_ids(answer_tokens)

            if turn_id == "1":
                dialog_id += 1

            dialogs_ids.append(dialog_id)
            dialogs_questions.append(question_tokens)
            dialogs_answers.append(answer_tokens)
            num_lines += 1

    return {
        "dialogs_ids": dialogs_ids,
        "dialogs_questions": dialogs_questions,
        "dialogs_answers": dialogs_answers
    }


def load_dialog_data_er(filename, token2id, id2token, entities):
    dialog_id = 0
    dialogs_ids = []
    dialogs_questions = []
    dialogs_answers = []
    num_lines = 1

    def build_ids(tokens):
        for token in tokens:
            if token not in token2id:
                num_tokens = len(token2id) + 1
                token2id[token] = num_tokens
                id2token[num_tokens] = token

    with open(filename) as in_file:
        for line in in_file:
            line = line.strip()
            if line:
                print("Line number {}".format(num_lines))
                line = line.lower()
                line = multireplace(line, entities)
                turn_id_question, answer = line.split("\t")
                turn_id, question = turn_id_question.split(" ", 1)

                question_tokens = [token for token in preprocess_text(question, lower=False)]
                answer_tokens = [token for token in preprocess_text(answer, lower=False)
                                 if token not in string.punctuation]

                build_ids(question_tokens)
                build_ids(answer_tokens)

                if turn_id == "1":
                    dialog_id += 1

                dialogs_ids.append(dialog_id)
                dialogs_questions.append(question_tokens)
                dialogs_answers.append(answer_tokens)
                num_lines += 1

    return {
        "dialogs_ids": dialogs_ids,
        "dialogs_questions": dialogs_questions,
        "dialogs_answers": dialogs_answers
    }


def vectorize_dialog_data(dialog_data, token2id):
    dialogs_questions = dialog_data["dialogs_questions"]
    dialogs_answers = dialog_data["dialogs_answers"]
    vectorized_questions = []
    vectorized_answers = []
    max_question_len = 0
    max_answer_len = 0

    for i in range(len(dialogs_questions)):
        vectorized_question = tokens2ids(dialogs_questions[i], token2id)
        vectorized_questions.append(vectorized_question)

        vectorized_answer = tokens2ids(dialogs_answers[i], token2id)
        vectorized_answers.append(vectorized_answer)

        max_question_len = max(max_question_len, len(vectorized_question))
        max_answer_len = max(max_answer_len, len(vectorized_answer))

    return {
        "dialogs_ids": dialog_data["dialogs_ids"],
        "dialogs_questions": vectorized_questions,
        "dialogs_answers": vectorized_answers,
        "max_question_len": max_question_len,
        "max_answer_len": max_answer_len
    }
