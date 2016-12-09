import pickle
import sys

from elasticsearch import Elasticsearch
from nltk import word_tokenize

from imnamap.utils.preprocessing import multireplace

predicates = {
    "directed_by",
    "written_by",
    "starred_actors",
    "release_year",
    "in_language",
    "has_tags",
    "has_genre",
    "has_imdb_votes",
    "has_imdb_rating"
}


def build_ids(tokens, token2id, id2token):
    for token in tokens:
        if token not in token2id:
            num_tokens = len(token2id) + 1
            token2id[token] = num_tokens
            id2token[num_tokens] = token


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Invalid number of parameters!")
        sys.exit(-1)

    kb_filename = sys.argv[1]
    kb_entities_filename = sys.argv[2]
    es_server_address = sys.argv[3]
    es_index_name = sys.argv[4]
    kb_index_filename = sys.argv[5]
    kb_entities_replacements_filename = sys.argv[6]

    es = Elasticsearch({es_server_address})
    es.indices.create(index=es_index_name, ignore=400)

    replacements = {}

    with open(kb_entities_filename) as in_file:
        for line in in_file:
            entity = line.strip().lower()
            replacements[entity] = entity.replace(" ", "_")

    with open(kb_filename) as in_file:
        current_title = ""
        current_doc = ""
        token2id = dict()
        id2token = dict()
        entities = dict()
        num_facts = 1
        max_doc_len = 0

        for line in in_file:
            line = line.strip()

            if line:
                current_triple = line.split(" ", 1)[1]

                for predicate in predicates:
                    if predicate in current_triple:
                        current_triple = current_triple.strip().lower()
                        current_triple = multireplace(current_triple, replacements)
                        subject, _ = current_triple.split(predicate, 1)

                        doc = {
                            "title": subject,
                            "text": current_triple
                        }

                        res = es.index(index=es_index_name, doc_type="movie", id=num_facts, body=doc)
                        print("Processing fact: {}".format(num_facts, res["created"]))

                        tokens = [token for token in word_tokenize(doc["text"])]
                        build_ids(tokens, token2id, id2token)
                        max_doc_len = max(max_doc_len, len(tokens))

                num_facts += 1

        print("-- Writing index data")
        with open(kb_index_filename, mode="wb") as out_file:
            pickle.dump({"token2id": token2id, "id2token": id2token, "max_doc_len": max_doc_len}, out_file)

        print("-- Writing entity replacements data")
        with open(kb_entities_replacements_filename, mode="wb") as out_file:
            pickle.dump(replacements, out_file)
