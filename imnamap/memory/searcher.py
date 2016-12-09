from nltk import word_tokenize


class Searcher:
    def search(self, query, topn=10):
        raise NotImplementedError()


class ElasticSearcher(Searcher):
    def __init__(self, es_client, index_name):
        super(ElasticSearcher, self).__init__()
        self._index_name = index_name
        self._es_client = es_client

    def search(self, query_tokens, topn=10):
        results = self._es_client.search(
            index=self._index_name,
            body={
                "size": topn,
                "query": {
                    "query_string": {
                        "query": " ".join(query_tokens),
                    }
                }
            }
        )

        if not results["hits"]["hits"]:
            return []

        return [word_tokenize(hit["_source"]["text"]) for hit in results["hits"]["hits"]]
