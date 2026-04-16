

class BaseSearcher:
    def __init__(self):
        pass

    def search(self, query: str, k: int = 10):
        raise NotImplementedError

    def get_doc(self, _id):
        raise NotImplementedError

    def get_doc_vec(self, _id):
        raise NotImplementedError