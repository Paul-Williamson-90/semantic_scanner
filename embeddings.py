

class Embeddings:

    def __init__(self,):
        pass

    def embed(self, chunks:list):
        """
        Args:
        - chunks (list): List of chunks to embed
        """
        pass

class HFEmbeddings(Embeddings):

    def __init__(self,):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def embed(self, chunks:list):
        """
        Args:
        - chunks (list): List of chunks to embed
        """
        return self.model.encode(chunks)