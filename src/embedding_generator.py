import os

from sentence_transformers import SentenceTransformer

DEBUG = os.environ.get("DEBUG", 0)


class EmbeddingGenerator:
    def __init__(self):
        self.model = None
        self.base_model_name = "all-MiniLM-L6-v2"

    def load_model(self):
        self.model = SentenceTransformer(self.base_model_name)

    def get_embedding(self, text):
        embd = self.model.encode(text)

        if DEBUG == "1":
            print("Input Text: ")
            print(text)
            print("Embedding: ")
            print(embd)
            print("Embedding Shape: ", embd.shape)

        return embd
