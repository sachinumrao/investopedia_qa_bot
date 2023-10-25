import os
import sqlite3
import time

from embedding_generator import EmbeddingGenerator
from vector_db import VectorDatabase


class QueryDB:
    def __init__(self, index_name, db_name, table_name):
        self.index_name = index_name
        self.db_name = db_name
        self.table_name = table_name

        # create db connection
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()

        # load index
        self.vec_db = VectorDatabase()
        self.vec_db.load_index_from_disk(self.index_name)

        # load embedding generator
        self.eg = EmbeddingGenerator()
        self.eg.load_model()

    def get_text_from_db(self, idx: int):
        query = f"SELECT uid, chunk_content from {self.table_name} where uid={idx}"
        self.cursor.execute(query)
        data = self.cursor.fetchall()
        return data[0][1]

    def get_query_response(self, query: str, k: int):
        # convert query to embedding
        embd = self.eg.get_embedding(query)

        # get top3 matching results
        neighbors, distances = self.vec_db.index.query(embd, k=3)

        # get text chunk corresponding to neighbors
        resp = {idx: self.get_text_from_db(idx) for idx in neighbors}

        return resp


def main():
    query = "What are the tax benefits of 529 plan?"
    db_name = "./../data/chunk_db.db"
    table_name = "investopedia_chunks"
    index_name = "./../data/investopedia_index1.voy"

    qdb = QueryDB(index_name, db_name, table_name)

    t_start = time.perf_counter()
    response = qdb.get_query_response(query, 3)
    t_stop = time.perf_counter()

    print(response)
    print("Time Taken: ", t_stop - t_start)


if __name__ == "__main__":
    main()
