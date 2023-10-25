import os
import time
import sqlite3

from vector_db import VectorDatabase
from embedding_generator import EmbeddingGenerator


class QueryDB:
    def __init__(self, index_name, db_name, table_name):
        self.index_name = index_name
        self.db_name = db_name
        self.table_name = table_name

        # create db connection
        self.db = None

        # load index
        self.index = None

        # load embedding generator

        pass

    def get_query_response(self, query: str, k: int):
        resp = {}
        return resp


def main():
    query = "What are the tax benefits of 529 plan?"
    qdb = QueryDB()
    response = qdb.get_query_response(query, 3)
    print(response)
