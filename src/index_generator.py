import os
import time
import sqlite3

import jsonlines
import numpy as np

from embedding_generator import EmbeddingGenerator
from vector_db import VectorDatabase

DB_NAME = "./../data/chunk_db.db"
TABLE_NAME = "investopedia_chunks"
INDEX_NAME = "./../data/investopedia_index1.voy"

DEBUG = os.environ.get("DEBUG", "0")


def generate_index():
    # read jsonlines data
    print("Reading chunks from database...")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    query = f"SELECT uid, chunk_content from {TABLE_NAME}"
    cursor.execute(query)

    chunk_data = cursor.fetchall()

    if DEBUG == "1":
        print("Rows Received: ", len(chunk_data))
        for i in range(5):
            print(chunk_data[i])

    # convert chunks to embeddings
    print("Loading the embedding model...")
    eg = EmbeddingGenerator()
    eg.load_model()

    # create a vector database
    vd = VectorDatabase(INDEX_NAME)

    pass


def main():
    generate_index()


if __name__ == "__main__":
    main()
