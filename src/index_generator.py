import os
import sqlite3
import time

import jsonlines
import numpy as np
from tqdm.auto import tqdm

from embedding_generator import EmbeddingGenerator
from vector_db import VectorDatabase

DB_NAME = "./../data/chunk_db.db"
TABLE_NAME = "investopedia_chunks"
INDEX_NAME = "./../data/investopedia_index1.voy"
N_DIM = 384

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

    # initialize a vector database
    vd = VectorDatabase()
    vd.create_new_index(n_dim=N_DIM)

    # iterate over chunks to store embeddings
    for idx, content in tqdm(chunk_data, total=len(chunk_data)):
        embd = eg.get_embedding(content)
        _ = vd.add_array_to_index(embd)

        if DEBUG == "1":
            print("Shape of embeddings: ", embd.shape)

    # save index
    vd.save_index_on_disk(INDEX_NAME)


def main():
    generate_index()


if __name__ == "__main__":
    main()


## TODO
# implement CRUD wrapper on vouyager index
# iterate over chunk data to generate embeddings
# save embeddings in index
