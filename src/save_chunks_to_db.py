import os
import sqlite3
import time

import jsonlines

DEBUG = os.environ.get("DEBUG", "0")

CHUNK_FILEPATH = "./../data/investopedia_chunks.jsonl"
DB_NAME = "./../data/chunk_db.db"
TABLE_NAME = "investopedia_chunks"


def read_chunks_data(chunk_file):
    print("Reading chunks from disk...")
    chunks = []
    with jsonlines.open(CHUNK_FILEPATH, mode="r") as reader:
        for item in reader:
            chunks.append(item)

    if DEBUG == "1":
        print("Number of Chunks: ", len(chunks))
        for i in range(5):
            print(chunks[i])
            print("-" * 120)
    return chunks


def save_chunks_to_db():
    # read chunks data
    chunks = read_chunks_data(CHUNK_FILEPATH)

    # create a sqlite database
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        f"""CREATE TABLE {TABLE_NAME}
        (uid real, doc_id real, chunk_id real, chunk_content text)
        """
    )

    # save chunks to sqlite database
    for idx, item in enumerate(chunks):
        cursor.execute(
            f"INSERT INTO {TABLE_NAME} VALUES (?, ?, ?, ?)",
            (idx, item["doc_id"], item["chunk_id"], item["chunk_content"]),
        )

    # close db connection
    conn.commit()
    conn.close()


def main():
    t1 = time.perf_counter()
    save_chunks_to_db()
    t2 = time.perf_counter()
    print("Time Taken: ", t2 - t1)


if __name__ == "__main__":
    main()
