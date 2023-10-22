"""
This code reads from the dump of investopedia pages and creates chunks of data.
chunks can be a jsonl file with chunks containing following attributes:
{
    doc_id: doc id of investopedia page
    chunk_id: denoting i th chunk from the given doc id
    content: text chunk from document
}
"""

import json
import os
import time
from pathlib import Path

import jsonlines
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm.auto import tqdm

RAW_FILES_PATH = Path("~/Data/Investopedia/investopedia.csv")
OUTPUT_FILEPATH = Path("./../data/investopedia_chunks.jsonl")

DEBUG = os.environ.get("DEBUG", 0)


def read_data():
    data_df = pd.read_csv(RAW_FILES_PATH)
    if DEBUG == "1":
        print("Data: ")
        print(data_df.head())
        print("Shape of data: ", data_df.shape)
        print("Columns in data: ", data_df.columns)

    return data_df


def chunkify(data_df: pd.DataFrame):
    max_word_count = 350
    chunks = []
    print("Chunking data...")
    for _, row in tqdm(data_df.iterrows(), total=data_df.shape[0]):
        text = row["Text"]
        doc_id = row["DocID"]
        # get list of sentecnes
        sents = sent_tokenize(text)

        # aggregate sentecnes into chunks of size ~350
        counter = 0

        len_counter = 0
        curr_sents = []

        for s in sents:
            n = len(word_tokenize(s))
            if len_counter + n < max_word_count:
                curr_sents.append(s)
                len_counter += n

            else:
                # append chunk
                chunks.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": counter,
                        "chunk_content": " ".join(curr_sents),
                    }
                )
                counter += 1

                # reset local counters
                curr_sents = []
                len_counter = 0

                # add current sentecne to new chunk segment
                curr_sents.append(s)
                len_counter == n

    return chunks


def main():
    # read data
    data_df = read_data()

    # clean data
    # print some samples to assess cleaning requirements
    if DEBUG == "2":
        tmp_df = data_df.sample(20)
        for _, row in tmp_df.iterrows():
            print("----------Sample----------")
            print(row["Text"])
            print("\n\n")

    # chunkfiy data
    chunks = chunkify(data_df)

    # save chunks
    with jsonlines.open(OUTPUT_FILEPATH, mode="w") as writer:
        for c in chunks:
            writer.write(c)


if __name__ == "__main__":
    main()


## TODO
# add support for command line args for input and output files, overlap window
# experiment with spacy for sentence tokenization
# support overlap window in chunks
