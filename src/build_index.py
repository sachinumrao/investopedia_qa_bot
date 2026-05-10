import os
import pandas as pd
from pprint import pprint
from tqdm.auto import tqdm

from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def load_documents(filepath):
    print("Loading csv docs...")
    data = CSVLoader(filepath, "Docs")
    return data.load()


def get_document_chunks(data):
    print("Chunking data...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
    chunked_data = text_splitter.split_documents(data)
    return chunked_data


def generate_chroma_index(chunked_docs):
    print("Embedding and storing data in vector database...")
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="investopedia",
        embedding_function=embeddings_model,
        persist_directory="./.db/investopedia_langchain_db"
    )

    # implement manual batching
    num_docs = len(chunked_docs)
    batch_size = 32
    num_steps = int(num_docs // batch_size) + 1

    for i in tqdm(range(num_steps), total=num_steps):
        document_batch = chunked_docs[i : i+batch_size]
        _ = vector_store.add_documents(document_batch)



def join_title_text(row):
    title = row["Title"]
    text = row["Text"]

    final_str = f"Title: {title.strip()} Document: {text.strip()}"
    row["Docs"] = final_str
    return row


def main():

    # parse input file and get data in single column for loader
    # input_data_file = "./data/investopedia_data.csv"
    # df = pd.read_csv(input_data_file)
    # print(df.head())
    # print(df.shape)

    # df = df.apply(join_title_text, axis=1)
    # df = df.drop(["Title", "Text"], axis=1)

    # print(df.head())
    # df.to_csv("./data/investopedia_processed_data.csv", index=False)

    # load documents
    input_data_file = "./data/investopedia_processed_data.csv"
    data = load_documents(input_data_file)
    print("Num docs: ", len(data))

    # split documents in chunks
    data = get_document_chunks(data)
    print("Num chunks: ", len(data))

    # embed and store docs in chroma db index
    generate_chroma_index(data)


if __name__ == "__main__":
    main()
