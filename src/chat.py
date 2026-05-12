import os

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def main():

    # instantiate embedding model
    embedding_model_id = "all-MiniLM-L6-v2"
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_id)

    # load vector store
    vector_store_path = "./.db/investopedia-langchain_db"
    vector_store = Chroma(
        persist_directory=vector_store_path,
        embedding_function=embedding_model
    )

    # load chat model
    llm = ChatOpenAI(
        model="gemma-4-26b-a4b-it-mlx",
        base_url="http://192.168.1.9:1234/v1",
        api_key="dummy",
        temperature=0.1
    )

    prompt = "What is capital of France?"
    response = llm.invoke(prompt)

    print(response.content)
    pass


if __name__ == "__main__":
    main()
