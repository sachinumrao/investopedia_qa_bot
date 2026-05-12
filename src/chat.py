import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

PROMPT_TEMPLATE = """You are a helpful financial education providing chatbot. Your task is to answer user query in factual, clear and concise manner. Your response needs to be grounded on documents provided below:"""


def get_llm_input_prompt(query, retrieved_docs: list[str]) -> str:
    document_str = ""
    for idx, doc in enumerate(retrieved_docs):
        document_str += f"\nDocument-{idx}\n{doc}"

    query_str = f"\nQuery: {query}"

    llm_prompt = PROMPT_TEMPLATE + document_str + query_str + "\nResponse: "
    return llm_prompt


class InvestopediaRetriever:
    def __init__(self, embedding_model_id, vector_store_path, collection_name):
        print("Initializing embedding model for RAG...")
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_id)

        print("Loading vector database...")
        self.vector_store = Chroma(
            persist_directory=vector_store_path,
            embedding_function=self.embedding_model,
            collection_name=collection_name,
        )

    def get_topk_documents(self, query: str) -> list[str]:
        print("Getting similar documents...")
        docs = self.vector_store.similarity_search(query, k=5)

        docs_content = [doc.page_content for doc in docs]
        return docs_content


def main():
    # instantiate RAG
    embedding_model_id = "all-MiniLM-L6-v2"
    vector_store_path = "./.db/investopedia_langchain_db"
    collection_name = "investopedia"
    rag = InvestopediaRetriever(embedding_model_id, vector_store_path, collection_name)

    # load chat model
    llm_model_id = "mlx-community/qwopus3.5-4b-v3"
    lm_studio_url = "http://192.168.1.9:1234/v1"
    llm = ChatOpenAI(model=llm_model_id, base_url=lm_studio_url, api_key="dummy", temperature=0.1)

    query = "What is mutual fund?"

    docs = rag.get_topk_documents(query)
    prompt = get_llm_input_prompt(query, docs)
    response = llm.invoke(prompt)

    print(response.content)


if __name__ == "__main__":
    main()
