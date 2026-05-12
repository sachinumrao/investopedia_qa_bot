import streamlit as st
from langchain_openai import ChatOpenAI

from chat import InvestopediaRetriever, get_llm_input_prompt

embedding_model_id = "all-MiniLM-L6-v2"
vector_store_path = "./.db/investopedia_langchain_db"
collection_name = "investopedia"
RAG = InvestopediaRetriever(embedding_model_id, vector_store_path, collection_name)

# load chat model
llm_model_id = "mlx-community/qwopus3.5-4b-v3"
lm_studio_url = "http://192.168.1.9:1234/v1"
LLM = ChatOpenAI(model=llm_model_id, base_url=lm_studio_url, api_key="dummy", temperature=0.1, streaming=True)


def run_app():
    st.title("Investopedia Chatbot")

    # init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # take user input
    prompt = st.chat_input()
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            docs = RAG.get_topk_documents(prompt)
            final_prompt = get_llm_input_prompt(prompt, docs)

            stream = LLM.invoke(final_prompt)
            response = st.write_stream(stream)

            # response = response.content
            # st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    pass


if __name__ == "__main__":
    run_app()

# TODO:
# - [] add cache support for vector store and embedding model
# - [] get streaming response working
# - [] add retrieved documents in sidebar
# - [] option to remove messages
# - [] option to save message history
# - [] beautify UI
