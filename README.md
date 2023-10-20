# investopedia_qa_bot
A RAG based QA bot for finance domain

## Components
- Document Splitter : Split investopedia pages into chunks
- Embedding Generator: Use sentecne transformer to generate embeddings of each chunk
- Index : Use Annoy from spotify for semnatic indexing of chunk embeddings
- Retriever: Fetch chunks from Index based on input query
- LLM : Use llama2 quantized version to read retrieved chunks and respond to user query
- Interface: Build a chat interface with streamlit for ease of use


