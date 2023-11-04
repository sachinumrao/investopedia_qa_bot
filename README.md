# investopedia_qa_bot
A RAG based QA bot for finance domain

## Components
- [x] Document Splitter : Split investopedia pages into chunks
    - [ ] Add support for overlapping chunk creation
- [x] Embedding Generator: Use sentecne transformer to generate embeddings of each chunk
- [x] Index : Use Annoy from spotify for semnatic indexing of chunk embeddings
- [x] Retriever: Fetch chunks from Index based on input query
- [ ] Keywords Based Retriever: Tag each chunk with keywords and use BM25 for retrieving documents
- [ ] LLM : Use llama2 quantized version to read retrieved chunks and respond to user query
- [ ] Interface: Build a chat interface with streamlit for ease of use


