# Que

LLM question answering + local document vector storage.

## How it works

The `que` command line utility recursively indexes the documents in any directory you invoke it in, and stores them in a vector database (ChromaDB) in `~/.config/que/index.chroma`. Then it uses cosine similarity to find related texts based on your query, feeds them as context to Llama2 and has it answer the question.

On following invocations, `que` re-checks if the indexed files have changed or have been deleted, or if new documents are present, and updates the internal vector database, avoiding an expensive re-indexing of files.

`que` relies on `llama-cpp`, a fast inference implementation compatible with MPS, CUDA and Vulkan.