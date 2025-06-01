import logging

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CHROMA_DB_PATH = './chroma_db'
DOCS_DIR = './docs'
COLLECTION_NAME = 'my_collection'

# Initialize LLM with timeout settings
llm = Ollama(
    model="qwen2.5",
    request_timeout=300.0,
    base_url="http://localhost:11434",
    temperature=0.1
)

# Initialize Embedding Model
embed_model = OllamaEmbedding(
    model_name="bge-m3",
    base_url="http://localhost:11434",
    timeout=300.0
)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512


def initialize_or_load_vector_store():
    """Initialize or load existing vector store"""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # Check if collection is empty
    if collection.count() == 0:
        logger.info("Vector store is empty, creating new index...")
        documents = SimpleDirectoryReader(DOCS_DIR).load_data()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
    else:
        logger.info("Loading existing vector store...")
        index = VectorStoreIndex.from_vector_store(vector_store)

    return index


def create_query_engine(index, similarity_top_k=5, retriever_mode="embedding"):
    """
        Query the index with configurable parameters

        :param index: VectorStoreIndex instance
        :param similarity_top_k: Number of top results to return
        :param retriever_mode: Retrieval strategy ("default", "embedding", "mmr")
        :return: Query result

        模式	            说明	                                使用场景
        "default"	    默认基于相似度的最近邻检索	            快速、轻量，适合大多数情况
        "embedding"	    嵌入向量排序器（embedding ranker）	    对检索结果再排序，提升准确率
        "mmr"	        MMR（Maximal Marginal Relevance）	兼顾相关性和多样性，防止重复答案

        """
    return index.as_query_engine(
        similarity_top_k = similarity_top_k,
        retriever_mode = retriever_mode,
        streaming = True,
        response_mode = "compact"
    )


def stream_response(response_gen):
    """Stream the response tokens to the console"""
    print("Bot: ", end="", flush=True)
    full_response = ""
    for token in response_gen:
        # print(token, end="", flush=True)
        full_response += token
    print()  # New line after streaming completes
    return full_response


def query_rag(prompt, query_engine):
    try:
        response = query_engine.query(prompt)

        if hasattr(response, 'response_gen'):  # Streaming response
            return stream_response(response.response_gen)
        elif hasattr(response, 'response'):  # Non-streaming fallback
            print("Bot:", response.response)
            return response.response
        else:
            print("Bot:", str(response))
            return str(response)

    except Exception as e:
        logger.error(f"Query error: {e}")
        error_msg = f"Error processing your request: {str(e)}"
        print("Bot:", error_msg)
        return error_msg


if __name__ == "__main__":
    try:
        # Initialize or load vector store
        index = initialize_or_load_vector_store()
        query_engine = create_query_engine(index, similarity_top_k=20, retriever_mode="embedding")

        # Interactive loop
        print("\nRAG system ready. Type 'exit' or 'quit' to end.")
        while True:
            query = input("\nUser: ")
            if query.lower() in ["exit", "quit"]:
                break
            print("Bot:", query_rag(query, query_engine))
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")