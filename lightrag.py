import os
import inspect
import logging
import asyncio
import nest_asyncio
nest_asyncio.apply()
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WORKING_DIR = "./dickens2"
DOCS_DIR = './docs'

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen2.5",
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {"num_ctx": 32768},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model="bge-m3", host="http://localhost:11434"
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)


def load_docs_from_folder(folder_path, rag):
    logger.info(f"Loading docs from {folder_path}")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                rag.insert(f.read())

def query_rag(query, rag):

    # stream response
    param = QueryParam(mode="global", stream=True)

    resp = rag.query(
        query,
        param=param,
    )

    if inspect.isasyncgen(resp):
        asyncio.run(print_stream(resp))
    else:
        print(resp)

if __name__ == "__main__":
    try:
        # Initialize
        rag = asyncio.run(initialize_rag())
        load_docs_from_folder(DOCS_DIR, rag)

        print("\nRAG system ready. Type 'exit' or 'quit' to end.")
        while True:
            query = input("\nUser ask: ")
            if query.lower() in ["exit", "quit"]:
                break
            query_rag(query, rag)
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
