from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings,
)
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os
import toml

load_dotenv()
embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.getenv("API_VERSION")

config = toml.load("config.toml")
CHUNK_SIZE = config["chunking"]["chunk_size"]
CHUNK_OVERLAP = config["chunking"]["chunk_overlap"]
SOURCE_DIR = config["data"]["source_dir"]
PERSIST_DIR = config["data"]["persist_dir"]

embed_model= AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=embedding_deployment,
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version=azure_api_version,
)
Settings.embed_model = embed_model

splitter = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

documents = SimpleDirectoryReader(SOURCE_DIR, recursive=True).load_data()
index = VectorStoreIndex.from_documents(documents, transformations=[splitter])
index.storage_context.persist(persist_dir=PERSIST_DIR)
