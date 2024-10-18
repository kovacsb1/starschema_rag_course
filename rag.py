from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage
)

from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from dotenv import load_dotenv
import os
import toml


load_dotenv()
llm_deployment = os.getenv("AZURE_LLM_DEPLOYMENT")
embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.getenv("API_VERSION")

config = toml.load("config.toml")
PERSIST_DIR = config["data"]["persist_dir"]
SIMILARITY_TOP_K = config["retrieval"]["similarity_top_k"]

# bge-base embedding model
embed_model= AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=embedding_deployment,
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version=azure_api_version,
)
Settings.embed_model = embed_model

llm = AzureOpenAI(
    model="gpt-4o",
    deployment_name=llm_deployment,
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version=azure_api_version,
)
Settings.llm = llm

storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(similarity_top_k=SIMILARITY_TOP_K)
response = query_engine.query("What did the author do growing up?")
print(response)

# for node in response.source_nodes:
#     print(node)
#     print(node.text)