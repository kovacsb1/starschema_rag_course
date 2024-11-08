from llama_index.core import (
    Settings,
    VectorStoreIndex,
    PromptTemplate
)

from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from dotenv import load_dotenv
import os
import toml
import qdrant_client


load_dotenv()
llm_deployment = os.getenv("AZURE_LLM_DEPLOYMENT")
embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.getenv("API_VERSION")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_port = os.getenv("QDRANT_PORT")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

config = toml.load("config.toml")
SIMILARITY_TOP_K = config["retrieval"]["similarity_top_k"]
QDRANT_COLLECTION_NAME = config["vectordb"]["collection_name"]

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

client = qdrant_client.QdrantClient(
    url=qdrant_url,
    port=qdrant_port,
    api_key=qdrant_api_key,
)

vector_store = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION_NAME)
index = VectorStoreIndex.from_vector_store(vector_store)

query_engine = index.as_query_engine(similarity_top_k=SIMILARITY_TOP_K)

updated_system_prompt = """
You are a helpful chatbot answering questions related to HCL policies.
Context information from HCL policty knowledge base is below.
---------------------
{context_str}
---------------------
Given this information, answer the query. 
Do not use any previous knowledge!
Query: {query_str}
Answer: 
"""

new_summary_tmpl = PromptTemplate(updated_system_prompt)

query_engine.update_prompts(
    {"response_synthesizer:summary_template": new_summary_tmpl}
)

query = "When will we get the bonus letters?"
response = query_engine.query(query)
print(response)

# for node in response.source_nodes:
#     print(node)
#     print(node.text)