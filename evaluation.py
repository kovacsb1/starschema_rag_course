from llama_index.core import (
    Settings,
    VectorStoreIndex,
    PromptTemplate
)

from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.llama_dataset import LabelledRagDataset
from dotenv import load_dotenv
import os
import toml
import qdrant_client
import nest_asyncio
from llama_index.core.llama_pack import download_llama_pack

nest_asyncio.apply()


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

# use as judge
llm_gpt4 = AzureOpenAI(
    model="gpt-4o",
    deployment_name=llm_deployment,
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version=azure_api_version,
)

# test this
llm_ollama = Ollama(model="phi3:latest", request_timeout=120.0, temperature=0.0)
# Settings.llm = llm

client = qdrant_client.QdrantClient(
    url=qdrant_url,
    port=qdrant_port,
    api_key=qdrant_api_key,
)

# load vector store
vector_store = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION_NAME)
index = VectorStoreIndex.from_vector_store(vector_store)

query_engine = index.as_query_engine(
    llm=llm_ollama,
    similarity_top_k=SIMILARITY_TOP_K)

def create_eval_dataset():
    nodes = vector_store.get_nodes()
    # only use the first 10 as these are slow
    nodes = nodes[:5]

    # generate test dataset
    eval_filename = "test_questions.json"
    dataset_generator = RagDatasetGenerator(
        nodes=nodes,
        llm=llm_gpt4,
        num_questions_per_chunk=1
    )
    rag_dataset = dataset_generator.generate_dataset_from_nodes()

    # save it to json
    rag_dataset.save_json(eval_filename)


create_eval_dataset()