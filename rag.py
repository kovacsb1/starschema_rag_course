from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage
)

from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from dotenv import load_dotenv
from IPython.display import Markdown, display
import os

# define prompt viewing function
def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}" f"**Text:** "
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown(""))

load_dotenv()
llm_deployment = os.getenv("AZURE_LLM_DEPLOYMENT")
embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.getenv("API_VERSION")

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

PERSIST_DIR = "./storage"
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(similarity_top_k=10)
response = query_engine.query("What did the author do growing up?")
print(response)

prompts_dict = query_engine.get_prompts()
display_prompt_dict(prompts_dict)

for node in response.source_nodes:
    print(node)
    print(node.text)