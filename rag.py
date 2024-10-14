from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.node_parser import SentenceSplitter
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
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.getenv("API_VERSION")

# bge-base embedding model
embed_model =  HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.embed_model = embed_model

llm = AzureOpenAI(
    model="gpt-4o",
    deployment_name=llm_deployment,
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version=azure_api_version,
)
Settings.llm = llm

splitter = SentenceSplitter(
    chunk_size=256,
    chunk_overlap=20,
)

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, transformations=[splitter])

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)

prompts_dict = query_engine.get_prompts()
display_prompt_dict(prompts_dict)

for node in response.source_nodes:
    print(node)
    print(node.text)