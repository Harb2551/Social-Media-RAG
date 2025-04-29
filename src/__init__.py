from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o")
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding()
