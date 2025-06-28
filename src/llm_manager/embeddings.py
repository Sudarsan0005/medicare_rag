from langchain_openai import OpenAIEmbeddings
from src.constants import (openai_embedding_model,vector_size,nemo_embedding_model)
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

class EmbeddingGenerator:
    def __init__(self):
        self.open_model= OpenAIEmbeddings(
    model=openai_embedding_model,
    dimensions=vector_size
)
        self.nvidia_model = NVIDIAEmbeddings(model=nemo_embedding_model)
    def generate(self,text:str=None,model_type:str=None)-> list[float]:
        try:
            if model_type=="nvidia":
                return self.nvidia_model.embed_query(text)
            else:
                return self.open_model.embed_query(text)
        except Exception as e:
            raise Exception(f"Failed to generate embedding {e}")
# emb=EmbeddingGenerator()
# print(emb.generate(text="who are you",model_type="nvidia"))