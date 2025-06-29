from src.data_manager.data_loader import DataExtractor
from src.llm_manager import nvidia_llm, embeddings
from src.llm_manager.prompt.prompt_manager import rag_prompt
from src.vector_db.db_manger import VectorDb
from src.constants import (llm_model)
import json
import logging

class RagRetrival:
    def __init__(self,db_path=None):
        self.data_extractor = DataExtractor()
        self.llm = nvidia_llm.LlmManager(model=llm_model, sys_prompt=rag_prompt)
        self.vector_db = VectorDb(db_path=db_path)
    async def retrival(self,query:str=None)->str:
        try:
            retrieved_docs =self. vector_db.vector_store.similarity_search_with_score(query, k=3)
            print(retrieved_docs)
            formated_data = {
                f"context_data: {res.page_content}": {
                    "meta_data": {"source_page": res.metadata["page_label"],
                                  "score": f"{score:3f}",
                    "chunk_size":1200}
                }
                for res, score in retrieved_docs
            }
            response = await self.llm.chat_llm(query, formated_data)
            print(formated_data)
            response =response.replace("```","").replace("json","")
            response = json.loads(response)
            return response
        except Exception as e:
            logging.critical(f"Error RagRetrival>retrival {e}")
            raise Exception(f"Error while retrieving data {e}")