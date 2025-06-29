from src.data_manager.data_loader import DataExtractor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.llm_manager import (embeddings)
from src.vector_db.db_manger import VectorDb
from uuid import uuid4

import logging

class DataInsertion:
    def __init__(self,chunk_size,chunk_overlap,db_path=None):
        self.data_extractor = DataExtractor()
        self.embedding_model = embeddings.EmbeddingGenerator().nvidia_model
        self.splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
        self.vector_db = VectorDb(db_path=db_path)
    async def insert_data(self,file_path):
        try:
            langchain_doc = self.data_extractor.data_loader(file_path=file_path)
            chunked_data = self.splitter.split_documents(langchain_doc)
            uuids = list(map(str, (uuid4() for _ in range(len(chunked_data)))))
            self.vector_db.vector_store.add_documents(documents=chunked_data, ids=uuids)
            self.vector_db.save()
            logging.info(f"data save to vectorDB")
        except Exception as e:
            logging.CRITICAL(f"Error while inserting data {e}")
            raise Exception(f"Error while inserting data {e}")