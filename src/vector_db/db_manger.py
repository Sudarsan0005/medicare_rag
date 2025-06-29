import logging
import os
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from src.llm_manager.embeddings import EmbeddingGenerator



class VectorDb:
    def __init__(self, db_path="faiss_db"):
        self.db_path = db_path
        self.embedding = EmbeddingGenerator()
        self.vector_store = self._load_or_initialize_vector_store()

    def _load_or_initialize_vector_store(self):
        index_path = os.path.join(self.db_path, "index.faiss")

        if os.path.exists(index_path):
            logging.info(" Loading existing FAISS index...")
            return FAISS.load_local(
                folder_path=self.db_path,
                embeddings=self.embedding.nvidia_model,
                allow_dangerous_deserialization=True  # if needed
            )
        else:
            logging.info(" Initializing new FAISS index...")
            index = faiss.IndexFlatL2(len(self.embedding.generate("hello world",model_type='nvidia')))
            return FAISS(
                embedding_function=self.embedding.nvidia_model,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )

    def   save(self):
        self.vector_store.save_local(self.db_path)
