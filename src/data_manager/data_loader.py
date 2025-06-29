from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from typing import List

class DataExtractor:
    def __init__(self):
        pass

    def data_loader(self, file_path: str = None)->List:
        try:
            file_loader = PyPDFLoader(file_path=Path(file_path))
            return file_loader.load()
        except Exception as e:
            raise Exception(f"pdf data loader error {e}")



