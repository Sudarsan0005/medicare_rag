import os
import logging
from src.constants import (DOC_DIR)
from fastapi import APIRouter, UploadFile
from src.engine.insert_vector_db import DataInsertion
from src.engine.engine import RagRetrival
from fastapi.responses import JSONResponse
from src.entity.entity_config import RagQuery

router = APIRouter()

if not os.path.exists(DOC_DIR):
    os.makedirs(DOC_DIR)
db_insert = DataInsertion(chunk_size=1200,chunk_overlap=120,db_path="faiss_db")
rag_retrival = RagRetrival(db_path="faiss_db")

@router.post("/insert_data")
async def insert_data(file: UploadFile):
    try:
        filename = file.filename
        if filename.split(".")[-1]!="pdf":
            return JSONResponse(
                status_code=400,
                content={"status": 0, "response": f"Only pdf file allowed"}
            )
        file_path = os.path.join(DOC_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        await db_insert.insert_data(file_path=file_path)
        return JSONResponse(
            status_code=200,
            content={"status": 1, "response": "data inserted successfully"}
        )
    except Exception as e:
        logging.critical("Error while inserting")
        return JSONResponse(
            status_code=400,
            content={"status": 0, "response": f"Error while inserting: {e}"}
        )

@router.post("/rag_qa")
async def rag_qa(request: RagQuery):
    try:
        query=request.query
        if not query:
            return JSONResponse(
                status_code=400,
                content={"status": 0, "response": "Query cannot be empty."}
            )

        rag_response = await rag_retrival.retrival(query=query)

        return JSONResponse(
            status_code=200,
            content={"status": 1, "response": rag_response}
        )
    except Exception as e:
        logging.critical("Error while retrieving from rag")
        return JSONResponse(
            status_code=400,
            content={"status": 0, "response": f"Error while retrieving: {e}"}
        )
