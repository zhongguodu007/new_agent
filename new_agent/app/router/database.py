from fastapi import APIRouter
from pydantic import BaseModel
from ..args import db

router = APIRouter()

class CollectionRequest(BaseModel):
    collection_name: str
    description: str = "null"

class DocumentRequest(BaseModel):
    collection_name: str
    file_name: str
    description: str = "null"

class DeleteDocumentRequest(BaseModel):
    collection_name: str
    file_name: str

class GetDocumentRequest(BaseModel):
    collection_name: str



@router.post("/new_collection")
async def new_collection(request: CollectionRequest):
    return db.new_collection(request.collection_name, request.description)

@router.post("/delete_collection")
async def delete_collection(request: CollectionRequest):
    return db.delete_collection(request.collection_name)

@router.post("/add_document")
async def add_document(request: DocumentRequest):
    return db.add_document(
        request.collection_name, 
        request.file_name, 
        request.description
    )

@router.post("/delete_document")
async def delete_document(request: DeleteDocumentRequest):
    return db.delete_document(
        request.collection_name, 
        request.file_name
    )

@router.post("/get_collection")
async def get_collection():
    return db.get_collection()

@router.post("/get_document")
async def get_document(request: GetDocumentRequest):
    return db.get_document(request.collection_name)
