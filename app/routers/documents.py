from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from langchain_core.documents import Document as LCDocument

from app.configs import get_settings
from app.limiter import limiter
from app.schemas import DocumentCreateSchema, DocumentListSchema, DocumentSchema
from app.services.embedding import DocumentIngestionPipeline
from app.services.factory import get_document_repository, get_ingestion_pipeline
from app.services.repository import DocumentRepository
from app.utils.loader import load_pdf

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/", status_code=status.HTTP_201_CREATED, response_model=list[DocumentSchema]
)
@limiter.limit("60/minute")
def add_document(
    request: Request,
    payload: DocumentCreateSchema,
    pipeline: DocumentIngestionPipeline = Depends(get_ingestion_pipeline),
):

    docs = pipeline.run([LCDocument(page_content=payload.text)])
    return [DocumentSchema(id=d.id, content=d.content) for d in docs]


@router.post(
    "/upload", status_code=status.HTTP_201_CREATED, response_model=list[DocumentSchema]
)
@limiter.limit("10/minute")
def upload_document(
    request: Request,
    file: UploadFile = File(...),
    pipeline: DocumentIngestionPipeline = Depends(get_ingestion_pipeline),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported",
        )
    content = file.file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file"
        )

    if len(content) > get_settings().MAX_FILESIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large (max 10MB)",
        )

    pdf_document = load_pdf(content)
    if not pdf_document:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No text could be extracted from this PDF",
        )

    docs = pipeline.run(pdf_document)
    return [DocumentSchema(id=d.id, content=d.content) for d in docs]


@router.get("/", response_model=DocumentListSchema)
@limiter.limit("120/minute")
def list_documents(
    request: Request,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    repo: DocumentRepository = Depends(get_document_repository),
):
    docs, total = repo.list_all(skip, limit)
    return DocumentListSchema(
        documents=[DocumentSchema(id=d.id, content=d.content) for d in docs],
        total=total,
    )


@router.get("/{document_id}", response_model=DocumentSchema)
@limiter.limit("120/minute")
def get_document(
    request: Request,
    document_id: int,
    repo: DocumentRepository = Depends(get_document_repository),
):
    doc = repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )
    return DocumentSchema(id=doc.id, content=doc.content)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("60/minute")
def delete_document(
    request: Request,
    document_id: int,
    repo: DocumentRepository = Depends(get_document_repository),
):
    if not repo.delete(document_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )
