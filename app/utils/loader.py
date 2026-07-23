from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from liteparse import LiteParse


def create_documents_from_texts(
    texts: list[str], metadatas: list[dict[str, str]] | None = None
) -> list[Document]:
    metadatas = metadatas or [{} for _ in texts]
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(texts, metadatas)
    ]


def load_txt(path: str) -> list[Document]:
    return TextLoader(path, encoding="UTF-8").load()


def load_pdf(file_data: str | Path | bytes) -> list[Document]:
    parser = LiteParse(ocr_enabled=True)
    result = parser.parse(file_data)

    documents = []
    for page in result.pages:
        full_text = "\n".join([item.text for item in page.text_items])

        file_path = file_data if not isinstance(file_data, bytes) else None
        metadata = {
            "source": file_path,
            "page_number": page.page_num,
            "bbox_count": len(page.text_items),
            "layout_type": "spatial_grid",
        }

        doc = Document(page_content=full_text, metadata=metadata)
        documents.append(doc)

    return documents
