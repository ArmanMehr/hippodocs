from pathlib import Path

from langchain_community.document_loaders import TextLoader
from liteparse import LiteParse


def load_txt(path: str) -> str:
    loaded_docs = TextLoader(path, encoding="UTF-8").load()
    return "\n\n".join([doc.page_content for doc in loaded_docs])


def load_pdf(file_data: str | Path | bytes) -> str:
    parser = LiteParse(ocr_enabled=True)
    result = parser.parse(file_data)

    all_texts = []
    for page in result.pages:
        full_text = "\n".join([item.text for item in page.text_items])
        all_texts.append(full_text)

    return "\n\n".join(all_texts)
