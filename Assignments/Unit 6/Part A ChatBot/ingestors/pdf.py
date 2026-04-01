"""PDF ingestor.

Fetches PDF article by topic and splits them into chunks
suitable for embedding and vector storage.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .base import BaseIngestor


class PDFIngestor(BaseIngestor):
    """Ingestor that fetches PDF articles by topic name."""

    def __init__(
        self,
        load_max_docs: int = 3,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        
        self.load_max_docs = load_max_docs
        
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @property
    def source_type(self) -> str:
        return "PDF"

    def load(self, source: str) -> list[Document]:
        """Fetch PDF articles matching *source* and return chunks.

        Args:
            source: PDF search query / topic name,
                    e.g. "Python programming language".

        Returns:
            List of text chunks as LangChain Documents.
        """

        # Set for single PDF document load
        loader = PyPDFLoader(source, mode="single")
        doc = loader.load()

        return doc, self._splitter.split_documents(doc)
