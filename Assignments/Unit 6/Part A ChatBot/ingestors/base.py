"""Base ingestor interface.

All document sources implement this interface. Students add new sources
by subclassing BaseIngestor and registering the new class in rag_demo.py.
"""

from abc import ABC, abstractmethod
from langchain_core.documents import Document


class BaseIngestor(ABC):
    """Abstract base class for all document ingestors."""

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Human-readable name for this source (e.g. 'PDF')."""

    @abstractmethod
    def load(self, source: str) -> list[Document]:
        """Load and split documents from the given source string.

        Args:
            source: Source identifier - could be a topic name, file path,
                    URL, etc. depending on the ingestor implementation.

        Returns:
            A list of split Document chunks ready to embed and store.
        """
