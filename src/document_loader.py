"""
Document Loader for Social Media App Support Agent

This module provides utilities for loading and processing documentation files for the Social Media App support chatbot. It handles reading files from various formats, preprocessing text, and chunking documents
into appropriate sizes for the vectorstore.
"""

import os
import glob
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

class SocialMediaDocumentLoader:
    """
    Loads, cleans, and processes social media documentation files for use in AI support agents.

    This class reads text files from a directory, cleans their content by removing unnecessary elements, and splits documents into optimally sized chunks for efficient vector embedding
    and retrieval. It handles common issues in social media documentation like non-breaking spaces and standardized footers.

    Attributes:
        data_dir (Path): Path to the directory containing documentation files.
        chunk_size (int): Maximum size of each text chunk in characters.
        chunk_overlap (int): Number of characters to overlap between chunks.
        min_chunk_size (int): Minimum size for a chunk to be considered valid.
        text_splitter (RecursiveCharacterTextSplitter): Utility to split text into chunks.
    """

    def __init__(
        self,
        data_dir: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 50
    ):
        """
        Initialize the SocialMediaDocumentLoader with specified parameters.

        What this does:
        1. Converts the data_dir string to a Path object for better file operations
        2. Stores chunking parameters as instance variables
        3. Creates a RecursiveCharacterTextSplitter with the specified parameters
        4. Sets up specific separators for smart document splitting

        Args:
            data_dir (str): Directory path where documentation files are stored.
            chunk_size (int, optional): Maximum size of each text chunk. Defaults to 1000.
            chunk_overlap (int, optional): Overlap between consecutive chunks. Defaults to 200.
            min_chunk_size (int, optional): Minimum size for a valid chunk. Defaults to 50.
        """
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )


    def load_documents(self, file_pattern: str = "*.txt") -> List[Document]:
        """
        Loads raw documents from files matching a pattern in the specified directory.

        What this does:
        1. Uses glob to find all files matching the pattern in data_dir
        2. Returns an empty list immediately if no matching files are found
        3. For each file path:
           a. Creates a TextLoader with UTF-8 encoding
           b. Loads document(s) from the file
           c. Adds the source filename to metadata for each document
           d. Adds documents to the result list
        4. Silently skips files that cause exceptions during loading

        Args:
            file_pattern (str, optional): Glob pattern to match files. Defaults to "*.txt".

        Returns:
            List[Document]: List of Document objects containing the loaded content.
                            Returns an empty list if no files are found or if errors occur during loading.

        Note:
            Each document's metadata will include the source filename.
            Continues processing even if some files raise exceptions during loading.
        """
        documents = []
        file_paths = glob.glob(str(self.data_dir / file_pattern))
        if not file_paths:
            return []

        for path in file_paths:
            try:
                loader = TextLoader(path, encoding='utf-8')
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source"] = os.path.basename(path)
                    documents.append(doc)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue
        return documents


    def clean_text(self, text: str) -> str:
        """
        Cleans and normalizes text content to improve quality for embedding.

        This method performs the following cleaning operations in sequence:
        1. Replaces non-breaking spaces (\xa0) with regular spaces
        2. Normalizes whitespace by tokenizing and rejoining with single spaces
        3. Fixes encoding issues for special characters:
        - Replaces "â" with "'" (apostrophe)
        - Replaces "â" with "'" (another apostrophe variant)
        - Replaces "â" with "-" (dash)
        4. Normalizes common Unicode characters:
        - "Ã§" → "ç" (c with cedilla)
        - "Ã¨" → "è" (e with grave accent)
        - "Ã©" → "é" (e with acute accent)
        - "Ãª" → "ê" (e with circumflex)
        - "Ã±" → "ñ" (n with tilde)
        - "Ø§Ù" → "ال" (Arabic characters)
        - "ÙØ§" → "فا" (Arabic characters)
        - "Ø¨Ù" → "بي" (Arabic characters)
        - "×¢×" → "עב" (Hebrew characters)
        - "×¨××ª" → "רית" (Hebrew characters)
        5. Removes common footer sections by finding markers like:
        - "Share this Post"
        - "About the company"
        - "X platform X.com"
        - "© 2023 X Corp"
        - "Did someone say â¦ cookies?"
        - "Cookies MStV Transparenzangaben"
        - "How To Contact Us"
        6. Removes navigation elements like:
        - "Skip to main content"
        - "Help Center"
        - "Contact our support team"
        - "Verify your account"
        - "Access your account"
        - "FAQ"
        7. Removes duplicate paragraphs
        8. Reduces multiple spaces to single spaces
        9. Reduces multiple consecutive newlines to at most double newlines

        Args:
            text (str): The original text to clean.

        Returns:
            str: Cleaned and normalized text with all transformations applied.

        Examples:
            >>> loader = SocialMediaDocumentLoader("data_dir")
            >>> text = "This  has \xa0 extra  \n spaces\nDid someone say â¦ cookies? Don't show this"
            >>> loader.clean_text(text)
            'This has extra spaces'
        """
        import re

        # Replacements
        text = text.replace('\xa0', ' ')
        text = re.sub(r"\s+", " ", text)

        replacements = {
            "â€™": "'",
            "â€œ": '"',
            "â€": '"',
            "â€“": "-",
            "Ã§": "ç",
            "Ã¨": "è",
            "Ã©": "é",
            "Ãª": "ê",
            "Ã±": "ñ",
            "Ø§Ù": "ال",
            "ÙØ§": "فا",
            "Ø¨Ù": "بي",
            "×¢×": "עב",
            "×¨××ª": "רית",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Remove footer/navigation markers
        removal_patterns = [
            r"Share this Post.*",
            r"About the company.*",
            r"X platform X.com.*",
            r"© \d{4} X Corp.*",
            r"Did someone say â¦ cookies?.*",
            r"Cookies MStV Transparenzangaben.*",
            r"How To Contact Us.*",
            r"Skip to main content.*",
            r"Help Center.*",
            r"Contact our support team.*",
            r"Verify your account.*",
            r"Access your account.*",
            r"FAQ.*"
        ]
        for pattern in removal_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Remove duplicate paragraphs
        paragraphs = list(dict.fromkeys(text.split('\n')))
        text = '\n'.join(paragraphs)

        # Final cleanup
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Processes documents by cleaning text and splitting into optimally sized chunks.

        What this does:
        1. Initializes an empty list for processed documents
        2. For each input document:
           a. Cleans the document text using the clean_text method
           b. Creates a new Document with cleaned text and original metadata
           c. Splits the document into chunks using the text_splitter
           d. Filters out chunks smaller than min_chunk_size
           e. Adds valid chunks to the result list

        Args:
            documents (List[Document]): List of Document objects to process.

        Returns:
            List[Document]: List of processed Document objects split into chunks.
                           Chunks smaller than min_chunk_size are filtered out.
        """
        processed_docs = []
        for doc in documents:
            cleaned_text = self.clean_text(doc.page_content)
            cleaned_doc = Document(page_content=cleaned_text, metadata=doc.metadata)
            chunks = self.text_splitter.split_documents([cleaned_doc])
            valid_chunks = [chunk for chunk in chunks if len(chunk.page_content) >= self.min_chunk_size]
            processed_docs.extend(valid_chunks)
        return processed_docs



    def load_and_process(self, file_pattern: str = "*.txt") -> List[Document]:
        """
        Performs end-to-end document loading and processing in a single operation.

        What this does:
        1. Calls load_documents with the provided file pattern
        2. Passes the loaded documents to process_documents
        3. Returns the processed document chunks

        This method chains the two main operations (loading and processing) for convenience.

        Args:
            file_pattern (str, optional): Glob pattern to match files. Defaults to "*.txt".

        Returns:
            List[Document]: List of processed Document objects ready for indexing.
                           Returns processed chunks from all successfully loaded documents.
        """
        raw_docs = self.load_documents(file_pattern=file_pattern)
        return self.process_documents(raw_docs)
