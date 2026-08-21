"""
Day 1 Starter — Ingestion Pipeline
-----------------------------------
Loads every PDF in ./data, splits it into overlapping chunks, embeds
those chunks, and stores them in a local ChromaDB collection. Every
chunk carries citation-ready metadata: document name, page number,
and a stable chunk id.

Usage:
    python ingest.py
"""
import sys
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import config
# Matches a section header only when it sits alone on its own line —
# e.g. a line containing exactly "3.6 Target blood pressure" and nothing
# else. This avoids false matches like a page-footer number ("...Page 9")
# being mistaken for a header just because a capitalized word follows it.
SECTION_HEADER_PATTERN = re.compile(
    r"^(\d+(?:\.\d+)*[a-z]?)\s+([A-Z][A-Za-z0-9\-,/ ]{3,80})\s*$",
    re.MULTILINE,
)

RECOMMENDATION_HEADER_PATTERN = re.compile(
    r"^Recommendation\s+(\d+):\s*(.+?)\s*$",
    re.IGNORECASE,
)
PAGE_NUMBER_TAG_PATTERN = re.compile(r"<page_number>(\d+)</page_number>")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def get_embedding_function():
    """Returns the embedding function based on config.EMBEDDING_PROVIDER."""
    if config.EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL)
    else:
        from langchain_community.embeddings import FastEmbedEmbeddings
        return FastEmbedEmbeddings(model_name=config.LOCAL_EMBEDDING_MODEL)


def load_pdfs(data_dir: Path):
    """Loads every PDF in data_dir and returns one LangChain Document per
    page, each carrying page-level metadata (document name, page number)."""
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {data_dir}/")
        print("Add a guideline PDF there, then re-run this script.")
        sys.exit(1)

    all_docs = []
    for pdf_path in pdf_files:
        print(f"Loading {pdf_path.name} ...")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            # Normalize metadata: every chunk downstream inherits this
            page.metadata["document_name"] = pdf_path.stem
            page.metadata["page_number"] = page.metadata.get("page", 0) + 1

            # Some PDFs (e.g. ESHRE) were exported from HTML and contain
            # leftover tags like <td>, <sup>, <page_number>. Capture the
            # guideline's own page number before stripping tags out.
            tag_match = PAGE_NUMBER_TAG_PATTERN.search(page.page_content)
            if tag_match:
                page.metadata["guideline_page"] = int(tag_match.group(1))

            cleaned = HTML_TAG_PATTERN.sub(" ", page.page_content)
            cleaned = re.sub(r"[ \t]+", " ", cleaned)
            page.page_content = cleaned
        all_docs.extend(pages)
        print(f"  -> {len(pages)} pages loaded")
    return all_docs


def chunk_documents(documents):
    """
    Creates section-aware chunks.

    - Detects normal numbered sections such as:
        3.1 Blood pressure threshold for initiation of pharmacological treatment

    - Detects recommendation headings such as:
        Recommendation 1: Blood pressure threshold for initiation of
        pharmacological treatment

    - Allows recommendation headings to span multiple PDF lines.
    - Carries the current section across page boundaries.
    - Splits each logical section into smaller chunks.
    """

    section_documents = []

    current_section = "N/A"
    current_text = []
    section_start_page = None
    current_document_name = None

    def flush_section():
        """Save the currently accumulated logical section."""
        nonlocal current_text
        nonlocal section_start_page
        nonlocal current_section
        nonlocal current_document_name

        text = "\n".join(current_text).strip()

        if not text:
            return

        metadata = {
            "document_name": current_document_name or "unknown",
            "page_number": (
                section_start_page
                if section_start_page is not None
                else 1
            ),
            "section": current_section,
        }

        section_documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

        current_text = []
        section_start_page = None

    # =========================================================
    # STEP 1 — Build logical sections
    # =========================================================

    for page_doc in documents:

        page_text = page_doc.page_content
        page_number = page_doc.metadata.get("page_number", "?")
        document_name = page_doc.metadata.get(
            "document_name",
            "unknown"
        )

        if document_name != current_document_name:
            flush_section()
            current_section = "N/A"

        current_document_name = document_name

        lines = page_text.splitlines()

        i = 0

        while i < len(lines):

            line = lines[i]
            stripped = line.strip()

            section_match = SECTION_HEADER_PATTERN.match(stripped)
            recommendation_match = RECOMMENDATION_HEADER_PATTERN.match(
                stripped
            )

            # -------------------------------------------------
            # Recommendation heading
            # -------------------------------------------------

            if recommendation_match:

                flush_section()

                recommendation_title = (
                    recommendation_match.group(2).strip()
                )

                # Recommendation headings can continue onto
                # the next PDF line.
                #
                # Example:
                #
                # Recommendation 1: Blood pressure threshold for initiation of
                # pharmacological treatment

                if i + 1 < len(lines):

                    next_line = lines[i + 1].strip()

                    if (
                        next_line
                        and not SECTION_HEADER_PATTERN.match(next_line)
                        and not RECOMMENDATION_HEADER_PATTERN.match(
                            next_line
                        )
                        and not next_line.startswith("WHO ")
                    ):
                        recommendation_title += " " + next_line

                        # We consumed the next line.
                        i += 1

                current_section = (
                    f"Recommendation "
                    f"{recommendation_match.group(1)}: "
                    f"{recommendation_title}"
                ).strip()

                section_start_page = page_number

                current_text.append(current_section)

            # -------------------------------------------------
            # Normal numbered section
            # -------------------------------------------------

            elif section_match:

                flush_section()

                current_section = (
                    f"{section_match.group(1)} "
                    f"{section_match.group(2)}"
                ).strip()

                section_start_page = page_number

                current_text.append(stripped)

            # -------------------------------------------------
            # Normal content
            # -------------------------------------------------

            else:

                if section_start_page is None:
                    section_start_page = page_number

                current_text.append(line)

            i += 1

    # Flush final section.
    flush_section()

    # =========================================================
    # STEP 2 — Split logical sections into chunks
    # =========================================================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE * 4,
        chunk_overlap=config.CHUNK_OVERLAP * 4,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = []

    for section_doc in section_documents:

        section_chunks = splitter.split_documents(
            [section_doc]
        )

        for chunk_index, chunk in enumerate(section_chunks):

            doc_name = chunk.metadata.get(
                "document_name",
                "unknown"
            )

            page = chunk.metadata.get(
                "page_number",
                "?"
            )

            section = chunk.metadata.get(
                "section",
                "N/A"
            )

            safe_section = re.sub(
                r"[^A-Za-z0-9]+",
                "_",
                section
            ).strip("_")

            chunk.metadata["section"] = section

            chunk.metadata["chunk_id"] = (
                f"{doc_name}-p{page}-"
                f"{safe_section}-c{chunk_index}"
            )

            chunks.append(chunk)

    return chunks


def build_index(chunks):
    """Embeds chunks and persists them into a local Chroma collection.
    Uses each chunk's chunk_id as its database ID, so re-running this
    script overwrites existing chunks instead of duplicating them."""
    embedding_fn = get_embedding_function()

    # Use the human-readable chunk_id as the real database ID.
    # This makes re-ingestion safe: same chunk_id = overwrite, not duplicate.
    chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]

    print(f"Embedding {len(chunks)} chunks using '{config.EMBEDDING_PROVIDER}' provider ...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DIR),
        ids=chunk_ids,
    )
    print(f"Done. Index saved to {config.CHROMA_DIR}/")
    return vectordb


def main():
    print("=== Day 1 Starter: Ingestion Pipeline ===\n")

    documents = load_pdfs(config.DATA_DIR)
    chunks = chunk_documents(documents)

    print(
        f"\nCreated {len(chunks)} chunks "
        f"from {len(documents)} pages.\n"
    )

    # TEMPORARY DEBUGGING

    print("\n=== RECOMMENDATION DEBUG ===")

    for chunk in chunks:
        if "Recommendation 1" in chunk.page_content:
            print("Chunk ID:", chunk.metadata.get("chunk_id"))
            print("Page:", chunk.metadata.get("page_number"))
            print("Section:", chunk.metadata.get("section"))
            print("Text:")
            print(chunk.page_content)
            print("============================")

    # IMPORTANT:
    # Keep this commented while debugging.
    #
    build_index(chunks)

    print(
        '\nNext step: run '
        'python query.py "your question here" '
        'to test retrieval.'
    )


if __name__ == "__main__":
    main()
