from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from .llm import get_embedding_model
import uuid
import pdfplumber
from pathlib import Path
import json
from dotenv import load_dotenv

load_dotenv()

VECTORDIR = "./vectorstore"

def extract_tables_rows(pdf_path):
    """
    Extract tables across pages and convert to structured row-documents.
    Returns list of (page_no, table_index, header_row, rows_list)
    """
    tables_info = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                page_tables = page.extract_tables()
                for t_idx, table in enumerate(page_tables):
                    if not table or len(table) == 0:
                        continue
                    # Try to detect header row: heuristically assume first non-empty row is header
                    header = None
                    # remove fully-empty rows
                    table = [r for r in table if any(cell not in (None, "") for cell in r)]
                    if len(table) == 0:
                        continue
                    header = table[0]
                    rows = table[1:] if len(table) > 1 else []
                    # If only one row and it's not a header, treat header as None and row as table[0]
                    if not rows and header:
                        # treat header as a single data row with no header
                        rows = [header]
                        header = None
                    tables_info.append({
                        "page": page_idx,
                        "table_index": t_idx,
                        "header": header,
                        "rows": rows
                    })
    except Exception as e:
        print("table extraction error:", e)
    return tables_info

def row_to_text(header, row):
    """
    Convert a row and optional header to a short textual representation,
    like "ColA: valA | ColB: valB" (for embeddings / retrieval).
    """
    cells = [str(c).strip() if c is not None else "" for c in row]
    if header:
        header_cells = [str(h).strip() if h is not None else "" for h in header]
        pieces = []
        for h, v in zip(header_cells, cells):
            if h:
                pieces.append(f"{h}: {v}")
            else:
                pieces.append(v)
        return " | ".join(pieces)
    else:
        # CSV style
        return ", ".join(cells)

def ingest_pdf(file_path: str, doc_id: str = None):
    """
    Ingest PDF into Chroma:
    - text chunks (split pages)
    - table rows as individual Documents with metadata: page, table_index, row_index, header_json
    returns doc_id
    """
    if doc_id is None:
        doc_id = str(uuid.uuid4())

    # Load PDF page-wise
    loader = PyPDFLoader(file_path)
    # load page docs then split into chunks for long pages
    pages = loader.load()  # list of Documents -> page_content + metadata
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = splitter.split_documents(pages)

    # add metadata for text chunks
    for d in text_chunks:
        d.metadata["doc_id"] = doc_id
        d.metadata.setdefault("source", str(Path(file_path).name))
        d.metadata.setdefault("type", "text")

    # Extract table rows and create one Document per row
    table_info = extract_tables_rows(file_path)
    table_row_docs = []
    for t in table_info:
        page_no = t["page"]
        t_idx = t["table_index"]
        header = t.get("header")
        header_json = None
        if header:
            # store header as json string in metadata
            header_json = json.dumps([h if h is not None else "" for h in header])
        for r_idx, row in enumerate(t["rows"]):
            text_repr = row_to_text(header, row)
            metadata = {
                "doc_id": doc_id,
                "source": str(Path(file_path).name),
                "type": "table_row",
                "page": page_no,
                "table_index": t_idx,
                "row_index": r_idx,
                "header": header_json
            }
            # store a small snippet + structured text
            doc = Document(page_content=text_repr, metadata=metadata)
            table_row_docs.append(doc)

    # Combine and persist into a Chroma collection named by doc_id
    all_docs = text_chunks + table_row_docs
    embeddings = get_embedding_model()
    # create a collection per document
    chroma_collection = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=VECTORDIR,
        collection_name=doc_id
    )
    chroma_collection.persist()
    return doc_id
