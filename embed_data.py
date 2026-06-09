import os
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import pandas as pd

def extract_and_index_pdf(pdf_path):
    print(f"Opening PDF: {pdf_path}...")

    if not os.path.exists(pdf_path):
        print(f"Error: Could not find the file at {pdf_path}. Please double-check the path.")
        return
    
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"Successfully loaded PDF with {total_pages} pages.")

    # Intialize ChromaDB client (creates folder called sdtmig_vector_db)
    chroma_client = chromadb.PersistentClient(path='./sdtmig_vector_db')
    default_ef = embedding_functions.DefaultEmbeddingFunction()

    # Create new collection for the SDTMIG
    collection = chroma_client.get_or_create_collection(
        name='sdtmig_v3_4',
        embedding_function=default_ef
    )

    documents = []
    metadatas = []
    ids = []

    print("Parsing pages and preparing text chunks...")

    # Loop through each page (starting after the table of contents/intro usually helps, 
    # but for safety, we will index everything)
    for pagenum, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text.strip():
            continue # skip blank pages
    
        # Strategy: Index text by page. If a page is huge we could split it, 
        # but for SDTMIG standard layout page-by-page tracking is best for tracking sources
        documents.append(text)
        metadatas.append({
            "source": os.path.basename(pdf_path),
            "page": pagenum
        })
        ids.append(f"page_{pagenum}")

    print(f"Adding {len(documents)} pages to the vector database. This might take a moment...")

    # Batch upload to prevent hitting memory or payload constraints
    batchsize = 100
    for i in range(0, len(documents), batchsize):
        collection.add(
            documents=documents[i:i+batchsize],
            metadatas=metadatas[i:i+batchsize],
            ids=ids[i:i+batchsize]
        )
        print(f"Index pages {i+1} to {min(i+batchsize, len(documents))}")

    print("\nSDTMIG v3.4 successfully vectorized and stored!")

# CDISC SDTM Controlled Terminolgy 
def extract_and_index_xls(xls_path):
    print(f"Opening XLS: {xls_path}...")

    if not os.path.exists(xls_path):
        print(f"Error: Could not find the file at {xls_path}. Please double-check the path.")
        return

    df = pd.read_excel(xls_path, sheet_name=1)
    df = df.fillna("")
    print(f"Successfully loaded excel.")

    # Intialize ChromaDB client (connects to same vector db storage location)
    chroma_client = chromadb.PersistentClient(path='./sdtmig_vector_db')
    default_ef = embedding_functions.DefaultEmbeddingFunction()

    # Create new collection for CDISC CT 
    collection = chroma_client.get_or_create_collection(
        name='cdisc_ct',
        embedding_function=default_ef
    )

    documents = []
    metadatas = []
    ids = []

    print("Processing excel rows into vector chunks...")
    for index, row in df.iterrows():
        text_content = (
            f"Codelist Name: {row['Codelist Name']} {row['CDISC Submission Value']} ({row['Codelist Code']}). "
            f"Submission Value: {row['CDISC Submission Value']}. "
            f"Synonyms: {row['CDISC Synonym(s)']}. "
            f"Definition: {row['CDISC Definition']}. "
            f"Extensible: {row['Codelist Extensible (Yes/No)']}."
        )

        # Strategy: flatten row-level relational columns into rich narrative strings.
        # Keeps the codelist name, submission value, and definitions coupled together
        # allowing the vector database to match on any attribute within that specific term
        documents.append(text_content)
        metadatas.append({
            "source": os.path.basename(xls_path),
            "codelist": str(row['Codelist Name']).strip(),
            "codelist_code": str(row['Codelist Code']).strip(),
            "submission_value": str(row['CDISC Submission Value']).strip().upper(),
            "synonyms": str(row['CDISC Synonym(s)']).strip().upper(),
            "is_extensible": str(row['Codelist Extensible (Yes/No)']).strip().upper()
        })
        unique_id = f"ct_{row['Codelist Code']}_{row['CDISC Submission Value']}_{index}"
        ids.append(unique_id)

    # Batch upload to prevent hitting memory or payload constraints
    batchsize = 500
    for i in range(0, len(documents), batchsize):
        collection.add(
            documents=documents[i:i+batchsize],
            metadatas=metadatas[i:i+batchsize],
            ids=ids[i:i+batchsize]
        )
        print(f"Index rows {i+1} to {min(i+batchsize, len(documents))}")

    print("\nCDISC SDTM Terminolgy successfully vectorized and stored!")


if __name__ == "__main__":
    # point to file location
    pdf_file_path = "./cdisc_docs/SDTMIG v3.4-FINAL_2022-07-21.pdf"
    xls_file_path ="./cdisc_docs/SDTM Terminology.xls"

    extract_and_index_pdf(pdf_file_path)
    extract_and_index_xls(xls_file_path)







