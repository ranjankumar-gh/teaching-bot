###############################################################
# Data ingestion pipeline 
# 1. Taking the input pdf file
# 2. Extracting the content
# 3. Divide into chunks
# 4. Use embeddings model to convet to the embedding vector
# 5. Store the embedding vectors to the qdrant (vector database)
################################################################
import os
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import CharacterTextSplitter
from qdrant_client import QdrantClient

path = "ix-sst-ncert-democratic-politics"
filenames = next(os.walk(path))[2]

for i, file_name in enumerate(filenames):
    print(f"Data ingestion for the chapter: {i}")

    # 1. Load the pdf document and extract text from it
    loader = PDFMinerLoader(path + "/" + file_name)
    pdf_content = loader.load()
    print(pdf_content)

    # 2. Split the text into small chunks
    CHUNK_SIZE = 1000 # chunk size not greater than 1000 chars
    CHUNK_OVERLAP = 30 # a bit of overlap is required for continued context

    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = text_splitter.split_documents(pdf_content)

    # Make a list of split docs
    documents = []
    for doc in docs:
        documents.append(doc.page_content)

    # 3. Create vectordatabase(qdrant) client 
    qdrant_client = QdrantClient(url="http://localhost:6333")

    # 4. Add document chunks in vectordb
    qdrant_client.add(
        collection_name="ix-sst-ncert-democratic-politics",
        documents=documents,
        #metadata=metadata,
        #ids=ids
    )

    # 5. Make a query from the vectordb(qdrant)
    search_results = qdrant_client.query(
        collection_name="ix-sst-ncert-democratic-politics",
        query_text="What is democracy?"
    )

    for search_result in search_results:
        print(search_result.document, search_result.score)