###############################################################
# Data ingestion pipeline
# 1. Taking the input pdf file
# 2. Extracting the content
# 3. Divide into chunks
# 4. Use embeddings model to convet to the embedding vector
# 5. Store the embedding vectors to the qdrant (vector database)
################################################################
import os
import uuid
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import CharacterTextSplitter
from qdrant_client import QdrantClient, models

path = "ix-sst-ncert-democratic-politics"
filenames = next(os.walk(path))[2]

COLLECTION = "ix-sst-ncert-democratic-politics"
MODEL = "BAAI/bge-small-en"  # use the same embedding model when querying

# 3. Create vectordatabase(qdrant) client and the collection (once)
qdrant_client = QdrantClient(url="http://localhost:6333")
if qdrant_client.collection_exists(COLLECTION):
    vectors = qdrant_client.get_collection(COLLECTION).config.params.vectors
    if isinstance(vectors, dict):  # named vectors: left by this repo's 2025 version
        qdrant_client.delete_collection(COLLECTION)
if not qdrant_client.collection_exists(COLLECTION):
    qdrant_client.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(
            size=qdrant_client.get_embedding_size(MODEL),
            distance=models.Distance.COSINE,
        ),
    )

for i, file_name in enumerate(filenames):
    print(f"Data ingestion for the chapter: {i}")

    # 1. Load the pdf document and extract text from it
    loader = PDFMinerLoader(path + "/" + file_name)
    pdf_content = loader.load()
    print(pdf_content)

    # 2. Split the text into small chunks
    CHUNK_SIZE = 1000 # target size; a paragraph longer than this stays whole
    CHUNK_OVERLAP = 30 # a bit of overlap is required for continued context

    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = text_splitter.split_documents(pdf_content)

    # Make a list of split docs
    documents = []
    for doc in docs:
        documents.append(doc.page_content)

    # 4. Add document chunks in vectordb
    qdrant_client.upsert(
        COLLECTION,
        points=[
            models.PointStruct(
                # same id on every run, so re-running updates instead of duplicating
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_name}:{idx}")),
                vector=models.Document(text=text, model=MODEL),
                payload={"document": text},
            )
            for idx, text in enumerate(documents)
        ],
    )

    # 5. Make a query from the vectordb(qdrant)
    search_results = qdrant_client.query_points(
        COLLECTION,
        query=models.Document(text="What is democracy?", model=MODEL),
    ).points

    for search_result in search_results:
        print(search_result.payload["document"], search_result.score)
