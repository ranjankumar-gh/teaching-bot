from qdrant_client import QdrantClient
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

qdrant_client = QdrantClient(url="http://localhost:6333")

# 2. Make a query to the vectordb (qdrant)
#query = "explain democracy in estonia?"
query = "what is democracy."

search_results = qdrant_client.query(
    collection_name="ix-sst-ncert-democratic-politics",
    query_text=query
)


context = ""
no_of_docs = 2
count = 1
for search_result in search_results:
    if search_result.score >= 0.8:
        #print(f"Retrieved document: {search_result.document}, Similarity score: {search_result.score}")
        context = context + search_result.document
    if count >= no_of_docs:
        break
    count = count + 1

#print(f"Retrieved Context: {context}")

# 4. Using LLM for forming the answer
template = """Instruction: {instruction}
Context: {context}
Query: {query}
"""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="llama3.2") # Using llama3.2 as llm model

chain = prompt | model

bot_response = chain.invoke({"instruction": "Answer the question based on the context below. If you cannot answer the question with the given context, answer with \"I don't know.\"", 
            "context": context,
            "query": query
            })

print(f'\nBot: {bot_response}')