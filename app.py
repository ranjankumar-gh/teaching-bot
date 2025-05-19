import streamlit as st
from qdrant_client import QdrantClient
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

qdrant_client = QdrantClient(url="http://localhost:6333")

st.markdown("# Teaching Bot")
st.markdown("#### Class: IX, Subject: sst, Board: NCERT, Topic: Democratic Politics")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if query := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(query)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    # Connect with vector db for getting the context
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

    # Using LLM for forming the answer
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

    #response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})