import streamlit as st
from pinecone import Pinecone, PodSpec
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


# Pinecone setup
api_key = st.secrets["pinecone_api_key"]  
pc = Pinecone(api_key)
index = pc.Index("example-index")

# LangChain setup
embeddings = HuggingFaceEmbeddings() 
retriever = index.as_retriever(embeddings)
chat = ChatOpenAI(model_name="gpt-3.5-turbo")
qa = RetrievalQA(retriever=retriever, chat_model=chat)

st.title("My QA Bot")
query = st.text_input("Ask a question:") 

if query:
  result = qa.run(query)
  st.write(result)
