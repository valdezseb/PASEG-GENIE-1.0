#import streamlit as st
from sentence_transformers import SentenceTransformer

import streamlit as st
from PyPDF2 import PdfReader
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
import openai
import os
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma, Pinecone
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA


st.set_page_config(page_title="PASEG Genie", page_icon=":coffee:")

# Load Pinecone API key
api_key = st.secrets["pinecone_api_key"]
pinecone.init(api_key=api_key, environment='asia-southeast1-gcp-free')
index_name = 'dbpaseg'


os.environ['OPENAI_API_KEY'] = st.secrets['openai_api_key']




# Create a function to load Sentence-Transformers embeddings model with caching
@st.cache(allow_output_mutation=True)
def load_embeddings_model():
    return SentenceTransformer('sentence-transformers/all-distilroberta-v1')

# Create the Chat and RetrievalQA objects outside the Streamlit app function
chat = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0.80)
qachain = load_qa_chain(chat, chain_type='stuff')
qa = RetrievalQA(combine_documents_chain=qachain, retriever=None)  # We'll set the retriever later

# Load the Sentence-Transformers embeddings model using caching
embeddings_model = load_embeddings_model()

# Create a function to load the Pinecone client with caching
@st.cache(allow_output_mutation=True)  # Allow output mutation for the Pinecone client
def load_pinecone_client():
    return Pinecone.from_existing_index(index_name, embeddings_model.encode)

# Load the Pinecone client using st.cache
docsearch = load_pinecone_client()

condition1 = '\n [organize information: organize text so its easy to read, and bullet points when needed.] \n [tone and voice style: clear sentences, avoid use of complex sentences]'

st.title("PASEG Genie // Donate a Coffee :coffee:")

# Let the user input a query
query = st.text_input("Enter your query:")

# Run the QA system and display the result using Streamlit
if query:
    # Set the Pinecone retriever for the QA system before running
    qa.retriever = docsearch.as_retriever()
    result = qa.run(query + '\n' + condition1)
    st.write(result)
