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


#st.set_page_config(page_title="PASEG Genie", page_icon=":coffee:")

# Load Pinecone API key
api_key = st.secrets["pinecone_api_key"]
pinecone.init(api_key=api_key, environment='asia-southeast1-gcp-free')
index_name = 'dbpaseg'


os.environ['OPENAI_API_KEY'] = st.secrets['openai_api_key']

#st.set_page_config(page_title="PASEG Genie", page_icon=":coffee:")

# Load Pinecone API key
api_key = st.secrets["pinecone_api_key"]
pinecone.init(api_key=api_key, environment='asia-southeast1-gcp-free')
index_name = 'dbpaseg'

# Create a function to load embeddings and Pinecone client
@st.cache(allow_output_mutation=True)  # Allow output mutation for the Pinecone client
def load_embeddings_and_pinecone():
    embeddings = HuggingFaceEmbeddings()
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    return docsearch

# Create the Chat and RetrievalQA objects outside the Streamlit app function
chat = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0.80)
qachain = load_qa_chain(chat, chain_type='stuff')
qa = RetrievalQA(combine_documents_chain=qachain)

# Load the Pinecone client using st.cache
docsearch = load_embeddings_and_pinecone()

condition1 = '\n [organize information: organize text so its easy to read, and bullet points when needed.] \n [tone and voice style: clear sentences, avoid use of complex sentences]'

st.title("PASEG Genie // Donate a Coffee :coffee:")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if query := st.chat_input("Enter your query:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        full_response = ""
        for response in chat.send_message(query, docsearch.as_retriever()):
            full_response += response + "\n"
            st.markdown(full_response + "▌")
    st.session_state.messages.append({"role": "assistant", "content": full_response})

