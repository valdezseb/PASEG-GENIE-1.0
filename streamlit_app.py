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


st.set_page_config(page_title="PASEG Genie", page_icon=":coffee:", layout="centered")

# Load Pinecone API key
api_key = st.secrets["pinecone_api_key"]
pinecone.init(api_key=api_key, environment='asia-southeast1-gcp-free')
#index_name = 'dbpaseg'


os.environ['OPENAI_API_KEY'] = st.secrets['openai_api_key']

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#st.set_page_config(page_title="PASEG Genie", page_icon=":coffee:")

# Load Pinecone API key
api_key = st.secrets["pinecone_api_key"]
pinecone.init(api_key=api_key, environment='asia-southeast1-gcp-free')
#index_name = 'db-paseg'

# Define Pinecone index name
index_name = 'db-paseg'

# Define Streamlit app
#st.set_page_config(page_title="PASEG Genie ", page_icon=":coffee:", layout="wide")

# Define username and password
username = "ppca"
password = "65326"

# Define Streamlit sidebar for login
st.sidebar.title("Login")
login_username = st.sidebar.text_input("Username")
login_password = st.sidebar.text_input("Password", type="password")
#os.environ['OPENAI_API_KEY'] = st.sidebar.text_input("Enter your Open AI API Key")
# Define Streamlit main page
st.title("PASEG Genie // for education purpose :coffee:")
st.markdown("*Chat With The Planning and Schedule Excellence Guide ver. 5.0*", unsafe_allow_html=True)
st.markdown("---")

# Define function to check login credentials
def check_login(username, password):
    if username == "ppca" and password == "65326":
        return True
    else:
        return False

# Check login credentials and show query input if login is successful
if check_login(login_username, login_password):
    st.success("Login successful!")
    query = st.text_input("Enter your query:")
    if query:
        # Load embeddings and Pinecone client
        @st.cache_resource
        def load_embedding():
            embeddings = HuggingFaceEmbeddings()
            return embeddings

        embeddings = load_embedding()

        def load_pinecone(embeddings, index_name):
            docsearch = Pinecone.from_existing_index(index_name, embeddings)
            return docsearch

        docsearch = load_pinecone(embeddings, index_name)

        # Create the Chat and RetrievalQA objects
        chat = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0.80)
        qachain = load_qa_chain(chat, chain_type='stuff')
        qa = RetrievalQA(combine_documents_chain=qachain, retriever=docsearch.as_retriever())

        condition1 = '\n [Generate Response/Text from my data.]  \n [organize information: organize text so its easy to read, and bullet points when needed.] \n [if applicable for the question response, add section: Things to Promote/Things to Avoid and Best Practices, give Examples] \n [tone and voice style: clear sentences, avoid use of complex sentences]'

        # Run the QA system and display the result using Streamlit
        result = qa.run(query + '\n' + condition1)
        st.write(result)
else:
    st.write("Login failed. Please check your credentials.")
