import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import chromadb

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = os.getcwd()
DOCS_DIR = os.path.join(BASE_DIR, "documents")
DB_DIR = os.path.join(BASE_DIR, "db")
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Hugging Face API Key
HF_API_KEY = os.getenv("HF_API_KEY")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Initialize LLM
@st.cache_resource
def initialize_llm():
    return HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
        huggingfacehub_api_token=HF_API_KEY,
        temperature=0.8,
        task="text-generation"
    )

# Initialize embeddings
@st.cache_resource
def initialize_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# Initialize UI
st.title("PDF AI Assistant")

# Initialize components
llm = initialize_llm()
embedding_function = initialize_embeddings()

# PDF processing using recursive character text splitter to split text into chunks with overlap
# This is to ensure that the model can retrieve relevant context for the user's question
@st.cache_resource
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    return text_splitter.split_documents(documents)

# vector store for each new document
# can use st.cache_resource to cache the vectorstore (choosing not to for now)
def create_new_vectorstore(docs):
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_function,
        persist_directory=DB_DIR
    )
    vectorstore.persist()
    return vectorstore

# File upload handling
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    file_path = os.path.join(DOCS_DIR, uploaded_file.name)
    with open(file_path, "wb") as buffer:
        buffer.write(uploaded_file.read())
    
    docs = process_pdf(file_path)
    vectorstore = create_new_vectorstore(docs)

    st.success("PDF uploaded and processed successfully!")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) #retrieve top 5 most similar

    # Prompt template
    qa_prompt = PromptTemplate(
        template="""Use the following context to answer the question. 
        If you don't know the answer, just say you don't know.
        
        Context: {context}
        Question: {question}
        
        Answer:""",
        input_variables=["context", "question"]
    )

    # retrieval chain
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt, "verbose": True}
    )

    # User input and response handling
    question = st.text_input("Ask a question about the document:")

    if question:
        try:
            response = retrieval_chain.invoke({"query": question})

            # Debugging: Show context and question
            #st.write(f"Context: {response.get('source_documents')}")
            st.write(f"Question: {question}")

            # Handle case when no context is found
            if not response.get("source_documents"):
                st.write("Sorry, I couldn't find any relevant information in the document.")
            else:
                st.subheader("Answer:")
                st.write(response["result"])

        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            st.info("Please try rephrasing your question or uploading a different document.")
else:
    st.info("Please upload a PDF to begin.")
