import streamlit as st 
import os
import base64
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline
import torch 
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA 
from constants import CHROMA_SETTINGS
from streamlit_chat import message
from chromadb import Client 

st.set_page_config(layout="wide")

device = torch.device('cpu')

checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

persist_directory = "db"

# Initialize the Chroma client globally
client = Client()
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client=client)

@st.cache_resource
def data_ingestion():
    # Initialize the Chroma client
    client = Client()
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(f"Processing file: {file}")  # Debugging
                loader = PDFMinerLoader(os.path.join(root, file))
                documents = loader.load()
                print(f"Loaded documents: {[doc.page_content for doc in documents]}")  # Debugging
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
                texts = text_splitter.split_documents(documents)
                # Create embeddings here
                embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client=client)
                db.persist()
                db = None # Persist after all documents have been added

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
        device=device
    )
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource
def qa_llm():
    retriever = db.as_retriever()  # Use the existing Chroma instance
    llm = llm_pipeline()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

def process_answer(instruction):
    qa = qa_llm()  # Ensure you're calling the QA instance correctly
    generated_text = qa({"query": instruction})  # Pass the input correctly
    return generated_text['result']

def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size

@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))

def main():
    st.markdown("<h1 style='text-align: center; color: blue;'>Chat with your PDF 🦜📄 </h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: grey;'>Built by <a href='https://github.com/indrajeet0906'>INDRAJEET ❤️ </a></h3>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color:red;'>Upload your PDF 👇</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["pdf"])

    if uploaded_file is not None:
        if not os.path.exists("docs"):
            os.makedirs("docs")
    
        file_details = {
            "Filename": uploaded_file.name,
            "File size": get_file_size(uploaded_file)
        }
        filepath = "docs/" + uploaded_file.name
        
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("<h4 style color:black;'>File details</h4>", unsafe_allow_html=True)
            st.json(file_details)
            st.markdown("<h4 style color:black;'>File preview</h4>", unsafe_allow_html=True)
            displayPDF(filepath)
        
        with col2:
            with st.spinner('Embeddings are in process...'):
                data_ingestion()  # Only call this once
            st.success('Embeddings are created successfully!')
            st.markdown("<h4 style color:black;'>Chat Here</h4>", unsafe_allow_html=True)

            user_input = st.text_input("", key="input")

            if "generated" not in st.session_state:
                st.session_state["generated"] = ["I am ready to help you"]
            if "past" not in st.session_state:
                st.session_state["past"] = ["Hey there!"]
            
            if user_input:
                 print(f"User input: {user_input}")  # Debugging line
                 answer = process_answer(user_input)
                 print(f"Generated answer: {answer}")  # Debugging line
                 st.session_state["past"].append(user_input)
                 st.session_state["generated"].append(answer)

            if st.session_state["generated"]:
                display_conversation(st.session_state)

if __name__ == "__main__":
    main()




