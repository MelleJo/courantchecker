import streamlit as st
import os 
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import AnalyzeDocumentChain
from langchain_community.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


st.header("Rekening courant checker")
uploaded_files = st.file_uploader("Upload pdf's", accept_multiple_files=True)
if uploaded_files is not None:
    file1 = PdfReader(uploaded_files[0])
    file2 = PdfReader(uploaded_files[1])
    st.write(f"Bestand 1: {file1.getDocumentInfo().title}\nBestand 2: {file2.getDocumentInfo().title}")
    st.write("Voer de query uit, en de tool zal de twee bestanden vergelijken.")




