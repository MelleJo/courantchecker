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

    text1 = ""
    for page in file1.pages:
        text1 += CharacterTextSplitter.split(page.extractText()) + "\n"

    text2 = ""
    for page in file2.pages:
        text2 += CharacterTextSplitter.split(page.extractText()) + "\n"


def process_document(uploaded_files, user_question):
    document1_text = text1
    document2_text = text2
    
    template = """

    Jij bent een expert boekhouder en accountant. Je beantwoord de {user_question} over de volgende documenten: {document1_text} en {document2_text}.
    Hierbij stel je nauwkeurigheid en volledigheid op prioriteit nummer 1. 
    
    """
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model = "gpt-4-0125-preview", temperature=0, streaming=True)
    chain = prompt | llm | StrOutputParser()
    return chain.stream({
        "document_text": document1_text + document2_text,
        "user_question": user_question
    })


def main():
    st.title("Courant Checker")
    selected_document = uploaded_files(file1, file2)
    if selected_document:
        user_question = st.text_input("Stel een vraag over de polisvoorwaarden:")
        if user_question:
            document_stream = process_document(selected_document['path'], user_question)
            st.write_stream(document_stream)  

if __name__ == "__main__":
    main()