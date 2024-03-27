import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

st.header("Rekening courant checker")
uploaded_files = st.file_uploader("Upload pdf's", accept_multiple_files=True)

# Check if at least two files have been uploaded
if uploaded_files is not None and len(uploaded_files) >= 2:
    file1 = PdfReader(uploaded_files[0])
    file2 = PdfReader(uploaded_files[1])
    st.write(f"Bestand 1: {file1.getDocumentInfo().title}\nBestand 2: {file2.getDocumentInfo().title}")
    st.write("Voer de query uit, en de tool zal de twee bestanden vergelijken.")

    text1 = ""
    for page in file1.pages:
        text1 += page.extractText() + "\n"

    text2 = ""
    for page in file2.pages:
        text2 += page.extractText() + "\n"

    def process_document(user_question):
        document1_text = text1
        document2_text = text2

        template = """
        Jij bent een expert boekhouder en accountant. Je beantwoord de {user_question} over de volgende documenten: {document1_text} en {document2_text}.
        Hierbij stel je nauwkeurigheid en volledigheid op prioriteit nummer 1. 
        """
        prompt = ChatPromptTemplate.from_template(template)

        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-0125-preview", temperature=0, streaming=True)
        chain = prompt | llm | StrOutputParser()
        return chain.stream({
            "document_text": document1_text + document2_text,
            "user_question": user_question
        })

    st.title("Courant Checker")
    user_question = st.text_input("Stel een vraag over de documenten:")
    if user_question:
        document_stream = process_document(user_question)
        for response in document_stream:
            st.write(response)
else:
    st.error("Please upload at least two PDF files.")
