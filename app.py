import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import re

st.header("Rekening Courant Checker")

uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True)

# Improved text extraction function with normalization
def normalize_text(text):
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Optional: Additional normalization logic can be added here
    return text.strip()


if uploaded_files and len(uploaded_files) == 2:
    file1, file2 = PdfReader(uploaded_files[0]), PdfReader(uploaded_files[1])
    text1, text2 = "", ""
    for page in file1.pages:
        raw_text = page.extract_text()
        if raw_text:  # Check if text was extracted
            text1 += normalize_text(raw_text) + "\n"

    for page in file2.pages:
        raw_text = page.extract_text()
        if raw_text:  # Check if text was extracted
            text2 += normalize_text(raw_text) + "\n"

    def process_document(user_question, text1, text2):
        
        template = f"""
        Jij bent een expert boekhouder en bent getraind op het herkennen van bedragen die niet overeenkomen in de rekening courant.

        De gebruiker stelt een vraag die je beantwoord: "{user_question}"

        Document 1:
        {text1}

        Document 2:
        {text2}

        Nadat je de vraag hebt beantwoord, zorg je dat je het antwoord aanlevert in een format dat duidelijk te lezen is zonder gekke spacing tussen de woorden. Dit dubbelcheck je.
        """

        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-0125-preview", temperature=0, streaming=True)
        chain = prompt | llm | StrOutputParser()
        
        responses = chain.stream({
            "user_question": user_question
        })

        full_response = ""
        for response in responses:
            full_response += response + " "
        return full_response.strip()

    user_question = st.text_input("Stel een vraag over de documenten:")
    
    if user_question:
        response = process_document(user_question, text1, text2)
        st.write(response)
else:
    st.error("Please upload exactly two PDF files.")
