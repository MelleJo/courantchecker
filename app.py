import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import re
import pytesseract
from PIL import Image
import pdf2image

st.header("Rekening Courant Checker")

uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True)

def normalize_text(text):
    # Normalize whitespace and strip leading/trailing whitespace
    return re.sub(r'\s+', ' ', text).strip()

def extract_text_with_ocr(pdf_path):
    # Convert PDF page to image and use pytesseract to extract text
    images = pdf2image.convert_from_path(pdf_path)
    text = " ".join([pytesseract.image_to_string(img) for img in images])
    return normalize_text(text)

def extract_or_ocr_pdf(pdf_reader):
    # Attempt to extract text from PDF, use OCR if necessary
    extracted_text = ""
    for page in pdf_reader.pages:
        raw_text = page.extract_text()
        if raw_text:
            extracted_text += normalize_text(raw_text) + "\n"
    return extracted_text.strip() if extracted_text.strip() else extract_text_with_ocr(pdf_reader.stream.name)

if uploaded_files and len(uploaded_files) == 2:
    text1 = extract_or_ocr_pdf(PdfReader(uploaded_files[0]))
    text2 = extract_or_ocr_pdf(PdfReader(uploaded_files[1]))

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

        responses = chain.stream({"user_question": user_question})
        return " ".join(responses).strip()

    user_question = st.text_input("Stel een vraag over de documenten:")
    
    if user_question:
        response = process_document(user_question, text1, text2)
        st.write(response)
else:
    st.error("Please upload exactly two PDF files.")
