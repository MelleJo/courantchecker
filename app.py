import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def clean_extracted_text(text):
    # Correct common spacing issues in PDF text extraction
    return " ".join(text.split())

st.header("Rekening courant checker")
uploaded_files = st.file_uploader("Upload pdf's", accept_multiple_files=True)

if uploaded_files is not None and len(uploaded_files) >= 2:
    file1, file2 = PdfReader(uploaded_files[0]), PdfReader(uploaded_files[1])
    title1 = file1.metadata.get('/Title', 'Unknown Title for Document 1')
    title2 = file2.metadata.get('/Title', 'Unknown Title for Document 2')
    st.write(f"Document 1: {title1}\nDocument 2: {title2}")

    text1, text2 = "", ""
    for page in file1.pages:
        text1 += clean_extracted_text(page.extract_text()) + "\n"
    for page in file2.pages:
        text2 += clean_extracted_text(page.extract_text()) + "\n"

    def process_document(user_question):
        template = """..."""  # Your detailed prompt template here
        
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-0125-preview", temperature=0, streaming=True)
        chain = prompt | llm | StrOutputParser()
        
        document_stream = chain.stream({
            "user_question": user_question,
            "document1_text": text1,
            "document2_text": text2
        })

        full_response = ""
        for response in document_stream:
            full_response += response + " "
        return clean_extracted_text(full_response)

    st.title("Courant Checker")
    user_question = st.text_input("Stel een vraag over de documenten:")
    if user_question:
        consolidated_response = process_document(user_question)
        st.write(consolidated_response)
else:
    st.error("Please upload at least two PDF files.")
