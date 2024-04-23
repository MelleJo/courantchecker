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

upload_doc1 = st.file_uploader("Upload document 1", type="pdf")
upload_doc2 = st.file_uploader("Upload document 2", type="pdf")

def extract_text_from_pdf_by_page_doc1(file_path):
    pages_text_doc1 = []
    with open(upload_doc1, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text_doc1.append(text)
    return pages_text_doc1

def extract_text_from_pdf_by_page_doc2(file_path):
    pages_text_doc2 = []
    with open(upload_doc1, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text_doc2.append(text)
    return pages_text_doc2


def process_document(upload_doc1, upload_doc2, user_question):
    with st.spinner('Denken...'):
        # Extract text from the document
        document_pages_doc1 = extract_text_from_pdf_by_page_doc1(document_path)
        if not document_pages_doc1 or all(page.strip() == "" for page in document_pages_doc1):
        
            st.error("No valid text extracted from the document 1. Please check the document format or content.")
            return

        document_pages_doc2 = extract_text_from_pdf_by_page_doc2(document_path)
        if not document_pages_doc2 or all(page.strip() == "" for page in document_pages_doc2):

            st.error("No valid text extracted from the document 2. Please check the document format or content.")
            return
        
        document_text_doc1 = " ".join(document_pages_doc1)
        document_text_doc2 = " ".join(document_pages_doc2)
    

        template = """
        Je bent een expert boekhouder. Je controleert de rekening couranten op discrepensies, je controleert daarbij de credit en debet van document1 = {document_text_doc1} en document2 = {document_text_doc2}. Deze vergelijk je met elkaar en je komt met een lijst van de exacte verschillen.
        Dit geef je zo duidelijk mogelijk weer per polisnummer. Als je bepaalde matches al op slimme wijze kunt maken stel je deze voor.
        Prioriteit nummer één is de nauwkeurigheid en volleheid.
        """
        
        prompt = ChatPromptTemplate.from_template(template)

        
        # Perform similarity search
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo-2024-04-09", temperature=0, streaming=True)
        chain = prompt | llm | StrOutputParser() 
        return chain.stream({
            "document_text_doc1": document_text_doc1,
            "document_text_doc2": document_text_doc2,
            "user_question": user_question,
        })

    

    


    

def main():
    st.title("Courantchecker - testversie 0.0.1")
    
    user_question = st.text_input("Wat wil je graag weten?")

    if user_question:
        answer = process_document(upload_doc1, upload_doc2, user_question)
        st.write(answer)

if __name__ == "__main__":
    main()
