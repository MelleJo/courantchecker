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

def extract_text_from_pdf_by_page(file):
    pages_text = []
    reader = PdfReader(file)
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    return pages_text

def process_document(file1, file2):
    with st.spinner('Denken...'):
        # Check if files are uploaded
        if not file1 or not file2:
            st.error("Please upload both documents.")
            return

        # Extract text from the documents
        document_pages_doc1 = extract_text_from_pdf_by_page(file1)
        document_pages_doc2 = extract_text_from_pdf_by_page(file2)

        if not document_pages_doc1 or all(page.strip() == "" for page in document_pages_doc1):
            st.error("No valid text extracted from document 1. Please check the document format or content.")
            return
        if not document_pages_doc2 or all(page.strip() == "" for page in document_pages_doc2):
            st.error("No valid text extracted from document 2. Please check the document format or content.")
            return

        document_text_doc1 = " ".join(document_pages_doc1)
        document_text_doc2 = " ".join(document_pages_doc2)

        template = """
        Je bent een expert boekhouder. Je controleert de rekening couranten op discrepensies, je controleert daarbij de credit en debet van document1 = {document_text_doc1} en document2 = {document_text_doc2}. Deze vergelijk je met elkaar en je komt met een lijst van de exacte verschillen.
        Dit geef je zo duidelijk mogelijk weer per polisnummer. Als je bepaalde matches al op slimme wijze kunt maken stel je deze voor.
        Prioriteit nummer één is de nauwkeurigheid en volledigheid. 
        Je antwoord bestaat uit een lijst van alle discrepensies.
        Je geeft geen stappenlijst of iets dergelijks, je produceert gewoon direct een lijst met alle polisnummers en de discrepensies. 
        Je hoeft geen uitleg te geven. Alleen de lijst. Je geeft alleen maar de polisnummers waarbij een discrepensie is en de beschrijving van de discrepensie. 
        Absoluut geen tekst om de lijst heen. Enkel de lijst, en dan per discrepensie wel kijken of je een mogelijke uitleg kunt vinden. 
        De bedragen 1,05 en 2,10 mag je negeren, deze hebben een reden (assurantiebelasting).
        Gebruik wel een duidelijke weergave van de verschillen, dus bijvoorbeeld als een tabel.
        """
        
        prompt = ChatPromptTemplate.from_template(template)

        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4-turbo-2024-04-09", temperature=0, streaming=True)
        chain = prompt | llm | StrOutputParser()
        return chain.stream({
            "document_text_doc1": document_text_doc1,
            "document_text_doc2": document_text_doc2
        })

def main():
    st.title("Courantchecker - testversie 0.0.1")

    if st.button('Start vergelijken'):
        answer = process_document(upload_doc1, upload_doc2)
        st.write(answer)


if __name__ == "__main__":
    main()
