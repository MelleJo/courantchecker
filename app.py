import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.graph import END, MessageGraph
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
from PyPDF2 import PdfReader


# API keys
openai_api_key = st.secrets["OPENAI_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]

GROQ_LLM = ChatGroq(
    model="llama3-70b-8192", api_key=groq_api_key, temperature=0
)

st.title("COURANT CHECKER")
doc_1 = st.file_uploader("Upload Document 1", type="pdf")
doc_2 = st.file_uploader("Upload Document 2", type="pdf")


def extract_pdf_text(pdf_file1, pdf_file2):
    reader1 = PdfReader(pdf_file1)
    reader2 = PdfReader(pdf_file2)
    text_doc1 = ""
    text_doc2 = ""
    for page in reader1.pages:
        text_doc1 += page.extractText()
    for page in reader2.pages:
        text_doc2 += page.extractText()
    return text_doc1, text_doc2


# text -> tabel
prompt = PromptTemplate(
    template="""
    Jij bent een expert in het omzetten van de tekst van het pdf bestand in een net Pandas dataframe, je zorgt ervoor dat het netjes per polisnummer is gesorteerd en dat alle transacties worden opgenomen in de tabel.
    Je gebruikt daarvoor {text_doc1} en {text_doc2}, je geeft duidelijk aan bij elke transactie van welk document het afkomstig is. 

    """,
)

extract_structure_agent = prompt | GROQ_LLM | StrOutputParser()

text_doc1, text_doc2 = extract_pdf_text(doc_1, doc_2)

result = extract_structure_agent.invoke({"text_doc1": text_doc1, "text_doc2": text_doc2})

st.write(result)
