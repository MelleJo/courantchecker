import streamlit as st
import fitz  # PyMuPDF voor het lezen van PDF's
import pandas as pd
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from langchain_groq import ChatGroq
from langchain.llms import GroqLLM  # Voor het gebruik van Groq LLM

# Configuratie van API sleutels
API_KEY = st.secrets["GROQ_API_KEY"]

# Initialisatie van Groq LLM
groq_llm = GroqLLM(api_key=API_KEY)

# Node functies
def extract_text_from_pdf(state):
    file_path = state["file_path"]
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    state["extracted_text"] = text
    return state

def extract_data(state):
    text = state["extracted_text"]
    lines = text.split('\n')
    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 2:  # Pas aan op basis van je PDF structuur
            data.append(parts)
    df = pd.DataFrame(data, columns=['Datum', 'Debet', 'Credit'])  # Pas kolomnamen aan
    state["dataframe"] = df
    return state

def compare_data(state):
    df1 = state["dataframe1"]
    df2 = state["dataframe2"]
    comparison = df1.compare(df2)
    state["comparison"] = comparison
    return state

def save_results(state):
    comparison = state["comparison"]
    comparison.to_excel("output_differences.xlsx")
    state["output_file"] = "output_differences.xlsx"
    return state

# LangGraph opzet
graph = LangGraph()
graph.add_node("extract_pdf1", Node(function=extract_text_from_pdf))
graph.add_node("extract_data1", Node(function=extract_data))
graph.add_node("extract_pdf2", Node(function=extract_text_from_pdf))
graph.add_node("extract_data2", Node(function=extract_data))
graph.add_node("compare", Node(function=compare_data))
graph.add_node("save", Node(function=save_results))

# Edges configureren
graph.add_edge("extract_pdf1", "extract_data1")
graph.add_edge("extract_data1", "extract_pdf2")
graph.add_edge("extract_pdf2", "extract_data2")
graph.add_edge("extract_data2", "compare")
graph.add_edge("compare", "save")

# Streamlit UI
st.title('Rekening Courant Document Vergelijker')
uploaded_files = st.file_uploader("Upload twee PDF-documenten", accept_multiple_files=True, type=["pdf"])
if uploaded_files and len(uploaded_files) == 2:
    state = {
        "file_path": uploaded_files[0],
        "file_path2": uploaded_files[1]
    }
    result = graph.run(state)  # Graph uitvoeren
    st.success('De verschillen zijn geanalyseerd en opgeslagen.')
    st.download_button('Download Resultaten', result["output_file"])
