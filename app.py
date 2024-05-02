import streamlit as st
import pandas as pd
import fitz  # PyMuPDF library
from langgraph import StateGraph, Tool, tool
from langchain.agents import OpenAIChatAgent
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Annotated, Sequence, TypedDict
import json

# API keys and environment variables (ensure to define these in your Streamlit secrets or environment variables)
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Define tool for reading PDFs
@tool
def read_pdf(pdf_file: Annotated[bytes, "The PDF file to extract text from"]):
    """Read text from a PDF file."""
    try:
        doc = fitz.open("pdf", pdf_file)
        text = ''
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"Failed to read PDF. Error: {repr(e)}"

# Define tool for executing Python code
@tool
def execute_python(code: Annotated[str, "The python code to execute to generate your chart."]):
    """Execute Python code and return output."""
    try:
        local_vars = {}
        exec(code, globals(), local_vars)
        return local_vars['result']
    except Exception as e:
        return f"Failed to execute Python code. Error: {repr(e)}"

# Define state for the application
class AgentState(TypedDict):
    text1: str
    text2: str
    comparison_result: pd.DataFrame

# Create agents using OpenAI's GPT-4
def create_agent(tools, system_message: str):
    """Function to create an agent with specified tools."""
    llm = OpenAIChatAgent(api_key=OPENAI_API_KEY, tools=tools, model="gpt-4")
    prompt = ChatPromptTemplate(
        system="You are a helpful AI assistant.",
        user=MessagesPlaceholder()
    )
    return llm.bind_prompt(prompt)

# Define the workflow
workflow = StateGraph()

# Node for extracting text from first PDF
@workflow.node(name="extract_text1")
def extract_text1(state, pdf_file):
    agent = create_agent([read_pdf], "Extract text from the first PDF")
    state['text1'] = agent.run_tool(read_pdf, pdf_file)
    return state

# Node for extracting text from second PDF
@workflow.node(name="extract_text2")
def extract_text2(state, pdf_file):
    agent = create_agent([read_pdf], "Extract text from the second PDF")
    state['text2'] = agent.run_tool(read_pdf, pdf_file)
    return state

# Node for comparing texts
@workflow.node(name="compare_texts")
def compare_texts(state):
    compare_code = """
result = {'differences': [str(item) for item in set(text1.split()) ^ set(text2.split())]}
"""
    agent = create_agent([execute_python], "Compare the extracted texts")
    state['comparison_result'] = agent.run_tool(execute_python, compare_code.format(text1=state['text1'], text2=state['text2']))
    return state

# Node for exporting to Excel
@workflow.node(name="export_excel")
def export_excel(state):
    excel_code = """
import pandas as pd
df = pd.DataFrame({'Differences': comparison_result['differences']})
df.to_excel('output.xlsx')
result = 'output.xlsx'
"""
    agent = create_agent([execute_python], "Export the results to an Excel file")
    state['excel_file'] = agent.run_tool(execute_python, excel_code.format(comparison_result=state['comparison_result']))
    return state

# Add edges and conditions to the workflow
workflow.add_edge("extract_text1", "extract_text2")
workflow.add_edge("extract_text2", "compare_texts")
workflow.add_edge("compare_texts", "export_excel")

# Streamlit interface for uploading PDFs and downloading the result
st.title('PDF Comparison Tool')
uploaded_files = st.file_uploader("Upload two PDF documents", type=["pdf"], accept_multiple_files=True)

if st.button('Process Files') and uploaded_files and len(uploaded_files) == 2:
    initial_state = {}
    final_state = workflow.run(initial_state, pdf_file1=uploaded_files[0], pdf_file2=uploaded_files[1])
    st.success('Processing complete. Download your results below.')
    with open(final_state['excel_file'], 'rb') as file:
        st.download_button('Download Excel File', file, file_name='comparison_results.xlsx')
