import streamlit as st
from crewai import Agent
# from crewai_tools import PDFSearchTool
import os
from crewai import Task
from crewai import Crew, Process
from crewai_tools import tool
import pandas as pd
import chromadb
from pydantic import BaseModel
import tempfile
import shutil
import xlsxwriter
import base64
from typing import Any
import PyPDF2
from io import BytesIO



# API Key (Make sure this is securely handled in your actual application)
api_key = st.secrets["OPENAI_API_KEY"]

st.title("Courantchecker")

doc_1 = st.file_uploader("Doc1", type="pdf")
doc_2 = st.file_uploader("Doc2", type="pdf")

if doc_1 and doc_2:
    # Custom tool to read PDFs
    @tool("custom_pdf_reader_tool")
    def custom_pdf_reader_tool(pdf_file):
        """Extract text from a PDF file using PyPDF2."""
        try:
            reader = PyPDF2.PdfReader(BytesIO(pdf_file.getvalue()))
            text = [page.extract_text() for page in reader.pages]
            return "\n".join(text)
        except Exception as e:
            return f"Failed to process PDF: {str(e)}"

    # Tool to convert text to DataFrame
    @tool("panda_dataframe_tool")
    def panda_dataframe_tool(question: str) -> pd.DataFrame:
        text = " ".join(question.split())
        rows = text.split("\n")
        df = pd.DataFrame([row.split("\t") for row in rows])
        df.set_index(df.columns[0], inplace=True)
        df.drop(df.columns[0], axis=1, inplace=True)
        return df

    # Tool to convert DataFrame to Excel
    @tool("pandas_to_excel_tool")
    def pandas_to_excel_tool(question: str, df: Any) -> str:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        writer = pd.ExcelWriter(tmp_file.name, engine='xlsxwriter')
        df.to_excel(writer, index=False)
        writer.save()
        with open(tmp_file.name, "rb") as f:
            file_bytes = f.read()
        file_b64 = base64.b64encode(file_bytes).decode()
        shutil.rmtree(os.path.dirname(tmp_file.name))
        return f"<a href='data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{file_b64}' download='file.xlsx'>Download Excel file</a>"

    # Agents defined with explicit roles and goals
    pdf_reader = Agent(
        role='PDF Reader',
        goal='Extract text from PDF files for processing.',
        verbose=True,
        tools=[custom_pdf_reader_tool],
        allow_delegation=False
    )

    transaction_organiser = Agent(
        role='Transaction Organiser',
        goal='Organise extracted data into structured form.',
        verbose=True,
        tools=[panda_dataframe_tool],
        allow_delegation=False
    )

    dataframe_comparer = Agent(
        role='DataFrame Comparer',
        goal='Identify differences between dataframes.',
        verbose=True,
        tools=[compare_dataframe_tool],
        allow_delegation=False
    )

    excel_bookkeeper = Agent(
        role='Excel Bookkeeper',
        goal='Convert dataframes to Excel format for download.',
        verbose=True,
        tools=[pandas_to_excel_tool],
        allow_delegation=False
    )

    # Form the crew
    crew = Crew(
        agents=[pdf_reader, transaction_organiser, dataframe_comparer, excel_bookkeeper],
        process=Process.sequential,
        memory=False
    )

    if st.button("Start Processing"):
        doc1_bytes = BytesIO(doc_1.getvalue())
        doc2_bytes = BytesIO(doc_2.getvalue())
        result = crew.kickoff(inputs={'doc1': doc1_bytes, 'doc2': doc2_bytes})
        if isinstance(result, str):
            st.success(result)
        elif isinstance(result, bytes):
            st.success("The Excel file is ready to download")
            href = f"<a href='data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{base64.b64encode(result).decode()}' download='comparison.xlsx'>Download Excel file</a>"
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.error("Something went wrong")
else:
    st.error("Please upload both PDF documents to proceed.")