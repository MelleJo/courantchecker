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




# API Key from Streamlit secrets
api_key = st.secrets["OPENAI_API_KEY"]

st.title("Courantchecker")

doc_1 = st.file_uploader("Doc1", type="pdf")
doc_2 = st.file_uploader("Doc2", type="pdf")

if doc_1 and doc_2:
    # Convert uploaded files to BytesIO
    doc1_bytes = BytesIO(doc_1.getvalue())
    doc2_bytes = BytesIO(doc_2.getvalue())

    # Define custom tool to extract text from PDF
    @tool("custom_pdf_reader_tool")
    def custom_pdf_reader_tool(pdf_file):
        """
        Custom tool to extract text from a PDF file uploaded by the user.
        Args:
        pdf_file (BytesIO): A stream containing the PDF file data.
        Returns:
        str: Extracted text from all pages of the PDF or an error message if the extraction fails.
        """
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            text = [page.extract_text() for page in reader.pages]
            return "\n".join(text)
        except Exception as e:
            return f"Failed to process PDF: {str(e)}"

    # Data processing tools
    @tool("panda_dataframe_tool")
    def panda_dataframe_tool(question: str) -> pd.DataFrame:
        """Converts a block of text to a structured pandas DataFrame."""
        text = " ".join(question.split())
        rows = text.split("\n")
        df = pd.DataFrame([row.split("\t") for row in rows])
        df.set_index(df.columns[0], inplace=True)
        df.drop(df.columns[0], axis=1, inplace=True)
        return df

    @tool("pandas_to_excel_tool")
    def pandas_to_excel_tool(question: str, df: pd.DataFrame) -> str:
        """Converts a pandas DataFrame into a downloadable Excel file link."""
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        writer = pd.ExcelWriter(tmp_file.name, engine='xlsxwriter')
        df.to_excel(writer, index=False)
        writer.save()
        with open(tmp_file.name, "rb") as f:
            file_bytes = f.read()
        file_b64 = base64.b64encode(file_bytes).decode()
        shutil.rmtree(os.path.dirname(tmp_file.name))
        return f"<a href='data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{file_b64}' download='file.xlsx'>Download Excel file</a>"

    @tool("compare_dataframe_tool")
    def compare_dataframe_tool(question: str, reference_df: pd.DataFrame) -> list:
        """Compares two dataframes and returns differences."""
        text = " ".join(question.split())
        rows = text.split("\n")
        df = pd.DataFrame([row.split("\t") for row in rows])
        df.set_index(df.columns[0], inplace=True)
        df.drop(df.columns[0], axis=1, inplace=True)
        return list(reference_df.compare(df).dropna())

    # Define agents
    pdf_reader = Agent(
        role='PDF Reader',
        goal='Extract all transactions from the uploaded PDF files {doc1} and {doc2}.',
        tools=[custom_pdf_reader_tool],
        allow_delegation=False,
        verbose=True,
        memory=True
    )

    transaction_organiser = Agent(
        role='Transaction Organiser',
        goal='Organise the transactions into a structured pandas DataFrame.',
        tools=[panda_dataframe_tool],
        allow_delegation=False,
        verbose=True,
        memory=True
    )

    dataframe_comparer = Agent(
        role='DataFrame Comparer',
        goal='Highlight discrepancies between the dataframes.',
        tools=[compare_dataframe_tool],
        allow_delegation=False,
        verbose=True,
        memory=True
    )

    excel_bookkeeper = Agent(
        role='Excel Bookkeeper',
        goal='Export the transaction list to an Excel file in a readable format.',
        tools=[pandas_to_excel_tool],
        allow_delegation=False,
        verbose=True,
        memory=True
    )

    # Form the crew
    crew = Crew(
        agents=[pdf_reader, transaction_organiser, dataframe_comparer, excel_bookkeeper],
        tasks=[
            Task(description="Extract transactions", expected_output="Dataframe", agent=transaction_organiser),
            Task(description="Compare transactions", expected_output="List of discrepancies", agent=dataframe_comparer),
            Task(description="Export transactions to Excel", expected_output="Excel file", agent=excel_bookkeeper)
        ],
        process=Process.sequential,
        memory=True,
        cache=True,
        max_rpm=100,
        share_crew=True
    )

    # Start processing
    if st.button("Start Processing"):
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
    st.warning("Please upload both PDF documents to start processing.")