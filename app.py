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




# API Key
api_key = st.secrets["OPENAI_API_KEY"]

st.title("Courantchecker")

doc_1 = st.file_uploader("Doc1", type="pdf")
doc_2 = st.file_uploader("Doc2", type="pdf")

if doc_1 and doc_2:
    @tool("custom_pdf_reader_tool")
    def custom_pdf_reader_tool(pdf_file: BytesIO) -> str:
        """
        Extract text from a PDF file using PyPDF2.

        Args:
            pdf_file (BytesIO): The PDF file in memory to be read.

        Returns:
            str: Extracted text from all pages of the PDF.
        """
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            text = [page.extract_text() for page in reader.pages if page.extract_text() is not None]
            return "\n".join(text)
        except Exception as e:
            return f"Failed to process PDF: {str(e)}"

    @tool("panda_dataframe_tool")
    def panda_dataframe_tool(text: str) -> pd.DataFrame:
        """
        Convert extracted text into a structured pandas DataFrame.

        Args:
            text (str): Text containing tab-separated values.

        Returns:
            pd.DataFrame: Data organized as a DataFrame.
        """
        rows = text.split("\n")
        df = pd.DataFrame([row.split("\t") for row in rows])
        df.set_index(df.columns[0], inplace=True)
        df.drop(df.columns[0], axis=1, inplace=True)
        return df

    @tool("pandas_to_excel_tool")
    def pandas_to_excel_tool(df: pd.DataFrame) -> str:
        """
        Save DataFrame to an Excel file and provide a download link.

        Args:
            df (pd.DataFrame): Data to be written to Excel.

        Returns:
            str: A hyperlink to download the Excel file.
        """
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
    def compare_dataframe_tool(df1: pd.DataFrame, df2: pd.DataFrame) -> list:
        """
        Compare two DataFrames and return the differences.

        Args:
            df1 (pd.DataFrame): The first DataFrame.
            df2 (pd.DataFrame): The second DataFrame to compare against.

        Returns:
            list: Differences between the two DataFrames.
        """
        return list(df1.compare(df2).dropna())

    # Form the crew with the defined agents using formatted strings for goals to prevent interpolation issues
    pdf_reader = Agent(
        role='PDF Reader',
        goal='Extract text from PDFs.',
        tools=[custom_pdf_reader_tool],
        allow_delegation=False,
        verbose=True,
        memory=False
    )

    transaction_organiser = Agent(
        role='Transaction Organiser',
        goal='Organise text into structured data.',
        tools=[panda_dataframe_tool],
        allow_delegation=False,
        verbose=True,
        memory=False
    )

    dataframe_comparer = Agent(
        role='DataFrame Comparer',
        goal='Identify discrepancies between datasets.',
        tools=[compare_dataframe_tool],
        allow_delegation=False,
        verbose=True,
        memory=False
    )

    excel_bookkeeper = Agent(
        role='Excel Bookkeeper',
        goal='Convert data to Excel and prepare for download.',
        tools=[pandas_to_excel_tool],
        allow_delegation=False,
        verbose=True,
        memory=False
    )

    crew = Crew(
        agents=[pdf_reader, transaction_organiser, dataframe_comparer, excel_bookkeeper],
        tasks=[
            Task(description="Extract text from PDF", expected_output="Text", agent=pdf_reader),
            Task(description="Organise text into DataFrame", expected_output="DataFrame", agent=transaction_organiser),
            Task(description="Compare DataFrames", expected_output="Differences", agent=dataframe_comparer),
            Task(description="Generate Excel file", expected_output="File link", agent=excel_bookkeeper)
        ],
        process=Process.sequential,
        memory=False,
        cache=True,
        max_rpm=100,
        share_crew=True
    )

    if st.button("Start Processing"):
        result = crew.kickoff(inputs={'doc1': doc_1.getvalue(), 'doc2': doc_2.getvalue()})
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