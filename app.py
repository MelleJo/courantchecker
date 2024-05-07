import streamlit as st
from crewai import Agent
import os
from crewai import Task
from crewai import Crew, Process
from crewai_tools import tool
import pandas as pd
import chromadb
import base64
from typing import Any, List
import PyPDF2
from PyPDF2 import PdfReader
import tempfile
import shutil
import io
from pydantic import BaseModel, validator, ValidationError, Field
import pydantic

# Custom pydantic model for pd.DataFrame

class DataFrameModel(BaseModel):
    data: List[List[Any]] = Field(...)
    columns: List[str] = Field(...)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        return cls(data=df.values.tolist(), columns=df.columns.tolist())

    def to_dataframe(self):
        return pd.DataFrame(self.data, columns=self.columns)

    @validator('data', each_item=True)
    def validate_data_row(cls, row, values):
        columns = values.get('columns')
        if columns and len(row) != len(columns):
            raise ValueError(f"Row length ({len(row)}) does not match the number of columns ({len(columns)})")
        return row

    class Config:
        arbitrary_types_allowed = True

# Enable arbitrary_types_allowed for Pydantic
pydantic.config.arbitrary_types_allowed = True

api_key = st.secrets["OPENAI_API_KEY"]

st.title("Courantchecker")

doc_1 = st.file_uploader("Doc1", type=["pdf"])
doc_2 = st.file_uploader("Doc2", type=["pdf"])

if doc_1 and doc_2:
    @tool("custom_pdf_reader_tool")
    def custom_pdf_reader_tool(uploaded_file) -> str:
        """
        Extracts text from the uploaded PDF files using PyPDF2.
        Args:
            uploaded_file: A file-like object containing the PDF file data.
        Returns:
            str: All text extracted from the PDF.
        """
        try:
            reader = PdfReader(uploaded_file)
            text = []
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text.append(extracted_text)
            return "\n".join(text)
        except Exception as e:
            return f"Failed to process PDF: {str(e)}"

    @tool("panda_dataframe_tool")
    def panda_dataframe_tool(text: str) -> DataFrameModel:
        """
        Convert extracted text into a structured pandas DataFrame.

        Args:
            text (str): Text containing tab-separated values.

        Returns:
            DataFrameModel: Data organized as a DataFrame.
        """
        rows = text.split("\n")
        df = pd.DataFrame([row.split("\t") for row in rows])
        df.set_index(df.columns[0], inplace=True)
        df.drop(df.columns[0], axis=1, inplace=True)
        return DataFrameModel.from_dataframe(df)

    @tool("pandas_to_excel_tool")
    def pandas_to_excel_tool(df_model: DataFrameModel) -> str:
        """
        Save DataFrame to an Excel file and provide a download link.

        Args:
            df_model (DataFrameModel): Data to be written to Excel.

        Returns:
            str: A hyperlink to download the Excel file.
        """
        df = df_model.to_dataframe()
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        with pd.ExcelWriter(tmp_file.name, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        with open(tmp_file.name, "rb") as f:
            file_bytes = f.read()
        file_b64 = base64.b64encode(file_bytes).decode()
        os.unlink(tmp_file.name)
        return f"<a href='data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{file_b64}' download='file.xlsx'>Download Excel file</a>"

    @tool("compare_dataframe_tool")
    def compare_dataframe_tool(df1: DataFrameModel, df2: DataFrameModel) -> list:
        """
        Compare two DataFrames and return the differences.

        Args:
            df1 (DataFrameModel): The first DataFrame.
            df2 (DataFrameModel): The second DataFrame to compare against.

        Returns:
            list: Differences between the two DataFrames.
        """
        df1 = df1.to_dataframe()
        df2 = df2.to_dataframe()
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
        if doc_1 and doc_2:
            # Assuming 'doc_1' and 'doc_2' are the files uploaded via Streamlit
            result = crew.kickoff(inputs={'doc1': doc_1, 'doc2': doc_2})
            if isinstance(result, str):
                st.success(result)
            elif isinstance(result, bytes):
                st.success("The Excel file is ready to download")
                href = f"<a href='data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{base64.b64encode(result).decode()}' download='comparison.xlsx'>Download Excel file</a>"
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.error("Something went wrong")

    st.warning("Please upload both PDF documents to start processing.")