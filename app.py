import streamlit as st
from crewai import Agent
from crewai_tools import PDFSearchTool
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


# API Key
api_key = st.secrets["OPENAI_API_KEY"]

st.title("Courantchecker")

doc_1 = st.file_uploader("Dco1", type="pdf")
doc_2 = st.file_uploader("Doc2", type="pdf")

if doc_1 and doc_2:
  # Tool Definitions
  @tool("panda_dataframe_tool")
  def panda_dataframe_tool(question: str) -> pd.DataFrame:
      """Tool that takes a block of text and returns a structured pandas dataframe"""
      text = " ".join(question.split())
      rows = text.split("\n")
      df = pd.DataFrame([row.split("\t") for row in rows])
      df.set_index(df.columns[0], inplace=True)
      df.drop(df.columns[0], axis=1, inplace=True)
      return df

  @tool("pandas_to_excel_tool")
  def pandas_to_excel_tool(question: str, df: Any) -> str:
      """Tool that takes a block of text and a pandas DataFrame, and returns a link to the Excel file."""
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

  @tool("compare_dataframe_tool")
  def compare_dataframe_tool(question: str, reference_df: Any) -> list:
      """Tool that compares a text-formatted table converted into a DataFrame with a reference DataFrame."""
      if not isinstance(reference_df, pd.DataFrame):
          raise TypeError("reference_df must be a pandas DataFrame")
      text = " ".join(question.split())
      rows = text.split("\n")
      df = pd.DataFrame([row.split("\t") for row in rows])
      df.set_index(df.columns[0], inplace=True)
      df.drop(df.columns[0], axis=1, inplace=True)
      return list(reference_df.compare(df).dropna())

  # PDF Tool Initialization
  try:
      pdf_search_tool = PDFSearchTool()
  except Exception as e:
      st.error("Failed to initialize PDF processing tool. Please check system configuration.")
      st.stop()

  # Streamlit File Uploaders


  # Detailed Agent Definitions
  pdf_reader = Agent(
      role='PDF Reader',
      goal='Carefully and in full completeness extract all the transactions from the pdf files {doc_1} and {doc_2}. You make sure to clearly define from which document each transaction comes from.',
      verbose=True,
      memory=False,
      backstory=(
          "Driven by accuracy and servitude, you extract all the transactions in full detail and without missing anything."
      ),
      tools=[pdf_search_tool],
      allow_delegation=False
  )

  transaction_organiser = Agent(
      role='Transaction Organiser',
      goal='Organise the extracted transactions from {pdf_reader} into a pandas dataframe',
      verbose=True,
      memory=False,
      backstory=(
          "With a flair for data manipulation, you take the transactions from {pdf_reader} and organise them into a dataframe, grouped by polisnummer, showing all transactions as either debit or credit. And most importantly, in which document {doc_1} or {doc_2}. You make sure all transactions are easy to read and understand."
      ),
      tools=[panda_dataframe_tool],
      allow_delegation=False
  )

  dataframe_comparer = Agent(
      role='DataFrame Comparer',
      goal='Compare the dataframes from {transaction_organiser} and highlight any discrepancies between {doc_1} and {doc_2}',
      verbose=True,
      memory=False,
      backstory=(
          "With a keen eye for detail, you compare the dataframes from {transaction_organiser} and highlight any discrepancies between {doc_1} and {doc_2}. You make sure to clearly categorize from which document each transaction comes, and order the discrepancies by polisnummer and define whether it is credit or debit."
      ),
      tools=[compare_dataframe_tool],
      allow_delegation=False
  )

  excel_bookkeeper = Agent(
      role='Excel Bookkeeper',
      goal='Bookkeep the transactions from {dataframe_comparer} and export the comparison list to Excel in a structured and easy to read format',
      verbose=True,
      memory=False,
      backstory=(
          "With a keen eye for detail and a lot of experience as a bookkeeper, you take all the transactions from {dataframe_comparer} and bookkeep them in a structured and easy to read format in an Excel file."
      ),
      tools=[pandas_to_excel_tool],
      allow_delegation=False
  )

  # Form the crew
  crew = Crew(
      agents=[pdf_reader, transaction_organiser, dataframe_comparer, excel_bookkeeper],
      tasks=[
          Task(description="Extract transactions", expected_output="Dataframe", tools=[panda_dataframe_tool], agent=transaction_organiser),
          Task(description="Compare transactions", expected_output="List of discrepancies", tools=[compare_dataframe_tool], agent=dataframe_comparer),
          Task(description="Export transactions to Excel", expected_output="Excel file", tools=[pandas_to_excel_tool], agent=excel_bookkeeper)
      ],
      process=Process.sequential,
      memory=False,
      cache=True,
      max_rpm=100,
      share_crew=True
  )

# Streamlit Interfac

if st.button("Start Processing"):
    if doc_1 and doc_2:
        # Process the documents
        result = crew.kickoff(inputs={'doc1': doc_1, 'doc2': doc_2})
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