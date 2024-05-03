import streamlit as st
from crewai import Agent
from crewai_tools import PDFSearchTool
import os
from crewai import Task
from crewai import Crew, Process
from crewai_tools import tool
import pandas as pd
import chromadb



#os.environ["SERPER_API_KEY"] = "Your Key"  # serper.dev API key
api_key = st.secrets["OPENAI_API_KEY"] = "Your Key"


@tool("panda_dataframe_tool")
def panda_dataframe_tool(question: str) -> pd.DataFrame:
    """Tool that takes a block of text and returns a structured pandas dataframe"""
    # Remove any whitespace
    text = " ".join(question.split())

    # Split the text into rows based on newlines
    rows = text.split("\n")

    # Load the data from the text into a dataframe
    df = pd.DataFrame([row.split("\t") for row in rows])

    # Set the index to the first column
    df.set_index(df.columns[0], inplace=True)

    # Drop the first column as it's now the index
    df.drop(df.columns[0], axis=1, inplace=True)

    return df


@tool("pandas_to_excel_tool")
def pandas_to_excel_tool(question: str, df: pd.DataFrame) -> str:
    """Tool that takes a block of text and a pandas dataframe and returns a link to the excel file"""
    import tempfile
    import shutil
    import xlsxwriter
    import base64

    # Create a temporary file
    tmp_file = tempfile.NamedTemporaryFile()

    # Create a writer
    writer = pd.ExcelWriter(tmp_file.name, engine='xlsxwriter')

    # Write the df to the excel file
    df.to_excel(writer, index=False)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

    # Read the excel file into a base64 string
    with open(tmp_file.name, "rb") as f:
        file_bytes = f.read()
        file_b64 = base64.b64encode(file_bytes).decode()

    # Remove the temp file
    shutil.rmtree(os.path.dirname(tmp_file.name))

    # Return a link to the excel file
    url = f"<a href=\"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{file_b64}\" download=\"file.xlsx\">Download Excel file</a>"

    return url

@tool("compare_dataframe_tool")
def compare_dataframe_tool(question: str, reference_df: pd.DataFrame) -> list:
    """Tool that takes a block of text and a reference dataframe and returns a list of differences"""
    # Remove any whitespace
    text = " ".join(question.split())

    # Split the text into rows based on newlines
    rows = text.split("\n")

    # Load the data from the text into a dataframe
    df = pd.DataFrame([row.split("\t") for row in rows])

    # Set the index to the first column
    df.set_index(df.columns[0], inplace=True)

    # Drop the first column as it's now the index
    df.drop(df.columns[0], axis=1, inplace=True)

    # Return a list of differences between the dataframes
    return list(reference_df.compare(df).dropna())




#search_tool = SerperDevTool()

doc_1 = st.file_uploader("Dco1", type="pdf")
doc_2 = st.file_uploader("Doc2", type="pdf")

# Creating a senior researcher agent with memory and verbose mode
pdf_reader = Agent(
  role='PDF Reader',
  goal='Carefully and in full completeness extract all the transactions from the pdf files {doc_1} and {doc_2}. You make sure to clearly define from which document each transaction comes from.',
  verbose=True,
  memory=True,
  backstory=(
    "Driven by accuracy and servitude, you extract all the transactions in full detail and without missing anything."
  ),
  tools=[PDFSearchTool],
  allow_delegation=False
)

# Creating a writer agent with custom tools and delegation capability
transaction_organiser = Agent(
  role='Transaction Organiser',
  goal='Organise the extracted transactions from {pdf_reader} into a pandas dataframe',
  verbose=True,
  memory=True,
  backstory=(
    "With a flair for data manipulation, you take the transactions from {pdf_reader} and organise them into a dataframe, grouped by polisnummer, showing all transactions as either debit or credit. And most importantly, in which document {doc_1} or {doc_2}. You make sure all transactions are easy to read and understand."
  ),
  tools=[panda_dataframe_tool],
  allow_delegation=False
)

# Creating a final agent that compares the dataframes and highlights any differences
dataframe_comparer = Agent(
  role='DataFrame Comparer',
  goal='Compare the dataframes from {transaction_organiser} and highlight any discrepancies between {doc_1} and {doc_2}',
  verbose=True,
  memory=True,
  backstory=(
    "With a keen eye for detail, you compare the dataframes from {transaction_organiser} and highlight any discrepancies between {doc_1} and {doc_2}. You make sure to clearly categorize from which document each transaction comes, and order the discrepancies by polisnummer and define whether it is credit or debit."
  ),
  tools=[compare_dataframe_tool],
  allow_delegation=False
)


# Creating a highly intelligent and sophisticated bookkeeper agent with custom tools and delegation capability
excel_bookkeeper = Agent(
  role='Excel Bookkeeper',
  goal='Bookkeep the transactions from {dataframe_comparer} and export the comparison list to Excel in a structured and easy to read format',
  verbose=True,
  memory=True,
  backstory=(
    "With a keen eye for detail and a lot of experience as a bookkeeper, you take all the transactions from {dataframe_comparer} and bookkeep them in a structured and easy to read format in an Excel file."
  ),
  tools=[pandas_to_excel_tool],
  allow_delegation=False
)


# Interface for uploading PDF files
# (skipping this step as the upload is already done in the previous task)

# Task 2: Extract transactions from PDF files
extract_transactions_task = Task(
  description="Extract transactions from the uploaded PDF files",
  expected_output="A pandas dataframe containing all the transactions",
  tools=[panda_dataframe_tool],
  agent=transaction_organiser,
)

# Task 3: Compare the transactions and highlight any discrepancies
compare_transactions_task = Task(
  description="Compare the transactions from the two PDF files and highlight any discrepancies",
  expected_output="A list of differences between the two dataframes",
  tools=[compare_dataframe_tool],
  agent=dataframe_comparer,
)

# Task 4: Bookkeep the transactions and export the comparison list to Excel
bookkeep_transactions_task = Task(
  description="Bookkeep the transactions and export the comparison list to Excel",
  expected_output="An Excel file containing the comparison of the transactions",
  tools=[pandas_to_excel_tool],
  agent=excel_bookkeeper,
)


# Interface for Streamlit
st.title("Courantchecker")

# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
  agents=[dataframe_comparer, transaction_organiser, pdf_reader, excel_bookkeeper],
  tasks=[extract_transactions_task, compare_transactions_task, bookkeep_transactions_task],
  process=Process.sequential,  # Optional: Sequential task execution is default
  memory=True,
  cache=True,
  max_rpm=100,
  share_crew=True
)

# Starting the task execution process with enhanced feedback
result = crew.kickoff(inputs={'topic': 'AI in healthcare'})

# Displaying the result in the Streamlit interface
if isinstance(result, str):
    st.success(result)
elif isinstance(result, bytes):
    st.success("The Excel file is ready to download")
    href = f"<a href='data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{base64.b64encode(result).decode()}' download='comparison.xlsx'>Download Excel file</a>"
    st.markdown(href, unsafe_allow_html=True)
else:
    st.error("Something went wrong")


