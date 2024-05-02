#import streamlit as st
#from langchain_openai import ChatOpenAI
#from langchain_core.messages import HumanMessage
#from langgraph.graph import END, MessageGraph

import streamlit as st
import pandas as pd
import pdfplumber
from crewai import Agent, Task, Crew, Process

# Custom tool for PDF data extraction
class PDFDataExtractor:
    def extract_data(self, pdf_file):
        transactions = []
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                for row in table[1:]:  # Skip header
                    if row and row[0] and "Datum" not in row[0]:
                        datum, provisie, polis_nummer, omschrijving, debet, credit = row
                        transactions.append([datum, provisie, polis_nummer, omschrijving, debet, credit])
        return pd.DataFrame(transactions, columns=["Date", "Provisie", "PolisNummer", "Description", "Debet", "Credit"])

# Define Agents
extractor = Agent(
    role='Data Extractor',
    goal='Extract transactions from PDF files.',
    tools=[PDFDataExtractor()]
)

organizer = Agent(
    role='Data Organizer',
    goal='Prepare data for discrepancy checking.'
)

checker = Agent(
    role='Data Checker',
    goal='Identify discrepancies between transactions in different documents.'
)

reporter = Agent(
    role='Data Reporter',
    goal='Generate an Excel report with all transactions and any discrepancies found.'
)

# Define Tasks
extraction_task = Task(
    description='Extract transaction data from provided PDF files.',
    agent=extractor,
    expected_output='DataFrame with transactions.'
)

organization_task = Task(
    description='Organize transactions into a structured DataFrame.',
    agent=organizer,
    expected_output='Structured DataFrame organized by polisnummer.'
)

checking_task = Task(
    description='Identify discrepancies between transactions in different documents.',
    agent=checker,
    expected_output='DataFrame with discrepancies highlighted.'
)

reporting_task = Task(
    description='Generate an Excel report with all transactions and any discrepancies found.',
    agent=reporter,
    expected_output='Path to the generated Excel report.'
)

# Forming the crew
crew = Crew(
    agents=[extractor, organizer, checker, reporter],
    tasks=[extraction_task, organization_task, checking_task, reporting_task],
    process=Process.sequential
)

# Streamlit Interface
st.title('Rekening Courant Transaction Processor')
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=['pdf'])

if uploaded_files and len(uploaded_files) == 2:
    # Kick off the crew with PDF inputs
    result = crew.kickoff(inputs={'pdf_files': uploaded_files})
    st.success("Processing complete. Download the Excel report below.")
    with open(result['path_to_excel'], "rb") as file:
        st.download_button(
            label="Download Excel Report",
            data=file,
            file_name="output.xlsx",
            mime="application/vnd.ms-excel"
        )

