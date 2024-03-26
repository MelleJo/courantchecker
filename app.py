import streamlit as st
import pandas as pd

def compare_excel_files(file1, file2):
    # Load the Excel files
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)

    # Check column names
    st.write("Columns in the first file:", df1.columns.tolist())
    st.write("Columns in the second file:", df2.columns.tolist())

    # Verify 'polisnummer' exists in both DataFrames
    if 'polisnummer' not in df1.columns or 'polisnummer' not in df2.columns:
        st.error("'polisnummer' column is missing in one of the files.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Continue with the original logic if the column exists
    merged_df = pd.merge(df1, df2, on='polisnummer', suffixes=('_file1', '_file2'))

    # Find mismatches in 'debet' and 'credit'
    mismatches = merged_df[(merged_df['debet_file1'] != merged_df['debet_file2']) | 
                           (merged_df['credit_file1'] != merged_df['credit_file2'])]

    return mismatches[['polisnummer', 'debet_file1', 'debet_file2', 'credit_file1', 'credit_file2']]


# Streamlit UI
st.title('Excel Comparison Tool')

file1 = st.file_uploader("Choose the first Excel file", type=['xlsx'])
file2 = st.file_uploader("Choose the second Excel file", type=['xlsx'])

if file1 and file2:
    result = compare_excel_files(file1, file2)
    if not result.empty:
        st.write("Mismatched Entries:")
        st.dataframe(result)
    else:
        st.success("All transactions match between the documents.")
