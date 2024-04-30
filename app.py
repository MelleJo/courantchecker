import getpass
import os
import pandas as pd
from typing import Annotated, List, Sequence, Tuple, TypedDict, Union
from langchain.agents import create_openai_functions_agent
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph import StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
)
import operator
import functools
from typing_extensions import TypedDict
import json
from langchain.tools import PyPDFReader
from langchain_experimental.utilities import PythonREPL
import streamlit as st
import fitz  # PyMuPDF library

# Set API keys from Streamlit secrets
OPENAI_API_KEY = st.secrets["openai_api_key"]
LANGCHAIN_API_KEY = st.secrets["langchain_api_key"]

# Set environment variables
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

# Enable tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Multi-agent Collaboration"

# Define tools
pdf_reader = PyPDFReader()

@tool
def read_pdf(pdf_file: Annotated[bytes, "The PDF file to extract text from"]):
    """Use this to read text from a PDF file."""
    try:
        with fitz.open(stream=pdf_file, filetype="pdf") as pdf:
            text = ""
            for page in pdf:
                text += page.get_text()
        return text
    except Exception as e:
        return f"Failed to read PDF. Error: {repr(e)}"
@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."]
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Succesfully executed:\n```python\n{code}\n```\nStdout: {result}"

# Define state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
    pdf_1_data: pd.DataFrame
    pdf_2_data: pd.DataFrame

# Create agents
llm = ChatOpenAI(model="gpt-4-1106-preview")

def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    functions = [format_tool_to_openai_function(t) for t in tools]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_functions(functions)

# Text Extraction Agent
text_extraction_agent = create_agent(
    llm,
    [read_pdf],
    system_message="You should extract text data from PDF documents and categorize it into pandas DataFrames.",
)

def text_extraction_node(state):
    result = text_extraction_agent.invoke(state)
    if isinstance(result, FunctionMessage):
        pass
    else:
        result = HumanMessage(**result.dict(exclude={"type", "name"}), name="Text Extractor")
    return {
        "messages": [result],
        "sender": "Text Extractor",
        "pdf_1_data": state.get("pdf_1_data", pd.DataFrame()),
        "pdf_2_data": state.get("pdf_2_data", pd.DataFrame()),
    }

# Document Comparison Agent
comparison_agent = create_agent(
    llm,
    [],
    system_message="You should compare the two pandas DataFrames and find the differences between them.",
)

def comparison_node(state):
    result = comparison_agent.invoke({"messages": state["messages"], "pdf_1_data": state["pdf_1_data"], "pdf_2_data": state["pdf_2_data"]})
    result = HumanMessage(**result.dict(exclude={"type", "name"}), name="Comparison Agent")
    return {
        "messages": [result],
        "sender": "Comparison Agent",
        "pdf_1_data": state["pdf_1_data"],
        "pdf_2_data": state["pdf_2_data"],
    }

# Excel Export Agent
excel_agent = create_agent(
    llm,
    [python_repl],
    system_message="You should export the differences between the two documents to an Excel file for the user.",
)

def excel_node(state):
    result = excel_agent.invoke(state)
    if isinstance(result, FunctionMessage):
        pass
    else:
        result = HumanMessage(**result.dict(exclude={"type", "name"}), name="Excel Agent")
    return {
        "messages": [result],
        "sender": "Excel Agent",
        "pdf_1_data": state["pdf_1_data"],
        "pdf_2_data": state["pdf_2_data"],
    }

# Tool Executor
tools = [read_pdf, python_repl]
tool_executor = ToolExecutor(tools)

def tool_node(state):
    """This runs tools in the graph

    It takes in an agent action and calls that tool and returns the result."""
    messages = state["messages"]
    last_message = messages[-1]
    tool_input = json.loads(
        last_message.additional_kwargs["function_call"]["arguments"]
    )
    if len(tool_input) == 1 and "__arg1" in tool_input:
        tool_input = next(iter(tool_input.values()))
    tool_name = last_message.additional_kwargs["function_call"]["name"]
    action = ToolInvocation(
        tool=tool_name,
        tool_input=tool_input,
    )
    response = tool_executor.invoke(action)
    function_message = FunctionMessage(
        content=f"{tool_name} response: {str(response)}", name=action.tool
    )
    return {"messages": [function_message]}

# Edge logic
def router(state):
    messages = state["messages"]
    last_message = messages[-1]
    if "function_call" in last_message.additional_kwargs:
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        return "end"
    return "continue"

# Define the graph
workflow = StateGraph(AgentState)

workflow.add_node("Text Extractor", text_extraction_node)
workflow.add_node("Comparison Agent", comparison_node)
workflow.add_node("Excel Agent", excel_node)
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "Text Extractor",
    router,
   {"continue": "Excel Agent", "call_tool": "call_tool", "end": END},
)
workflow.add_conditional_edges(
   "Excel Agent",
   router,
   {"continue": "Text Extractor", "call_tool": "call_tool", "end": END},
)

workflow.add_conditional_edges(
   "call_tool",
   lambda x: x["sender"],
   {
       "Text Extractor": "Text Extractor",
       "Comparison Agent": "Comparison Agent",
       "Excel Agent": "Excel Agent",
   },
)
workflow.set_entry_point("Text Extractor")
graph = workflow.compile()