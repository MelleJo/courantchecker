import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessageGraph

st.title("Courantchecker")
openai_api_key = st.secrets["OPENAI_API_KEY"]

model = ChatOpenAI(temperature=0)

graph = MessageGraph()

graph.add_node("oracle", model)
graph.add_edge("oracle", END)

graph.set_entry_point("oracle")

runnable = graph.compile()

st.write(runnable.invoke(HumanMessage("What is 1 + 1?")))
