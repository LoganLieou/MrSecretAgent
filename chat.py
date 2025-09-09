from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.chat_models import ChatOpenAI

# Assume you have LMStudioEmbeddings() already implemented
from embeddings import LMStudioEmbeddings 

# 1. Build FAISS retriever
embedding_model = LMStudioEmbeddings()
faiss_index = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = faiss_index.as_retriever()

# 2. Define the LLM
llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="fake_key"
)

# 3. Define the prompt with memory + retrieved context
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that uses past chat history and retrieved documents."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("system", "Relevant context:\n{context}")
])

# 4. Define the workflow state
def retrieve_docs(state):
    query = state["input"]
    docs = retriever.get_relevant_documents(query)
    state["context"] = "\n\n".join(d.page_content for d in docs)
    return state

def call_llm(state):
    chain = prompt | llm
    response = chain.invoke({
        "chat_history": state["chat_history"],
        "input": state["input"],
        "context": state.get("context", "")
    })
    state["chat_history"].append(HumanMessage(content=state["input"]))
    state["chat_history"].append(response)
    state["output"] = response.content
    return state

# 5. Build the LangGraph
workflow = StateGraph(dict)

workflow.add_node("retrieve", retrieve_docs)
workflow.add_node("llm", call_llm)

workflow.add_edge("retrieve", "llm")
workflow.set_entry_point("retrieve")
workflow.set_finish_point("llm")

# 6. Add memory (persists across turns)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Example usage
if __name__ == "__main__":
    thread = {"configurable": {"thread_id": "chat-1"}}  # required for MemorySaver

    state = {"chat_history": []}
    print("Chatbot ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break

        state["input"] = user_input
        result = app.invoke(state, config=thread)

        print("Bot:", result["output"])

        # Carry chat history forward
        # state["chat_history"] = result["chat_history"]