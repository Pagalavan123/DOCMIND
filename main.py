from typing import TypedDict, List, Dict
import re

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END


# ================= CONFIG =================

PDF_PATH = "./Payslips.pdf"

SYSTEM_PROMPT = """
You are a financial assistant.
Explain results clearly.
Never invent numbers.
Use only provided facts.
"""

# ================= LOAD PDF =================

loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
docs = [Document(page_content=p.page_content) for p in pages]

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(docs, embeddings)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

# ================= STATE =================

class AgentState(TypedDict):
    question: str
    answer: str
    facts: Dict[str, int]
    history: List[Dict]


# ================= STRUCTURED EXTRACTION =================

def extract_structured_salary(chunks):
    text = "\n".join(c.page_content for c in chunks)

    data = {}
    blocks = re.findall(r"PAYSLIP\s*-\s*(\w+\s+\d{4})(.*?)(?=PAYSLIP|\Z)", text, re.S)

    for month, content in blocks:
        net = re.search(r"NET SALARY.*?(\d+)", content)
        if net:
            data[month] = int(net.group(1))

    return data


# ================= REASONING =================

def reason_from_facts(q: str, facts: Dict[str, int]):
    q = q.lower()

    if "all" in q and "salary" in q:
        lines = [f"{m}: Rs. {v}" for m, v in facts.items()]
        return "\n".join(lines)

    if "add" in q or "total" in q:
        return f"Total salary = Rs. {sum(facts.values())}"

    if "why" in q and "compare" in q:
        months = list(facts)
        if len(months) >= 2:
            a, b = months[-2], months[-1]
            diff = facts[b] - facts[a]
            if diff < 0:
                return f"{b} salary is lower than {a} by Rs. {abs(diff)}."
            else:
                return f"{b} salary is higher than {a} by Rs. {diff}."

    return None


# ================= RAG NODE =================

def rag_node(state: AgentState):

    # Step 1: Retrieve
    results = vectorstore.similarity_search(state["question"], k=5)

    # Step 2: Extract deterministic facts
    new_facts = extract_structured_salary(results)
    state["facts"].update(new_facts)

    # Step 3: Reason from memory
    reasoning = reason_from_facts(state["question"], state["facts"])
    if reasoning:
        state["answer"] = reasoning
        state["history"].append({"q": state["question"], "a": reasoning})
        return state

    # Step 4: Explain using LLM
    context = "\n".join(r.page_content for r in results)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{state['question']}")
    ]

    answer = llm.invoke(messages).content

    state["answer"] = answer
    state["history"].append({"q": state["question"], "a": answer})
    state["history"] = state["history"][-5:]

    return state


# ================= GRAPH =================

graph = StateGraph(AgentState)
graph.add_node("rag", rag_node)
graph.set_entry_point("rag")
graph.add_edge("rag", END)
agent = graph.compile()

# ================= RUN =================

state = {"facts": {}, "history": []}

while True:
    q = input("\nAsk from PDF (or 'exit'): ")
    if q.lower() == "exit":
        break

    state["question"] = q
    state = agent.invoke(state)
    print("\nAnswer:\n", state["answer"])
