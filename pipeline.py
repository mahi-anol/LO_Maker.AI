"""
LangGraph pipeline for generating Bengali lesson plans.
Each node generates one section of the lesson plan template.
"""

import os
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, SystemMessage
import tempfile


# ── State Schema ──────────────────────────────────────────────────────────────

class LessonPlanState(TypedDict):
    # Inputs
    teacher_name: str
    subject: str
    grade: str
    duration: str
    learning_outcome: str
    textbook_pdf_path: str
    model_name: str

    # Retrieved context from textbook
    context: str

    # Generated sections (Bangla content)
    lesson_vision: str          # Why (academic + non-academic) + What
    key_points: str             # How steps
    assessment: str             # Assessment questions + exemplar
    launch: str                 # SEL + prior knowledge + sparking curiosity
    explore: str                # Exploration activity + probing questions
    conceptualize: str          # Teacher action / do / say / CFU
    guided_practice: str        # Guided practice problem + solution
    independent_practice: str   # Independent practice problem
    lesson_closing: str         # Exit ticket + closing

    # Final assembled dict (used by renderers)
    lesson_plan: dict
    error: Optional[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """তুমি একজন অভিজ্ঞ শিক্ষক যিনি বাংলাদেশের পাঠ্যক্রম অনুযায়ী পাঠ পরিকল্পনা তৈরি করেন।
তোমাকে প্রদত্ত টেমপ্লেট অনুসারে বাংলায় পাঠ পরিকল্পনার বিভিন্ন অংশ তৈরি করতে হবে।
শুধুমাত্র বাংলায় উত্তর দাও। কোনো ইংরেজি অনুবাদ বা ব্যাখ্যা যোগ করো না।
উত্তর সংক্ষিপ্ত, স্পষ্ট এবং শ্রেণিকক্ষ-উপযোগী হতে হবে।"""


def get_llm(model_name: str) -> ChatOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return ChatOpenAI(
        model=model_name,
        temperature=0.4,
        openai_api_key=api_key,
    )


def build_vector_store(pdf_path: str, model_name: str) -> FAISS:
    """Load PDF, split into chunks, embed with OpenAI, store in FAISS."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", "।", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def retrieve_context(vectorstore: FAISS, learning_outcome: str, k: int = 6) -> str:
    """Retrieve top-k relevant chunks from the textbook for the given LO."""
    results = vectorstore.similarity_search(learning_outcome, k=k)
    context_parts = [doc.page_content for doc in results]
    return "\n\n---\n\n".join(context_parts)


def call_llm(llm: ChatOpenAI, prompt: str) -> str:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


# ── Graph Nodes ───────────────────────────────────────────────────────────────

def node_retrieve_context(state: LessonPlanState) -> LessonPlanState:
    """Node 1: Build vector store from textbook and retrieve relevant content."""
    try:
        vectorstore = build_vector_store(
            state["textbook_pdf_path"], state["model_name"]
        )
        context = retrieve_context(vectorstore, state["learning_outcome"])
        return {**state, "context": context, "error": None}
    except Exception as e:
        return {**state, "context": "", "error": f"Context retrieval error: {str(e)}"}


def node_generate_lesson_vision(state: LessonPlanState) -> LessonPlanState:
    """Node 2: Generate Why (academic + non-academic) and What sections."""
    if state.get("error"):
        return state

    llm = get_llm(state["model_name"])
    lo = state["learning_outcome"]
    context = state["context"]

    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল (Learning Outcome): {lo}

পাঠ্যপুস্তক থেকে প্রাসঙ্গিক অংশ:
{context}

নিচের তিনটি অংশ আলাদা আলাদাভাবে বাংলায় লেখো:

### WHY (Academic):
শিক্ষার্থীরা এই বিষয়টি শিখলে গাণিতিক/একাডেমিক দিক থেকে কী কী সুবিধা পাবে? (২-৩ বাক্য)

### WHY (Non-Academic):
এই বিষয়টি শিখলে দৈনন্দিন জীবনে কীভাবে কাজে লাগবে? (২-৩ বাক্য)

### WHAT (Concept):
এই টপিকের মূল ধারণা কী? সংক্ষেপে সংজ্ঞা ও মূল বিষয় ব্যাখ্যা করো। (৩-৫ বাক্য)

প্রতিটি অংশ ### দিয়ে শুরু করো।"""

    result = call_llm(llm, prompt)
    return {**state, "lesson_vision": result}


def node_generate_key_points(state: LessonPlanState) -> LessonPlanState:
    """Node 3: Generate HOW steps."""
    if state.get("error"):
        return state

    llm = get_llm(state["model_name"])
    lo = state["learning_outcome"]

    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {lo}

শিক্ষার্থীরা কীভাবে এই সমস্যা সমাধান করবে তার ধাপগুলো বাংলায় লেখো।
সর্বোচ্চ ৪টি ধাপ লেখো, প্রতিটি ধাপ নম্বর দিয়ে শুরু করো।
উদাহরণ:
১) সমস্যাটি বোঝার চেষ্টা করো।
২) ...
৩) ...
৪) ...

শুধু ধাপগুলো লেখো, অন্য কিছু নয়।"""

    result = call_llm(llm, prompt)
    return {**state, "key_points": result}


def node_generate_assessment(state: LessonPlanState) -> LessonPlanState:
    """Node 4: Generate assessment questions with exemplar answers."""
    if state.get("error"):
        return state

    llm = get_llm(state["model_name"])
    lo = state["learning_outcome"]
    context = state["context"]

    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {lo}

পাঠ্যপুস্তক প্রসঙ্গ:
{context}

নিচের তিনটি অংশ তৈরি করো:

### সময় ও পূর্ণমান:
(যেমন: সময়: ৪ মিনিট | পূর্ণমান: ৬ মার্ক)

### প্রশ্নসমূহ:
দুটি প্রশ্ন তৈরি করো:
১) একটি MCQ প্রশ্ন (ক, খ, গ, ঘ অপশনসহ) - ২ মার্ক
২) একটি গণনামূলক প্রশ্ন - ৪ মার্ক
প্রশ্নগুলো শিক্ষার ফলাফলের সাথে সরাসরি সম্পর্কিত হতে হবে।

### আদর্শ উত্তর (Exemplar):
উভয় প্রশ্নের ধাপে ধাপে সমাধান।

প্রতিটি অংশ ### দিয়ে শুরু করো।"""

    result = call_llm(llm, prompt)
    return {**state, "assessment": result}


def node_generate_launch(state: LessonPlanState) -> LessonPlanState:
    """Node 5: Generate the Launch section (SEL + curiosity + prior knowledge)."""
    if state.get("error"):
        return state

    llm = get_llm(state["model_name"])
    lo = state["learning_outcome"]

    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {lo}
বিষয়: {state['subject']} | শ্রেণি: {state['grade']}

পাঠের শুরুর অংশ (Launch) বাংলায় তৈরি করো। নিচের চারটি অংশ আলাদাভাবে লেখো:

### SEL (Mood Checker):
শিক্ষার্থীদের আবেগ-অনুভূতি যাচাই করার জন্য শিক্ষকের কথা (২-৩ বাক্য)। "ফিস্ট টু ফাইভ" বা অনুরূপ পদ্ধতি ব্যবহার করো।

### Sparking Curiosity:
একটি সহজ গল্প বা উদাহরণ দিয়ে শিক্ষার্থীদের আগ্রহ তৈরি করো। গল্পটি আজকের টপিকের সাথে সম্পর্কিত হতে হবে। (৩-৫ বাক্য)

### Learning Outcome ঘোষণা:
শিক্ষক আজকের পাঠের লক্ষ্য ঘোষণা করবেন। (১-২ বাক্য)

### Prior Knowledge:
শিক্ষার্থীদের পূর্বজ্ঞান যাচাই করার জন্য একটি প্রশ্ন।

প্রতিটি অংশ ### দিয়ে শুরু করো।"""

    result = call_llm(llm, prompt)
    return {**state, "launch": result}


def node_generate_explore(state: LessonPlanState) -> LessonPlanState:
    """Node 6: Generate the Explore section."""
    if state.get("error"):
        return state

    llm = get_llm(state["model_name"])
    lo = state["learning_outcome"]
    context = state["context"]

    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {lo}
পাঠ্যপুস্তক প্রসঙ্গ: {context}

Explore পর্বের জন্য বাংলায় লেখো:

### Exploration Activity:
শিক্ষার্থীদের জন্য একটি সহজ অনুসন্ধানমূলক কাজ। শিক্ষক কী বলবেন তা সংলাপ আকারে লেখো।

### Probing Questions:
শিক্ষার্থীদের চিন্তা উদ্দীপিত করার জন্য ২-৩টি প্রশ্ন।

প্রতিটি অংশ ### দিয়ে শুরু করো।"""

    result = call_llm(llm, prompt)
    return {**state, "explore": result}


def node_generate_conceptualize(state: LessonPlanState) -> LessonPlanState:
    """Node 7: Generate Conceptualize section (Do, Say, Example, CFU)."""
    if state.get("error"):
        return state

    llm = get_llm(state["model_name"])
    lo = state["learning_outcome"]
    context = state["context"]

    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {lo}
পাঠ্যপুস্তক প্রসঙ্গ: {context}
How ধাপগুলো: {state.get('key_points', '')}

Conceptualize পর্বের জন্য বাংলায় লেখো:

### Teacher Do:
শিক্ষক বোর্ডে/স্লাইডে কী লিখবেন বা দেখাবেন। ধাপগুলো ক্রমানুসারে লেখো।

### Teacher Say:
শিক্ষক কী বলবেন - সংলাপ আকারে (২-৩ বাক্য)।

### Worked Example:
শিক্ষক একটি সম্পূর্ণ উদাহরণ সমাধান করে দেখাবেন। ধাপে ধাপে সমাধান দেখাও।

### Check for Understanding (CFU):
শিক্ষার্থীদের বোঝার মাত্রা যাচাইয়ের জন্য ১টি প্রশ্ন।

প্রতিটি অংশ ### দিয়ে শুরু করো।"""

    result = call_llm(llm, prompt)
    return {**state, "conceptualize": result}


def node_generate_guided_practice(state: LessonPlanState) -> LessonPlanState:
    """Node 8: Generate Guided Practice problem."""
    if state.get("error"):
        return state

    llm = get_llm(state["model_name"])
    lo = state["learning_outcome"]
    context = state["context"]

    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {lo}
পাঠ্যপুস্তক প্রসঙ্গ: {context}

Guided Practice পর্বের জন্য বাংলায় লেখো:

### Teacher Say:
শিক্ষক কীভাবে এই অনুশীলন পরিচালনা করবেন (সংক্ষেপে)।

### সমস্যা:
একটি অনুশীলন সমস্যা তৈরি করো (শিক্ষার্থীর জন্য)।

### WTD (What To Do) নির্দেশনা:
৪টি ধাপে শিক্ষার্থীদের কী করতে হবে তা লেখো।

### সমাধান (শুধু শিক্ষকের জন্য):
ধাপে ধাপে সম্পূর্ণ সমাধান।

প্রতিটি অংশ ### দিয়ে শুরু করো।"""

    result = call_llm(llm, prompt)
    return {**state, "guided_practice": result}


def node_generate_independent_practice(state: LessonPlanState) -> LessonPlanState:
    """Node 9: Generate Independent Practice problem."""
    if state.get("error"):
        return state

    llm = get_llm(state["model_name"])
    lo = state["learning_outcome"]

    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {lo}

Independent Practice পর্বের জন্য বাংলায় লেখো:

### Teacher Say:
শিক্ষার্থীরা কীভাবে একা কাজ করবে তার নির্দেশনা (সংক্ষেপে)।

### সমস্যা:
একটি স্বতন্ত্র অনুশীলন সমস্যা (Guided Practice এর চেয়ে একটু কঠিন)।

### WTD নির্দেশনা:
৪টি ধাপে কী করতে হবে।

প্রতিটি অংশ ### দিয়ে শুরু করো।"""

    result = call_llm(llm, prompt)
    return {**state, "independent_practice": result}


def node_generate_closing(state: LessonPlanState) -> LessonPlanState:
    """Node 10: Generate Lesson Closing and Exit Ticket."""
    if state.get("error"):
        return state

    llm = get_llm(state["model_name"])
    lo = state["learning_outcome"]
    assessment = state.get("assessment", "")

    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {lo}
মূল্যায়ন প্রশ্নসমূহ: {assessment}

Lesson Closing পর্বের জন্য বাংলায় লেখো:

### Teacher Say:
পাঠ শেষে শিক্ষক কীভাবে পর্যালোচনা করবেন এবং শিক্ষার্থীদের প্রশংসা করবেন।

### Exit Ticket নির্দেশনা:
শিক্ষার্থীরা কীভাবে Exit Ticket সম্পন্ন করবে তার নির্দেশনা।

### WTD নির্দেশনা:
৪টি ধাপে কী করতে হবে।

প্রতিটি অংশ ### দিয়ে শুরু করো।"""

    result = call_llm(llm, prompt)
    return {**state, "lesson_closing": result}


def node_assemble(state: LessonPlanState) -> LessonPlanState:
    """Node 11: Assemble all sections into the final lesson_plan dict."""
    lesson_plan = {
        "teacher_name": state["teacher_name"],
        "subject": state["subject"],
        "grade": state["grade"],
        "duration": state["duration"],
        "learning_outcome": state["learning_outcome"],
        "lesson_vision": state.get("lesson_vision", ""),
        "key_points": state.get("key_points", ""),
        "assessment": state.get("assessment", ""),
        "launch": state.get("launch", ""),
        "explore": state.get("explore", ""),
        "conceptualize": state.get("conceptualize", ""),
        "guided_practice": state.get("guided_practice", ""),
        "independent_practice": state.get("independent_practice", ""),
        "lesson_closing": state.get("lesson_closing", ""),
    }
    return {**state, "lesson_plan": lesson_plan}


# ── Build Graph ────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(LessonPlanState)

    graph.add_node("retrieve_context", node_retrieve_context)
    graph.add_node("generate_lesson_vision", node_generate_lesson_vision)
    graph.add_node("generate_key_points", node_generate_key_points)
    graph.add_node("generate_assessment", node_generate_assessment)
    graph.add_node("generate_launch", node_generate_launch)
    graph.add_node("generate_explore", node_generate_explore)
    graph.add_node("generate_conceptualize", node_generate_conceptualize)
    graph.add_node("generate_guided_practice", node_generate_guided_practice)
    graph.add_node("generate_independent_practice", node_generate_independent_practice)
    graph.add_node("generate_closing", node_generate_closing)
    graph.add_node("assemble", node_assemble)

    graph.set_entry_point("retrieve_context")

    graph.add_edge("retrieve_context", "generate_lesson_vision")
    graph.add_edge("generate_lesson_vision", "generate_key_points")
    graph.add_edge("generate_key_points", "generate_assessment")
    graph.add_edge("generate_assessment", "generate_launch")
    graph.add_edge("generate_launch", "generate_explore")
    graph.add_edge("generate_explore", "generate_conceptualize")
    graph.add_edge("generate_conceptualize", "generate_guided_practice")
    graph.add_edge("generate_guided_practice", "generate_independent_practice")
    graph.add_edge("generate_independent_practice", "generate_closing")
    graph.add_edge("generate_closing", "assemble")
    graph.add_edge("assemble", END)

    return graph.compile()


def run_pipeline(
    teacher_name: str,
    subject: str,
    grade: str,
    duration: str,
    learning_outcome: str,
    textbook_pdf_path: str,
    model_name: str,
) -> dict:
    """Entry point: run the full LangGraph pipeline and return lesson_plan dict."""
    graph = build_graph()

    initial_state: LessonPlanState = {
        "teacher_name": teacher_name,
        "subject": subject,
        "grade": grade,
        "duration": duration,
        "learning_outcome": learning_outcome,
        "textbook_pdf_path": textbook_pdf_path,
        "model_name": model_name,
        "context": "",
        "lesson_vision": "",
        "key_points": "",
        "assessment": "",
        "launch": "",
        "explore": "",
        "conceptualize": "",
        "guided_practice": "",
        "independent_practice": "",
        "lesson_closing": "",
        "lesson_plan": {},
        "error": None,
    }

    final_state = graph.invoke(initial_state)

    if final_state.get("error"):
        raise RuntimeError(final_state["error"])

    return final_state["lesson_plan"]
