"""
LangGraph pipeline — per-cell generation with explicit context chaining.

Architecture change:
  - Every DOCX cell gets its own dedicated LLM call and its own state field.
  - No ### section parsing at runtime — each LLM call returns exactly ONE cell's
    worth of plain text (no markers needed).
  - Context chaining: cells that depend on earlier cells receive those cells'
    already-generated content as explicit prompt inputs.
  - clean_text() only strips LaTeX / bold markers — never touches content structure.

Cell → State field mapping (mirrors docx_generator exactly):
  Table 0  : teacher_name, subject, grade, duration        (user inputs, no LLM)
  Table 1  :
    R2      : lo_text          — learning outcome           (user input, no LLM)
    R4      : assess_time      — "সময়: X | পূর্ণমান: Y"
    R4      : assess_questions — MCQ + calculation question
    R6      : assess_exemplar  — step-by-step answers (depends on assess_questions)
    R8 C0   : vision_what      — WHAT concept definition
    R8 C0   : vision_why_ac    — WHY academic
    R8 C0   : vision_why_no    — WHY non-academic
    R8 C1   : key_points       — How steps
    R10 C0  : knowledge_text   — knowledge from LO
    R12 C0  : blooms_skills    — Bloom's verb list
  Table 2  :
    R3  C0  : launch_teacher   — teacher action
    R3  C1  : launch_student   — student action
    R3  C2  : launch_materials — materials
    R6  C0  : explore_teacher
    R6  C1  : explore_student
    R6  C2  : explore_materials
    R9  C0  : concept_teacher
    R9  C1  : concept_student
    R9  C2  : concept_materials
    R12 C0  : guided_teacher
    R12 C1  : guided_student
    R12 C2  : guided_materials
    R15 C0  : indep_teacher
    R15 C1  : indep_student
    R15 C2  : indep_materials
    R18 C0  : closing_teacher
    R18 C1  : closing_student
    R18 C2  : closing_materials
"""

import os
import re
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, SystemMessage


# ── State schema ───────────────────────────────────────────────────────────────

class LessonPlanState(TypedDict):
    # User inputs
    teacher_name: str
    subject: str
    grade: str
    duration: str
    learning_outcome: str
    textbook_pdf_path: str
    model_name: str
    use_saved_embedding: bool
    saved_embedding_alias: str
    save_new_embedding: bool
    new_embedding_alias: str

    # RAG context
    context: str

    # ── Table 1 cells ──────────────────────────────────────────────────────
    assess_time: str        # R4: "সময়: ৫ মিনিট | পূর্ণমান: ৬ মার্ক"
    assess_questions: str   # R4: MCQ + calculation question (plain text)
    assess_exemplar: str    # R6: step-by-step answers (depends on assess_questions)

    vision_why_ac: str      # R8 C0 block 1
    vision_why_no: str      # R8 C0 block 2
    vision_what: str        # R8 C0 block 3
    key_points: str         # R8 C1: How steps
    knowledge_text: str     # R10 C0
    blooms_skills: str      # R12 C0

    # ── Table 2 cells ──────────────────────────────────────────────────────
    launch_teacher: str
    launch_student: str
    launch_materials: str

    explore_teacher: str
    explore_student: str
    explore_materials: str

    concept_teacher: str
    concept_student: str
    concept_materials: str

    guided_teacher: str
    guided_student: str
    guided_materials: str

    indep_teacher: str
    indep_student: str
    indep_materials: str

    closing_teacher: str
    closing_student: str
    closing_materials: str

    # Final assembled dict passed to docx_generator
    lesson_plan: dict
    error: Optional[str]


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """তুমি একজন অভিজ্ঞ শিক্ষক যিনি বাংলাদেশের পাঠ্যক্রম অনুযায়ী পাঠ পরিকল্পনা তৈরি করেন।

কঠোর নিয়মসমূহ — প্রতিটি নিয়ম অবশ্যই মেনে চলতে হবে:
১) শুধুমাত্র সাধারণ বাংলা টেক্সট লেখো। কোনো মার্কডাউন নয়।
২) কোনো ** বা * বা # চিহ্ন ব্যবহার করবে না।
৩) কোনো LaTeX নয়: কোনো \\( \\) নেই, কোনো $ $ নেই।
৪) গণিত সরাসরি লেখো: 3(x+2) = 3x+6
৫) শুধুমাত্র বাংলায় লেখো।
৬) তুমি শুধু যা চাওয়া হয়েছে তাই লিখবে — অতিরিক্ত কোনো শিরোনাম, লেবেল বা ব্যাখ্যা যোগ করবে না।"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )


def get_llm(model_name: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        temperature=0.4,
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    )


def build_vector_store_from_pdf(pdf_path: str) -> FAISS:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", "।", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


def retrieve_context(vectorstore: FAISS, learning_outcome: str, k: int = 6) -> str:
    results = vectorstore.similarity_search(learning_outcome, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in results)


def call_llm(llm: ChatOpenAI, prompt: str) -> str:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    return llm.invoke(messages).content.strip()


def clean(text: str) -> str:
    """Strip LaTeX and markdown formatting. Never touch content structure."""
    text = re.sub(r'\\\(|\\\)', '', text)
    text = re.sub(r'\\\[|\\\]', '', text)
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\$([^$\n]+?)\$', r'\1', text)
    text = re.sub(r'\\([(){}\[\]])', r'\1', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\\\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ── Node: RAG context ──────────────────────────────────────────────────────────

def node_retrieve_context(state: LessonPlanState) -> LessonPlanState:
    try:
        from embedding_manager import load_vectorstore, save_vectorstore

        if state.get("use_saved_embedding") and state.get("saved_embedding_alias"):
            embeddings = get_embeddings()
            vectorstore = load_vectorstore(state["saved_embedding_alias"], embeddings)
        else:
            vectorstore = build_vector_store_from_pdf(state["textbook_pdf_path"])
            if state.get("save_new_embedding") and state.get("new_embedding_alias", "").strip():
                pdf_name = os.path.basename(state["textbook_pdf_path"])
                save_vectorstore(vectorstore, state["new_embedding_alias"], pdf_name)

        context = retrieve_context(vectorstore, state["learning_outcome"])
        return {**state, "context": context, "error": None}
    except Exception as e:
        return {**state, "context": "", "error": f"Context retrieval error: {str(e)}"}


# ── Nodes: Assessment (3 separate cells) ──────────────────────────────────────

def node_assess_time(state: LessonPlanState) -> LessonPlanState:
    """R4 part 1: just the time and marks line."""
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}

শুধুমাত্র একটি লাইন লেখো — পরীক্ষার সময় ও পূর্ণমান।
ঠিক এই ফরম্যাটে: সময়: ৫ মিনিট | পূর্ণমান: ৬ মার্ক
অন্য কিছু লিখবে না।"""
    return {**state, "assess_time": clean(call_llm(llm, prompt))}


def node_assess_questions(state: LessonPlanState) -> LessonPlanState:
    """R4 part 2: the actual MCQ + calculation questions."""
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}
পাঠ্যপুস্তক প্রসঙ্গ: {state["context"]}

ঠিক দুটি প্রশ্ন লেখো। অন্য কিছু লিখবে না — কোনো শিরোনাম নয়, কোনো ব্যাখ্যা নয়।

প্রশ্ন ১) একটি বহুনির্বাচনি প্রশ্ন (২ মার্ক) — বাস্তব জীবনের প্রেক্ষাপটে।
ক) ... খ) ... গ) ... ঘ) ...

প্রশ্ন ২) একটি গণনামূলক প্রশ্ন (৪ মার্ক) — সরাসরি গণিত লেখো।"""
    return {**state, "assess_questions": clean(call_llm(llm, prompt))}


def node_assess_exemplar(state: LessonPlanState) -> LessonPlanState:
    """R6: step-by-step answers. Explicitly receives the questions as context."""
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""নিচের প্রশ্নগুলোর আদর্শ উত্তর ধাপে ধাপে লেখো।

প্রশ্নসমূহ:
{state["assess_questions"]}

নিয়ম:
- শুধু উত্তর লেখো, প্রশ্ন পুনরায় লিখবে না।
- প্রতিটি উত্তর নম্বর দিয়ে শুরু করো: ১) উত্তর: ...
- গণিত সরাসরি লেখো: 3(x+2) = 3x+6
- ধাপে ধাপে সমাধান দেখাও।"""
    return {**state, "assess_exemplar": clean(call_llm(llm, prompt))}


# ── Nodes: Lesson Vision (5 separate cells) ───────────────────────────────────

def node_vision_why_ac(state: LessonPlanState) -> LessonPlanState:
    """R8 C0 block: WHY (Academic)"""
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}

শুধু ২-৩ বাক্য লেখো: এই বিষয়টি শিখলে একাডেমিক দিক থেকে কী কী সুবিধা পাবে?
কোনো শিরোনাম বা লেবেল লিখবে না — সরাসরি বাক্য দিয়ে শুরু করো।"""
    return {**state, "vision_why_ac": clean(call_llm(llm, prompt))}


def node_vision_why_no(state: LessonPlanState) -> LessonPlanState:
    """R8 C0 block: WHY (Non-Academic)"""
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}

শুধু ২-৩ বাক্য লেখো: দৈনন্দিন বাস্তব জীবনে এই বিষয়টি কীভাবে কাজে লাগে?
কোনো শিরোনাম বা লেবেল লিখবে না — সরাসরি বাক্য দিয়ে শুরু করো।"""
    return {**state, "vision_why_no": clean(call_llm(llm, prompt))}


def node_vision_what(state: LessonPlanState) -> LessonPlanState:
    """R8 C0 block: WHAT (Concept)"""
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}
পাঠ্যপুস্তক প্রসঙ্গ: {state["context"]}

শুধু ২-৩ বাক্য লেখো: এই টপিকের মূল ধারণা কী? সংক্ষিপ্ত সংজ্ঞা ও উদাহরণসহ।
গণিতের উদাহরণ সরাসরি লেখো: যেমন 3(x+2) = 3x+6
কোনো শিরোনাম বা লেবেল লিখবে না।"""
    return {**state, "vision_what": clean(call_llm(llm, prompt))}


def node_key_points(state: LessonPlanState) -> LessonPlanState:
    """R8 C1: How steps"""
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}

শিক্ষার্থীরা কীভাবে এই সমস্যা সমাধান করবে তার সর্বোচ্চ ৪টি ধাপ নম্বর দিয়ে লেখো।
কোনো শিরোনাম লিখবে না — সরাসরি ধাপ দিয়ে শুরু করো।
উদাহরণ ফরম্যাট:
১) সমস্যাটি মনোযোগ দিয়ে পড়ো।
২) কোন অপারেশন দরকার তা চিহ্নিত করো।"""
    return {**state, "key_points": clean(call_llm(llm, prompt))}


def node_knowledge_text(state: LessonPlanState) -> LessonPlanState:
    """R10 C0: knowledge from LO — short conceptual sentence."""
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}
WHAT ধারণা: {state["vision_what"]}

এই পাঠের জন্য শিক্ষার্থীদের কোন মূল গাণিতিক জ্ঞান প্রয়োজন?
শুধু ১-২টি সংক্ষিপ্ত বাক্য লেখো। কোনো শিরোনাম নয়।"""
    return {**state, "knowledge_text": clean(call_llm(llm, prompt))}


def node_blooms_skills(state: LessonPlanState) -> LessonPlanState:
    """R12 C0: Bloom's verb list — 3 verbs, one per line."""
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}

Bloom's Taxonomy অনুযায়ী এই পাঠের জন্য ৩টি ক্রিয়া-দক্ষতা লেখো।
প্রতিটি একটি লাইনে, বাংলায়, কোনো নম্বর বা বুলেট ছাড়া।
উদাহরণ ফরম্যাট:
বিশ্লেষণ করা (Analyze)
রাশি গঠন করা (Construct)
সমস্যা সমাধান করা (Solve)"""
    return {**state, "blooms_skills": clean(call_llm(llm, prompt))}


# ── Nodes: Launch (3 cells) ───────────────────────────────────────────────────

def node_launch_teacher(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}
বিষয়: {state["subject"]} | শ্রেণি: {state["grade"]}

Launch পর্বে (৫ মিনিট) শিক্ষকের কাজ লেখো।
নিচের ক্রমে লেখো — প্রতিটি ধাপ নতুন লাইনে:

SEL মুড চেকার:
(শিক্ষক ফিস্ট টু ফাইভ পদ্ধতিতে কী বলবেন — ১-২ বাক্য)

কৌতূহল জাগানো:
(টপিক সম্পর্কিত সহজ গল্প বা প্রশ্ন — ২-৩ বাক্য)

শিক্ষার ফলাফল ঘোষণা:
(আজকের লক্ষ্য — ১ বাক্য)

পূর্বজ্ঞান যাচাই:
(একটি প্রশ্ন শিক্ষার্থীদের উদ্দেশ্যে)

কোনো অতিরিক্ত শিরোনাম বা মার্কডাউন যোগ করবে না।"""
    return {**state, "launch_teacher": clean(call_llm(llm, prompt))}


def node_launch_student(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""Launch পর্বে শিক্ষক এটি করবেন:
{state["launch_teacher"]}

শিক্ষার্থীরা এই পর্বে কী কী করবে? ৩-৪টি সংক্ষিপ্ত বাক্যে লেখো।
কোনো শিরোনাম বা লেবেল নয় — সরাসরি বাক্য দিয়ে শুরু করো।"""
    return {**state, "launch_student": clean(call_llm(llm, prompt))}


def node_launch_materials(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""Launch পর্বের জন্য প্রয়োজনীয় উপকরণের তালিকা লেখো।
প্রতিটি উপকরণ আলাদা লাইনে। কোনো নম্বর বা বুলেট নয়।
বিষয়: {state["subject"]} | শ্রেণি: {state["grade"]}"""
    return {**state, "launch_materials": clean(call_llm(llm, prompt))}


# ── Nodes: Explore (3 cells) ──────────────────────────────────────────────────

def node_explore_teacher(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}
পাঠ্যপুস্তক প্রসঙ্গ: {state["context"]}

Explore পর্বে (৭-১০ মিনিট) শিক্ষকের কাজ লেখো।
প্রতিটি উপ-বিভাগ নিচের ঠিক এই শিরোনামে (একা একটি লাইনে):

Teacher Action
শিক্ষার্থীদের সামনে একটি বাস্তবধর্মী সমস্যা বা পরিস্থিতি উপস্থাপন করো।
শিক্ষক বোর্ডে কী লিখবেন বা করবেন — ২-৩ বাক্য। সরাসরি গণিত লেখো: 3(x+2)

গুরুত্বপূর্ণ: শিরোনামগুলো হুবহু উপরের ইংরেজিতে লিখবে। কোনো ** বা # নয়।"""
    return {**state, "explore_teacher": clean(call_llm(llm, prompt))}


def node_explore_student(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""Explore পর্বে শিক্ষক এটি করবেন:
{state["explore_teacher"]}

শিক্ষার্থীরা এই পর্বে কী কী করবে? ৩-৪টি সংক্ষিপ্ত বাক্যে।
কোনো শিরোনাম বা লেবেল নয়।"""
    return {**state, "explore_student": clean(call_llm(llm, prompt))}


def node_explore_materials(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""Explore পর্বের জন্য প্রয়োজনীয় উপকরণের তালিকা।
প্রতিটি আলাদা লাইনে। কোনো নম্বর বা বুলেট নয়।
কার্যক্রম: {state["explore_teacher"][:100]}"""
    return {**state, "explore_materials": clean(call_llm(llm, prompt))}


# ── Nodes: Conceptualize (3 cells) ────────────────────────────────────────────

def node_concept_teacher(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}
পাঠ্যপুস্তক প্রসঙ্গ: {state["context"]}
How ধাপগুলো: {state["key_points"]}

Conceptualize পর্বে (৮ মিনিট) শিক্ষকের কাজ লেখো।
প্রতিটি উপ-বিভাগ নিচের ঠিক এই শিরোনামে (একা একটি লাইনে):

Suggested Time: 8 minutes (Abstracting the Steps + Generalization)
(এই লাইনটি হুবহু লিখবে)

Teacher Action
শিক্ষক বোর্ডে কী লিখবেন বা করবেন — ধাপে ধাপে। সরাসরি গণিত লেখো: 2(x+3) = 2x+6

Pictorial / Representation / Demonstration
শিক্ষক বোর্ডে উদাহরণ দেন। ধাপে ধাপে সম্পূর্ণ সমাধান দেখাও।

গুরুত্বপূর্ণ: শিরোনামগুলো হুবহু উপরের ইংরেজিতে লিখবে। কোনো ** বা # নয়।"""
    return {**state, "concept_teacher": clean(call_llm(llm, prompt))}


def node_concept_student(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""Conceptualize পর্বে শিক্ষক এটি করবেন:
{state["concept_teacher"]}

শিক্ষার্থীরা এই পর্বে কী কী করবে? ৩-৪টি সংক্ষিপ্ত বাক্যে।
কোনো শিরোনাম বা লেবেল নয়।"""
    return {**state, "concept_student": clean(call_llm(llm, prompt))}


def node_concept_materials(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""Conceptualize পর্বের উপকরণ তালিকা।
প্রতিটি আলাদা লাইনে। কোনো নম্বর বা বুলেট নয়।"""
    return {**state, "concept_materials": clean(call_llm(llm, prompt))}


# ── Nodes: Guided Practice (3 cells) ─────────────────────────────────────────

def node_guided_teacher(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}
পাঠ্যপুস্তক প্রসঙ্গ: {state["context"]}
How ধাপগুলো: {state["key_points"]}

Guided Practice পর্বে (১০ মিনিট) শিক্ষকের কাজ লেখো।
প্রতিটি উপ-বিভাগ নিচের ঠিক এই শিরোনামে (একা একটি লাইনে):

Teacher Action
শিক্ষক কীভাবে এই অনুশীলন পরিচালনা করবেন — ২-৩ বাক্য।

Problem 1:
একটি অনুশীলন সমস্যা। সরাসরি গণিত লেখো।

Problem 2:
আরেকটি সমস্যা (বাস্তব প্রেক্ষাপটে)।

Teacher Feedback
শিক্ষার্থীদের উত্তরে কীভাবে feedback দেবেন — ২-৩ উদাহরণ।

গুরুত্বপূর্ণ: শিরোনামগুলো হুবহু উপরের ইংরেজিতে লিখবে। কোনো ** বা # নয়।"""
    return {**state, "guided_teacher": clean(call_llm(llm, prompt))}


def node_guided_student(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""Guided Practice পর্বে শিক্ষক এটি করবেন:
{state["guided_teacher"]}

শিক্ষার্থীরা এই পর্বে কী কী করবে? ৩-৪টি সংক্ষিপ্ত বাক্যে।
কোনো শিরোনাম বা লেবেল নয়।"""
    return {**state, "guided_student": clean(call_llm(llm, prompt))}


def node_guided_materials(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""Guided Practice পর্বের উপকরণ তালিকা।
প্রতিটি আলাদা লাইনে। কোনো নম্বর বা বুলেট নয়।"""
    return {**state, "guided_materials": clean(call_llm(llm, prompt))}


# ── Nodes: Independent Practice (3 cells) ────────────────────────────────────

def node_indep_teacher(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}
Guided Practice সমস্যা: {state["guided_teacher"][:300]}

Independent Practice পর্বে (৮ মিনিট) শিক্ষকের কাজ লেখো।
(Guided Practice এর চেয়ে সামান্য কঠিন সমস্যা দিতে হবে)
প্রতিটি উপ-বিভাগ নিচের ঠিক এই শিরোনামে (একা একটি লাইনে):

Teacher Action
শিক্ষক কীভাবে একা কাজের নির্দেশনা দেবেন — ২-৩ বাক্য।

Task 1:
একটি স্বতন্ত্র অনুশীলন সমস্যা। সরাসরি গণিত লেখো।

Task 2:
আরেকটি সমস্যা (একটু বেশি কঠিন)।

গুরুত্বপূর্ণ: শিরোনামগুলো হুবহু উপরের ইংরেজিতে লিখবে। কোনো ** বা # নয়।"""
    return {**state, "indep_teacher": clean(call_llm(llm, prompt))}


def node_indep_student(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""Independent Practice পর্বে শিক্ষক এটি করবেন:
{state["indep_teacher"]}

শিক্ষার্থীরা এই পর্বে কী কী করবে? ৩-৪টি সংক্ষিপ্ত বাক্যে।
কোনো শিরোনাম বা লেবেল নয়।"""
    return {**state, "indep_student": clean(call_llm(llm, prompt))}


def node_indep_materials(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""Independent Practice পর্বের উপকরণ তালিকা।
প্রতিটি আলাদা লাইনে। কোনো নম্বর বা বুলেট নয়।"""
    return {**state, "indep_materials": clean(call_llm(llm, prompt))}


# ── Nodes: Lesson Closing (3 cells) ──────────────────────────────────────────

def node_closing_teacher(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}
মূল্যায়ন প্রশ্নসমূহ: {state["assess_questions"]}

Lesson Closing পর্বে (৭ মিনিট) শিক্ষকের কাজ লেখো।
প্রতিটি উপ-বিভাগ নিচের ঠিক এই শিরোনামে (একা একটি লাইনে):

Teacher Action
পাঠ শেষে পর্যালোচনা ও প্রশংসার কথা — ২-৩ বাক্য, মূল পয়েন্ট তালিকা।

Exit Ticket
নিচের assessment প্রশ্নগুলো হুবহু লিখবে:
{state["assess_questions"]}

গুরুত্বপূর্ণ: শিরোনামগুলো হুবহু উপরের ইংরেজিতে লিখবে। কোনো ** বা # নয়।"""
    return {**state, "closing_teacher": clean(call_llm(llm, prompt))}


def node_closing_student(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""Lesson Closing পর্বে শিক্ষক এটি করবেন:
{state["closing_teacher"]}

শিক্ষার্থীরা এই পর্বে কী কী করবে? ৩-৪টি সংক্ষিপ্ত বাক্যে।
কোনো শিরোনাম বা লেবেল নয়।"""
    return {**state, "closing_student": clean(call_llm(llm, prompt))}


def node_closing_materials(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""Lesson Closing পর্বের উপকরণ তালিকা।
প্রতিটি আলাদা লাইনে। কোনো নম্বর বা বুলেট নয়।"""
    return {**state, "closing_materials": clean(call_llm(llm, prompt))}


# ── Node: Assemble final dict ──────────────────────────────────────────────────

def node_assemble(state: LessonPlanState) -> LessonPlanState:
    lesson_plan = {
        # Header
        "teacher_name":     state["teacher_name"],
        "subject":          state["subject"],
        "grade":            state["grade"],
        "duration":         state["duration"],
        "learning_outcome": state["learning_outcome"],

        # Table 1 — per-cell fields (docx_generator reads these directly)
        "assess_time":      state.get("assess_time", ""),
        "assess_questions": state.get("assess_questions", ""),
        "assess_exemplar":  state.get("assess_exemplar", ""),
        "vision_why_ac":    state.get("vision_why_ac", ""),
        "vision_why_no":    state.get("vision_why_no", ""),
        "vision_what":      state.get("vision_what", ""),
        "key_points":       state.get("key_points", ""),
        "knowledge_text":   state.get("knowledge_text", ""),
        "blooms_skills":    state.get("blooms_skills", ""),

        # Table 2 — per-cell fields
        "launch_teacher":   state.get("launch_teacher", ""),
        "launch_student":   state.get("launch_student", ""),
        "launch_materials": state.get("launch_materials", ""),

        "explore_teacher":   state.get("explore_teacher", ""),
        "explore_student":   state.get("explore_student", ""),
        "explore_materials": state.get("explore_materials", ""),

        "concept_teacher":   state.get("concept_teacher", ""),
        "concept_student":   state.get("concept_student", ""),
        "concept_materials": state.get("concept_materials", ""),

        "guided_teacher":   state.get("guided_teacher", ""),
        "guided_student":   state.get("guided_student", ""),
        "guided_materials": state.get("guided_materials", ""),

        "indep_teacher":   state.get("indep_teacher", ""),
        "indep_student":   state.get("indep_student", ""),
        "indep_materials": state.get("indep_materials", ""),

        "closing_teacher":   state.get("closing_teacher", ""),
        "closing_student":   state.get("closing_student", ""),
        "closing_materials": state.get("closing_materials", ""),
    }
    return {**state, "lesson_plan": lesson_plan}


# ── Build graph ────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(LessonPlanState)

    nodes = [
        # RAG
        ("node_retrieve_context",    node_retrieve_context),
        # Assessment
        ("node_assess_time",         node_assess_time),
        ("node_assess_questions",    node_assess_questions),
        ("node_assess_exemplar",     node_assess_exemplar),   # depends on assess_questions
        # Vision
        ("node_vision_why_ac",       node_vision_why_ac),
        ("node_vision_why_no",       node_vision_why_no),
        ("node_vision_what",         node_vision_what),
        ("node_key_points",          node_key_points),
        ("node_knowledge_text",      node_knowledge_text),    # depends on vision_what
        ("node_blooms_skills",       node_blooms_skills),
        # Launch
        ("node_launch_teacher",      node_launch_teacher),
        ("node_launch_student",      node_launch_student),    # depends on launch_teacher
        ("node_launch_materials",    node_launch_materials),
        # Explore
        ("node_explore_teacher",     node_explore_teacher),
        ("node_explore_student",     node_explore_student),   # depends on explore_teacher
        ("node_explore_materials",   node_explore_materials),
        # Conceptualize
        ("node_concept_teacher",     node_concept_teacher),   # depends on key_points
        ("node_concept_student",     node_concept_student),   # depends on concept_teacher
        ("node_concept_materials",   node_concept_materials),
        # Guided Practice
        ("node_guided_teacher",      node_guided_teacher),    # depends on key_points
        ("node_guided_student",      node_guided_student),    # depends on guided_teacher
        ("node_guided_materials",    node_guided_materials),
        # Independent Practice
        ("node_indep_teacher",       node_indep_teacher),     # depends on guided_teacher
        ("node_indep_student",       node_indep_student),     # depends on indep_teacher
        ("node_indep_materials",     node_indep_materials),
        # Closing
        ("node_closing_teacher",     node_closing_teacher),   # depends on assess_questions
        ("node_closing_student",     node_closing_student),   # depends on closing_teacher
        ("node_closing_materials",   node_closing_materials),
        # Final
        ("node_assemble",            node_assemble),
    ]

    for name, fn in nodes:
        graph.add_node(name, fn)

    graph.set_entry_point("node_retrieve_context")
    names = [n for n, _ in nodes]
    for i in range(len(names) - 1):
        graph.add_edge(names[i], names[i + 1])
    graph.add_edge(names[-1], END)
    return graph.compile()


# ── Public entry point ─────────────────────────────────────────────────────────

def run_pipeline(
    teacher_name: str,
    subject: str,
    grade: str,
    duration: str,
    learning_outcome: str,
    textbook_pdf_path: str,
    model_name: str,
    use_saved_embedding: bool = False,
    saved_embedding_alias: str = "",
    save_new_embedding: bool = False,
    new_embedding_alias: str = "",
) -> dict:
    graph = build_graph()
    initial_state: LessonPlanState = {
        "teacher_name": teacher_name,
        "subject": subject,
        "grade": grade,
        "duration": duration,
        "learning_outcome": learning_outcome,
        "textbook_pdf_path": textbook_pdf_path,
        "model_name": model_name,
        "use_saved_embedding": use_saved_embedding,
        "saved_embedding_alias": saved_embedding_alias,
        "save_new_embedding": save_new_embedding,
        "new_embedding_alias": new_embedding_alias,
        "context": "",
        # Table 1
        "assess_time": "", "assess_questions": "", "assess_exemplar": "",
        "vision_why_ac": "", "vision_why_no": "", "vision_what": "",
        "key_points": "", "knowledge_text": "", "blooms_skills": "",
        # Table 2
        "launch_teacher": "", "launch_student": "", "launch_materials": "",
        "explore_teacher": "", "explore_student": "", "explore_materials": "",
        "concept_teacher": "", "concept_student": "", "concept_materials": "",
        "guided_teacher": "", "guided_student": "", "guided_materials": "",
        "indep_teacher": "", "indep_student": "", "indep_materials": "",
        "closing_teacher": "", "closing_student": "", "closing_materials": "",
        "lesson_plan": {},
        "error": None,
    }
    final_state = graph.invoke(initial_state, config={"recursion_limit": 100})
    if final_state.get("error"):
        raise RuntimeError(final_state["error"])
    return final_state["lesson_plan"]