"""
LangGraph pipeline — per-cell generation with explicit context chaining.

Cell → State field mapping (mirrors docx_generator exactly):
  Table 0  : teacher_name, subject, grade, duration        (user inputs, no LLM)
  Table 1  :
    R2      : lo_text          — learning outcome           (user input, no LLM)
    R4      : assess_time      — "সময়: X | পূর্ণমান: Y"
    R4      : assess_questions — MCQ + calculation question
    R6      : assess_exemplar  — step-by-step answers
    R8 C0   : vision_what / vision_why_ac / vision_why_no
    R8 C1   : key_points
    R10 C0  : knowledge_text
    R12 C0  : blooms_skills
  Table 2  : launch/explore/concept/guided/indep/closing × teacher/student/materials
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
    manual_context: str          # NEW: bypasses PDF/embedding entirely
    user_assess_questions: str   # NEW: optional user-provided assessment questions

    # RAG context
    context: str

    # Table 1
    assess_time: str
    assess_questions: str
    assess_exemplar: str
    vision_why_ac: str
    vision_why_no: str
    vision_what: str
    key_points: str
    knowledge_text: str
    blooms_skills: str

    # Table 2
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

    lesson_plan: dict
    error: Optional[str]


# ── System prompt ──────────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """তুমি একজন অভিজ্ঞ শিক্ষক যিনি বাংলাদেশের পাঠ্যক্রম অনুযায়ী পাঠ পরিকল্পনা তৈরি করেন।

কঠোর নিয়মসমূহ — প্রতিটি নিয়ম অবশ্যই মেনে চলতে হবে:
১) সম্পূর্ণ পাঠ পরিকল্পনা শুদ্ধ বাংলায় (Raw Bengali) লিখতে হবে। কোনো Banglish ব্যবহার করা যাবে না।
২) কোনো মার্কডাউন নয়। কোনো ** বা * বা # চিহ্ন ব্যবহার করবে না।
৩) কোনো LaTeX নয়: কোনো \\( \\) নেই, কোনো $ $ নেই।
৪) গণিতের সূত্র ও সংখ্যা copy-paste friendly plain text-এ লেখো: যেমন (a+b)^2 = a^2 + 2ab + b^2
৫) তুমি শুধু যা চাওয়া হয়েছে তাই লিখবে — অতিরিক্ত কোনো শিরোনাম, লেবেল বা ব্যাখ্যা যোগ করবে না।

লেখার ধরন ও ফরম্যাট:
- পাঠ পরিকল্পনা একটি ক্লাসরুমের দৃশ্য বর্ণনার মতো লেখো — শিক্ষক ধাপে ধাপে কী করেন ও কী বলেন তা দেখাও।
- বেশিরভাগ অংশ অনুচ্ছেদ আকারে (paragraph) লেখো, বুলেট-ভারী নয়।
- তবে কোনো নির্দেশনা বা "What To Do" অংশ পরিষ্কার নম্বরযুক্ত পয়েন্টে লেখো (ঠিক ৪টি ধাপ)।
- Teacher Action ও Teacher Statement আলাদাভাবে বোঝা যাবে এমনভাবে লেখো।
- প্রতিটি Teacher Action এর সাথে Student Action থাকবে — শিক্ষার্থীরা কী করছে, ভাবছে, বলছে তা সুনির্দিষ্ট ও পর্যবেক্ষণযোগ্যভাবে লেখো।

শিক্ষকের কথা ও কাজ লেখার নিয়ম (বৈচিত্র্য আনো — একই বাক্যাংশ বারবার ব্যবহার করো না):
- শিক্ষকের কথা বোঝাতে বিভিন্ন রূপ ব্যবহার করো, যেমন:
  শিক্ষক বলেন, "..."  /  শিক্ষক জিজ্ঞেস করেন, "..."  /  শিক্ষক ব্যাখ্যা করেন, "..."
  শিক্ষক শিক্ষার্থীদের উদ্দেশ্যে বলেন, "..."  /  এরপর শিক্ষক জানতে চান, "..."
  শিক্ষক উৎসাহ দিয়ে বলেন, "..."  /  শিক্ষক নির্দেশ দেন, "..."
- শিক্ষকের বোর্ডে লেখা বোঝাতে:
  শিক্ষক বোর্ডে লেখেন, "..."  /  শিক্ষক বোর্ডে সমস্যাটি তুলে ধরেন  /  বোর্ডে ধাপগুলো লেখা হয়
  শিক্ষক বোর্ডে উদাহরণ দেখান  /  শিক্ষক বোর্ডে সমাধান করে দেখান
- শিক্ষকের কাজ বোঝাতে:
  শিক্ষক ঘুরে ঘুরে দেখেন  /  শিক্ষক পর্যবেক্ষণ করেন  /  শিক্ষক দলে দলে গিয়ে সাহায্য করেন
  এরপর শিক্ষক ক্লাসের দিকে ফিরে  /  শিক্ষক শিক্ষার্থীদের কাজ দেখে
- গুরুত্বপূর্ণ: পরপর দুটি বাক্য একই "শিক্ষক বলেন" বা "শিক্ষক লিখেন" দিয়ে শুরু করা যাবে না। প্রতিবার ভিন্ন রূপ ব্যবহার করো।

Assessment ও Exit Ticket নিয়ম:
- Assessment এর প্রথম প্রশ্নটি হুবহু Exit Ticket-এ পুনরায় ব্যবহার করতে হবে।
- সঠিক উত্তরের লেবেল: ESR (Expected Student Response)

How Keypoint নিয়ম:
- How ধাপগুলো সুনির্দিষ্ট procedural steps (কীভাবে সমাধান করবে) হবে।
- এই ধরনের অস্পষ্ট কথা How ধাপে লেখা যাবে না: "সমস্যাটি মনোযোগ দিয়ে পড়ো", "সমস্যাটি পড়ো", "প্রশ্নটি পড়ো"।
- Conceptualize পর্বে এই How ধাপগুলো হুবহু ব্যবহার করতে হবে।

Precise Praise (নির্দিষ্ট প্রশংসা):
- সাধারণ "Good job" না বলে নির্দিষ্ট কাজের প্রশংসা করো।
- যেমন: "আমি দেখতে পাচ্ছি রাফিক ইতিমধ্যে প্রথম দুটি ধাপ সঠিকভাবে শেষ করেছে।"

Positive Correction (ইতিবাচক সংশোধন):
- ভুল সংশোধন করো কিন্তু লজ্জা দিও না।
- যেমন: "তোমার চেষ্টাটা চমৎকার হয়েছে, তবে সূত্রের এই অংশটা আরেকবার মিলিয়ে দেখো তো?"

গুরুত্বপূর্ণ (Context ব্যবহারের নিয়ম):
- তোমাকে পাঠ্যপুস্তক থেকে কিছু context দেওয়া হতে পারে। এই context-এর সব অংশ প্রাসঙ্গিক নাও হতে পারে।
- শুধুমাত্র Learning Outcome এর সাথে সরাসরি সম্পর্কিত context ব্যবহার করো।
- যদি context-এ অন্য অধ্যায়ের (অপ্রাসঙ্গিক) বিষয় থাকে, সেগুলো সম্পূর্ণ উপেক্ষা করো।
- যদি কোনো প্রাসঙ্গিক context না থাকে, তোমার নিজের জ্ঞান থেকে Learning Outcome অনুযায়ী উত্তর দাও।
- কখনোই অপ্রাসঙ্গিক context থেকে উদাহরণ, সূত্র বা সমস্যা ব্যবহার করবে না।"""


def _build_system_prompt() -> str:
    """Build system prompt with active guidelines injected."""
    try:
        from guidelines_manager import build_system_prompt_section
        guidelines_section = build_system_prompt_section()
    except Exception:
        guidelines_section = ""
    if guidelines_section:
        return BASE_SYSTEM_PROMPT + "\n\n" + guidelines_section
    return BASE_SYSTEM_PROMPT


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


def build_vector_store_from_pdf(pdf_path: str, progress_callback=None) -> FAISS:
    """Build FAISS vector store from PDF. Falls back to OCR for vector-path PDFs.
    
    Detection logic:
      - If standard extraction gets 0 pages with text → OCR fallback
      - If standard extraction gets text but total lines < 60 → likely vector-path
        PDF with partial/garbage extraction → OCR fallback
    """
    import logging
    logger = logging.getLogger(__name__)

    def _log(msg):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    # ── Step 1: Try standard text extraction ─────────────────────────────
    _log("📄 PDF থেকে টেক্সট বের করার চেষ্টা করা হচ্ছে (standard extraction)...")

    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()

    _log(f"   PyPDFLoader: {len(raw_docs)} পৃষ্ঠা পড়া হয়েছে")

    # Filter pages with meaningful text (>20 chars)
    docs = [d for d in raw_docs if d.page_content and d.page_content.strip()
            and len(d.page_content.strip()) > 20]

    pages_with_text = len(docs)
    pages_without_text = len(raw_docs) - pages_with_text

    _log(f"   টেক্সটসহ পৃষ্ঠা: {pages_with_text} | খালি পৃষ্ঠা: {pages_without_text}")

    # Count total lines across all extracted text
    total_lines = 0
    total_chars = 0
    if docs:
        for d in docs:
            text = d.page_content.strip()
            total_lines += len([l for l in text.split("\n") if l.strip()])
            total_chars += len(text)

    _log(f"   মোট লাইন: {total_lines} | মোট অক্ষর: {total_chars}")

    # ── Step 2: Decide if OCR fallback is needed ─────────────────────────
    needs_ocr = False

    if pages_with_text == 0:
        _log("⚠️ কোনো পৃষ্ঠায় টেক্সট পাওয়া যায়নি — OCR fallback প্রয়োজন")
        needs_ocr = True
    elif total_lines < 60:
        _log(
            f"⚠️ মাত্র {total_lines} লাইন পাওয়া গেছে {pages_with_text} পৃষ্ঠায় — "
            f"সম্ভবত vector-based PDF। OCR fallback চেষ্টা করা হচ্ছে..."
        )
        needs_ocr = True
    elif pages_without_text > pages_with_text:
        _log(
            f"⚠️ বেশিরভাগ পৃষ্ঠা খালি ({pages_without_text}/{len(raw_docs)}) — "
            f"আংশিক vector-based PDF হতে পারে। OCR fallback চেষ্টা করা হচ্ছে..."
        )
        needs_ocr = True
    else:
        _log(f"✅ Standard extraction সফল — {pages_with_text} পৃষ্ঠা, {total_lines} লাইন")

    if needs_ocr:
        _log("🔍 OCR (Tesseract) দিয়ে টেক্সট বের করা হচ্ছে (এতে সময় লাগতে পারে)...")
        try:
            ocr_docs = _ocr_extract_pdf(pdf_path, progress_callback)
            if ocr_docs:
                ocr_lines = sum(
                    len([l for l in d.page_content.split("\n") if l.strip()])
                    for d in ocr_docs
                )
                _log(f"   OCR: {len(ocr_docs)} পৃষ্ঠা, {ocr_lines} লাইন পাওয়া গেছে")

                # Use OCR results if they're better than standard extraction
                if ocr_lines > total_lines:
                    _log(f"   OCR ফলাফল ভালো ({ocr_lines} > {total_lines} লাইন) — OCR ব্যবহার করা হচ্ছে")
                    docs = ocr_docs
                else:
                    _log(f"   OCR ফলাফল ভালো নয় ({ocr_lines} <= {total_lines}) — standard extraction রাখা হচ্ছে")
            else:
                _log("   OCR থেকেও কোনো টেক্সট পাওয়া যায়নি")
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            _log(f"   OCR ব্যর্থ: {str(e)[:100]}")

    if not docs:
        raise ValueError(
            "PDF থেকে কোনো টেক্সট বের করা যায়নি (সাধারণ ও OCR উভয় পদ্ধতিতে)। "
            "'সরাসরি context লিখুন' অপশন ব্যবহার করুন।"
        )

    # ── Step 3: Split into chunks ────────────────────────────────────────
    _log(f"✂️ টেক্সট chunk-এ ভাগ করা হচ্ছে...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", "।", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    _log(f"   প্রাথমিক chunking: {len(chunks)} chunks (size=1000, overlap=150)")

    if not chunks:
        splitter2 = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200, separators=[" ", ""],
        )
        chunks = splitter2.split_documents(docs)
        _log(f"   Fallback chunking: {len(chunks)} chunks (size=2000, overlap=200)")

    if not chunks:
        chunks = docs
        _log(f"   Raw pages as chunks: {len(chunks)}")

    # Log chunk size stats
    chunk_sizes = [len(c.page_content) for c in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    min_size = min(chunk_sizes) if chunk_sizes else 0
    max_size = max(chunk_sizes) if chunk_sizes else 0
    _log(f"   Chunk stats: মোট={len(chunks)}, গড়={avg_size:.0f} অক্ষর, "
         f"সর্বনিম্ন={min_size}, সর্বোচ্চ={max_size}")

    # ── Step 4: Create embeddings ────────────────────────────────────────
    _log(f"🧠 {len(chunks)} chunks-এর Embedding তৈরি হচ্ছে (OpenAI API)...")

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    _log(f"✅ FAISS vector store তৈরি সম্পন্ন — {vectorstore.index.ntotal} vectors")

    return vectorstore


def _ocr_extract_pdf(pdf_path: str, progress_callback=None) -> list:
    """Extract text from PDF using OCR (for scanned/vector-path PDFs)."""
    from langchain.schema import Document as LCDocument

    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        raise ImportError(
            "OCR dependencies not available. "
            "Install: pip install pdf2image pytesseract"
        )

    # Determine available languages
    try:
        available_langs = pytesseract.get_languages()
    except Exception:
        available_langs = ["eng"]

    lang = "ben+eng" if "ben" in available_langs else "eng"

    # Convert PDF pages to images in batches to manage memory
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    docs = []
    batch_size = 10

    for start in range(0, total_pages, batch_size):
        end = min(start + batch_size, total_pages)
        if progress_callback:
            progress_callback(f"OCR: পৃষ্ঠা {start+1}-{end}/{total_pages} প্রসেস হচ্ছে...")

        try:
            images = convert_from_path(
                pdf_path,
                first_page=start + 1,
                last_page=end,
                dpi=200,
                fmt="jpeg",
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"pdf2image failed for pages {start+1}-{end}: {e}")
            continue

        for i, img in enumerate(images):
            page_num = start + i + 1
            try:
                text = pytesseract.image_to_string(img, lang=lang)
                text = text.strip()
                if text and len(text) > 30:
                    docs.append(LCDocument(
                        page_content=text,
                        metadata={"source": pdf_path, "page": page_num - 1},
                    ))
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"OCR failed on page {page_num}: {e}")
            finally:
                del img  # free memory

        del images

    return docs


def retrieve_context_with_scores(
    vectorstore: FAISS, learning_outcome: str, k: int = 6
) -> list[tuple]:
    """Retrieve context chunks with their similarity scores.

    Builds a better search query from the learning outcome by:
    1. Using the raw LO as primary query
    2. Also searching with key topic words extracted from LO
    3. Merging and deduplicating results

    Returns list of (text, score) tuples sorted by score (lower = more similar).
    No filtering is done here — the UI handles that.
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        total_docs = vectorstore.index.ntotal
        if total_docs == 0:
            return []
        safe_k = min(k, total_docs)

        # Primary search: full LO text
        results1 = vectorstore.similarity_search_with_score(
            learning_outcome, k=safe_k
        )

        # Secondary search: build a focused topic query
        # Strip common Bengali verb endings and helper words to get topic keywords
        import re
        # Remove common suffixes: করতে পারবে, নির্ণয় করে, সমাধান করতে, ব্যাখ্যা করতে, etc.
        topic_query = learning_outcome
        for phrase in [
            "করতে পারবে", "করে সমস্যা সমাধান করতে পারবে", "সমস্যা সমাধান করতে পারবে",
            "ব্যাখ্যা করতে পারবে", "প্রয়োগ করতে পারবে", "নির্ণয় করতে পারবে",
            "গঠন করতে পারবে", "অঙ্কন করতে পারবে", "চিহ্নিত করতে পারবে",
            "পারবে", "করে", "করতে",
        ]:
            topic_query = topic_query.replace(phrase, "").strip()
        topic_query = re.sub(r'\s+', ' ', topic_query).strip()

        logger.info(f"Retrieval queries: LO='{learning_outcome[:60]}...' | Topic='{topic_query}'")

        results2 = []
        if topic_query and topic_query != learning_outcome:
            results2 = vectorstore.similarity_search_with_score(
                topic_query, k=safe_k
            )

        # Merge: keep best score for each unique chunk
        seen_texts = {}
        for doc, score in results1 + results2:
            text = doc.page_content
            text_key = text[:200]  # use first 200 chars as dedup key
            if text_key not in seen_texts or score < seen_texts[text_key][1]:
                seen_texts[text_key] = (text, round(float(score), 3))

        # Sort by score (lower = better) and take top k
        sorted_results = sorted(seen_texts.values(), key=lambda x: x[1])[:safe_k]

        for text, score in sorted_results:
            preview = text[:80].replace("\n", " ")
            logger.info(f"  Retrieval score={score:.3f} : {preview}...")

        return sorted_results

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"retrieve_context error: {e}")
        return []


def get_all_chunks_as_context(vectorstore: FAISS) -> str:
    """Return ALL chunks from the vector store concatenated as context.
    Used when the user wants to use the entire book."""
    import logging
    logger = logging.getLogger(__name__)

    try:
        total = vectorstore.index.ntotal
        if total == 0:
            return ""
        # Retrieve all using a dummy query with k=total
        results = vectorstore.similarity_search("", k=total)
        logger.info(f"Full-book context: {len(results)} chunks, {total} vectors")
        return "\n\n---\n\n".join(doc.page_content for doc in results)
    except Exception as e:
        logger.warning(f"get_all_chunks error: {e}")
        return ""


def retrieve_context(vectorstore: FAISS, learning_outcome: str, k: int = 6) -> str:
    """Legacy wrapper — returns filtered context as a single string."""
    results = retrieve_context_with_scores(vectorstore, learning_outcome, k=k)
    if not results:
        return ""
    return "\n\n---\n\n".join(text for text, score in results)


def call_llm(llm: ChatOpenAI, prompt: str) -> str:
    messages = [
        SystemMessage(content=_build_system_prompt()),
        HumanMessage(content=prompt),
    ]
    return llm.invoke(messages).content.strip()


def clean(text: str) -> str:
    """Strip LaTeX and markdown formatting."""
    text = re.sub(r'\\\(|\\\)', '', text)
    text = re.sub(r'\\\[|\\\]', '', text)
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\$([^$\n]+?)\$', r'\1', text)
    text = re.sub(r'\\([(){}\\[\\]])', r'\1', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\\\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _ctx(state: dict, lo_key: str = "learning_outcome") -> str:
    """Build context block for prompts with strict relevance instructions.
    
    Instead of dumping raw context, this wraps it with explicit instructions
    to only use parts relevant to the learning outcome.
    """
    context = state.get("context", "").strip()
    lo = state.get(lo_key, "").strip()
    if not context:
        return ""
    return (
        f"নিচে পাঠ্যপুস্তক থেকে কিছু অংশ দেওয়া হলো। এর মধ্যে কিছু অংশ Learning Outcome "
        f"'{lo}' এর সাথে প্রাসঙ্গিক নাও হতে পারে।\n"
        f"সতর্কতা: শুধুমাত্র '{lo}' বিষয়ের সাথে সরাসরি সম্পর্কিত অংশ ব্যবহার করো। "
        f"অন্য অধ্যায়ের সূত্র, উদাহরণ, বা সমস্যা ব্যবহার করো না। "
        f"যদি নিচের কোনো অংশই প্রাসঙ্গিক না হয়, তাহলে context উপেক্ষা করে "
        f"তোমার নিজের জ্ঞান থেকে '{lo}' বিষয়ে উত্তর দাও।\n\n"
        f"পাঠ্যপুস্তকের অংশসমূহ:\n{context}"
    )


# Anti-repetition suffix — appended to every teacher action prompt
VARIETY_RULE = """

অত্যন্ত গুরুত্বপূর্ণ নিয়ম — এটি ভঙ্গ করলে পুরো উত্তর বাতিল হবে:
১) পরপর দুটি বাক্য "শিক্ষক বলেন" দিয়ে শুরু করা নিষিদ্ধ।
২) পরপর দুটি বাক্য "শিক্ষক" শব্দ দিয়ে শুরু করা নিষিদ্ধ।
৩) প্রতিটি বাক্য ভিন্নভাবে শুরু করো। নিচের তালিকা থেকে পালা করে ব্যবহার করো:
   - শিক্ষক বলেন, "..."
   - এরপর জিজ্ঞেস করেন, "..."
   - বোর্ডে লেখা হয়: ...
   - শিক্ষার্থীদের উদ্দেশ্যে নির্দেশ দেন, "..."
   - ব্যাখ্যা করতে গিয়ে বলেন, "..."
   - উৎসাহ দিয়ে বলেন, "..."
   - এবার শিক্ষক জানতে চান, "..."
৪) লেখা শেষে নিজে যাচাই করো — কোনো পরপর দুটি বাক্য একই শব্দে শুরু হয়েছে কি না।"""


# ── Node: RAG context ──────────────────────────────────────────────────────────

def node_retrieve_context(state: LessonPlanState) -> LessonPlanState:
    try:
        # Option 1: Manual context provided directly — skip PDF/embedding entirely
        if state.get("manual_context", "").strip():
            return {**state, "context": state["manual_context"].strip(), "error": None}

        from embedding_manager import load_vectorstore, save_vectorstore

        # Option 2: Load saved embedding
        if state.get("use_saved_embedding") and state.get("saved_embedding_alias"):
            embeddings = get_embeddings()
            vectorstore = load_vectorstore(state["saved_embedding_alias"], embeddings)

        # Option 3: Build from uploaded PDF
        else:
            if not state.get("textbook_pdf_path", "").strip():
                raise ValueError(
                    "কোনো PDF পাথ বা manual context দেওয়া হয়নি। "
                    "PDF আপলোড করুন অথবা সরাসরি context লিখুন।"
                )
            vectorstore = build_vector_store_from_pdf(state["textbook_pdf_path"])
            if state.get("save_new_embedding") and state.get("new_embedding_alias", "").strip():
                pdf_name = os.path.basename(state["textbook_pdf_path"])
                save_vectorstore(vectorstore, state["new_embedding_alias"], pdf_name)

        context = retrieve_context(vectorstore, state["learning_outcome"])
        return {**state, "context": context, "error": None}

    except Exception as e:
        return {**state, "context": "", "error": f"Context retrieval error: {str(e)}"}


# ── Nodes: Assessment ──────────────────────────────────────────────────────────

def node_assess_time(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}

শুধুমাত্র একটি লাইন লেখো — পরীক্ষার সময় ও পূর্ণমান।
ঠিক এই ফরম্যাটে: সময়: ৫ মিনিট | পূর্ণমান: ৬ মার্ক
অন্য কিছু লিখবে না।"""
    return {**state, "assess_time": clean(call_llm(llm, prompt))}


def node_assess_questions(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state

    # If user provided their own assessment questions, use them directly
    user_q = state.get("user_assess_questions", "").strip()
    if user_q:
        return {**state, "assess_questions": clean(user_q)}

    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}
{_ctx(state)}

ঠিক দুটি প্রশ্ন লেখো। প্রশ্নগুলো অবশ্যই "{state["learning_outcome"]}" এর সাথে সরাসরি সম্পর্কিত হতে হবে।
অন্য কিছু লিখবে না — কোনো শিরোনাম নয়, কোনো ব্যাখ্যা নয়।

প্রশ্ন ১) একটি বহুনির্বাচনি প্রশ্ন (২ মার্ক) — বাস্তব জীবনের প্রেক্ষাপটে।
ক) ... খ) ... গ) ... ঘ) ...

প্রশ্ন ২) একটি গণনামূলক প্রশ্ন (৪ মার্ক)।"""
    return {**state, "assess_questions": clean(call_llm(llm, prompt))}


def node_assess_exemplar(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""নিচের প্রশ্নগুলোর সঠিক উত্তর লেখো।

প্রশ্নসমূহ:
{state["assess_questions"]}

নিয়ম:
- শুধু উত্তর লেখো, প্রশ্ন পুনরায় লিখবে না।
- প্রতিটি উত্তর নম্বর দিয়ে শুরু করো: ১) উত্তর: ...
- বহুনির্বাচনি প্রশ্নে শুধু সঠিক অপশন লেখো।
- গণনামূলক প্রশ্নে ধাপে ধাপে সমাধান দেখাও — সংক্ষেপে, কোনো অতিরিক্ত ব্যাখ্যা ছাড়া।"""
    return {**state, "assess_exemplar": clean(call_llm(llm, prompt))}


# ── Nodes: Lesson Vision ───────────────────────────────────────────────────────

def node_vision_why_ac(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}

শুধু ২-৩ বাক্য লেখো: এই বিষয়টি শিখলে একাডেমিক দিক থেকে কী কী সুবিধা পাবে?
কোনো শিরোনাম বা লেবেল লিখবে না — সরাসরি বাক্য দিয়ে শুরু করো।"""
    return {**state, "vision_why_ac": clean(call_llm(llm, prompt))}


def node_vision_why_no(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}

শুধু ২-৩ বাক্য লেখো: দৈনন্দিন বাস্তব জীবনে এই বিষয়টি কীভাবে কাজে লাগে?
কোনো শিরোনাম বা লেবেল লিখবে না — সরাসরি বাক্য দিয়ে শুরু করো।"""
    return {**state, "vision_why_no": clean(call_llm(llm, prompt))}


def node_vision_what(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}
{_ctx(state)}

শুধু ২-৩ বাক্য লেখো: "{state["learning_outcome"]}" এই টপিকের মূল ধারণা কী? সংক্ষিপ্ত সংজ্ঞা ও উদাহরণসহ।
কোনো শিরোনাম বা লেবেল লিখবে না।"""
    return {**state, "vision_what": clean(call_llm(llm, prompt))}


def node_key_points(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}
বিষয়: {state["subject"]} | শ্রেণি: {state["grade"]}

"{state["learning_outcome"]}" অর্জনের জন্য শিক্ষার্থীরা কীভাবে সমস্যা সমাধান করবে তার সুনির্দিষ্ট procedural ধাপ লেখো (৩-৪টি)।

নিয়ম:
- প্রতিটি ধাপ সুনির্দিষ্ট procedural step হবে (কীভাবে সমাধান করবে — step-by-step solving process)।
- নিচের ধরনের অস্পষ্ট কথা লেখা যাবে না:
  "সমস্যাটি মনোযোগ দিয়ে পড়ো", "সমস্যাটি পড়ো", "প্রশ্নটি পড়ো", "কোন অপারেশন দরকার তা চিহ্নিত করো"
- এই ধাপগুলো পরে Conceptualize পর্বে হুবহু ব্যবহার করা হবে।

কোনো শিরোনাম লিখবে না — সরাসরি নম্বরযুক্ত ধাপ দিয়ে শুরু করো।"""
    return {**state, "key_points": clean(call_llm(llm, prompt))}


def node_knowledge_text(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}
WHAT ধারণা: {state["vision_what"]}

এই পাঠের জন্য শিক্ষার্থীদের কোন মূল গাণিতিক জ্ঞান প্রয়োজন?
শুধু ১-২টি সংক্ষিপ্ত বাক্য লেখো। কোনো শিরোনাম নয়।"""
    return {**state, "knowledge_text": clean(call_llm(llm, prompt))}


def node_blooms_skills(state: LessonPlanState) -> LessonPlanState:
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


# ── Nodes: Launch ──────────────────────────────────────────────────────────────

def node_launch_teacher(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}
বিষয়: {state["subject"]} | শ্রেণি: {state["grade"]}

Launch পর্বে (৫ মিনিট) শিক্ষকের কাজ ক্লাসরুমের দৃশ্য বর্ণনার মতো অনুচ্ছেদে লেখো।
শিক্ষক কী বলেন তা এভাবে লেখো: শিক্ষক বলেন, "..."

নিচের ক্রমে লেখো:

Attention Grabber ও SEL:
শিক্ষক বলেন, "৫...৪...৩...২...১ — সবাই আমার দিকে তাকাও।" এর মতো attention grabber দিয়ে শুরু করো।
তারপর SEL check করো — যেমন: শিক্ষক বলেন, "থাম্বস আপ বা থাম্বস ডাউন করে জানাও আজ গণিত ক্লাস নিয়ে তোমাদের আগ্রহ কেমন?"

কৌতূহল জাগানো:
"{state["learning_outcome"]}" সম্পর্কিত বাস্তব জীবনের একটি গল্প বা প্রশ্ন দিয়ে কৌতূহল জাগাও।

পূর্বজ্ঞান যাচাই:
গতকালের বিষয়ের সাথে আজকের বিষয় সংযুক্ত করে একটি CFU প্রশ্ন করো।

শিক্ষার ফলাফল ঘোষণা:
শিক্ষক বলেন, "আজ আমরা শিখবো..." — ১ বাক্য।

কোনো অতিরিক্ত শিরোনাম বা মার্কডাউন যোগ করবে না। সব শুদ্ধ বাংলায়।"""
    return {**state, "launch_teacher": clean(call_llm(llm, prompt + VARIETY_RULE))}


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


# ── Nodes: Explore ─────────────────────────────────────────────────────────────

def node_explore_teacher(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}
বিষয়: {state["subject"]} | শ্রেণি: {state["grade"]}
{_ctx(state)}

Explore পর্বে (৭ মিনিট) শিক্ষকের কাজ ক্লাসরুমের দৃশ্য বর্ণনার মতো অনুচ্ছেদে লেখো।
গুরুত্বপূর্ণ: সমস্যা অবশ্যই "{state["learning_outcome"]}" এর সাথে সরাসরি সম্পর্কিত হতে হবে।
শিক্ষক বলেন, "..." ফরম্যাট ব্যবহার করো।

প্রতিটি উপ-বিভাগ নিচের ঠিক এই শিরোনামে (একা একটি লাইনে):

Teacher Action
- শিক্ষার্থীদের সামনে Learning Outcome সম্পর্কিত একটি বাস্তবধর্মী সমস্যা উপস্থাপন করো।
- Think-Pair-Share ব্যবহার করো: শিক্ষক বলেন, "প্রথমে ৩০ সেকেন্ড নিজে ভাবো, তারপর পাশের বন্ধুর সাথে আলোচনা করো।"
- CFU: শিক্ষক ঘুরে ঘুরে দেখার সময় ২-৩টি সুনির্দিষ্ট প্রশ্ন করবেন (যেমন: "তোমরা কেন মনে করো এখানে এই পদ্ধতি ব্যবহার করতে হবে?")

গুরুত্বপূর্ণ: শিরোনামগুলো হুবহু উপরের ইংরেজিতে লিখবে। কোনো ** বা # নয়। সব শুদ্ধ বাংলায়।"""
    return {**state, "explore_teacher": clean(call_llm(llm, prompt + VARIETY_RULE))}


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
    prompt = f"""Explore পর্বে শিক্ষক যা করবেন:
{state["explore_teacher"][:200]}

বিষয়: {state["subject"]} | শ্রেণি: {state["grade"]}

এই কার্যক্রমের জন্য শ্রেণিকক্ষে কী কী শিক্ষা উপকরণ লাগবে?
শুধুমাত্র বাস্তব উপকরণের নাম লেখো (যেমন: বোর্ড, মার্কার, ফ্লিপচার্ট ইত্যাদি)।
প্রতিটি আলাদা লাইনে। কোনো নম্বর বা বুলেট নয়। কোনো নির্দেশনা, প্রশ্ন বা ব্যাখ্যা লিখবে না।"""
    return {**state, "explore_materials": clean(call_llm(llm, prompt))}


# ── Nodes: Conceptualize ───────────────────────────────────────────────────────

def node_concept_teacher(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}
বিষয়: {state["subject"]} | শ্রেণি: {state["grade"]}
{_ctx(state)}
How ধাপগুলো: {state["key_points"]}

Conceptualize পর্বে (৮ মিনিট) শিক্ষকের কাজ লেখো।
গুরুত্বপূর্ণ: সবকিছু অবশ্যই "{state["learning_outcome"]}" এই Learning Outcome এর সাথে সম্পর্কিত হতে হবে। অন্য কোনো বিষয়ের উদাহরণ ব্যবহার করো না।

প্রতিটি উপ-বিভাগ নিচের ঠিক এই শিরোনামে (একা একটি লাইনে):

Suggested Time: 8 minutes (Abstracting the Steps + Generalization)
(এই লাইনটি হুবহু লিখবে)

Teacher Action
শিক্ষক বোর্ডে কী লিখবেন বা করবেন — ধাপে ধাপে। Learning Outcome অনুযায়ী সুনির্দিষ্ট উদাহরণ দাও।

Pictorial / Representation / Demonstration
শিক্ষক বোর্ডে Learning Outcome সম্পর্কিত উদাহরণ দেন। ধাপে ধাপে সম্পূর্ণ সমাধান দেখাও।

গুরুত্বপূর্ণ: শিরোনামগুলো হুবহু উপরের ইংরেজিতে লিখবে। কোনো ** বা # নয়।"""
    return {**state, "concept_teacher": clean(call_llm(llm, prompt + VARIETY_RULE))}


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
    prompt = f"""Conceptualize পর্বে শিক্ষক যা করবেন:
{state["concept_teacher"][:200]}

বিষয়: {state["subject"]} | শ্রেণি: {state["grade"]}

এই কার্যক্রমের জন্য শ্রেণিকক্ষে কী কী শিক্ষা উপকরণ লাগবে?
শুধুমাত্র বাস্তব উপকরণের নাম লেখো (যেমন: বোর্ড, মার্কার, চার্ট, ওয়ার্কশীট ইত্যাদি)।
প্রতিটি আলাদা লাইনে। কোনো নম্বর বা বুলেট নয়। কোনো নির্দেশনা বা ব্যাখ্যা লিখবে না।"""
    return {**state, "concept_materials": clean(call_llm(llm, prompt))}


# ── Nodes: Guided Practice ─────────────────────────────────────────────────────

def node_guided_teacher(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}
বিষয়: {state["subject"]} | শ্রেণি: {state["grade"]}
{_ctx(state)}
How ধাপগুলো: {state["key_points"]}

Guided Practice পর্বে (৫ মিনিট) শিক্ষকের কাজ ক্লাসরুমের দৃশ্য বর্ণনার মতো অনুচ্ছেদে লেখো।
গুরুত্বপূর্ণ: সমস্যাগুলো অবশ্যই "{state["learning_outcome"]}" এর সাথে সরাসরি সম্পর্কিত হতে হবে।
শিক্ষক বলেন, "..." ফরম্যাট ব্যবহার করো।

প্রতিটি উপ-বিভাগ নিচের ঠিক এই শিরোনামে (একা একটি লাইনে):

Teacher Action
শিক্ষক কীভাবে অনুশীলন পরিচালনা করবেন — Try it, Pair, Share পদ্ধতিতে।

Problem 1:
Learning Outcome সম্পর্কিত একটি অনুশীলন সমস্যা।

Problem 2:
Learning Outcome সম্পর্কিত আরেকটি সমস্যা (বাস্তব প্রেক্ষাপটে)।

Teacher Feedback
Precise Praise: নির্দিষ্ট কাজের প্রশংসা (যেমন: শিক্ষক বলেন, "আমি দেখতে পাচ্ছি পেছনের সারির সবাই খুব মন দিয়ে কাজ করছো — দারুণ একাগ্রতা!")
Positive Correction: ভুল সংশোধন (যেমন: শিক্ষক বলেন, "তোমার চেষ্টাটা চমৎকার হয়েছে, তবে সূত্রের এই অংশটা আরেকবার মিলিয়ে দেখো তো?")

গুরুত্বপূর্ণ: শিরোনামগুলো হুবহু উপরের ইংরেজিতে লিখবে। কোনো ** বা # নয়। সব শুদ্ধ বাংলায়।"""
    return {**state, "guided_teacher": clean(call_llm(llm, prompt + VARIETY_RULE))}


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
    prompt = f"""Guided Practice পর্বে শিক্ষক যা করবেন:
{state["guided_teacher"][:200]}

বিষয়: {state["subject"]} | শ্রেণি: {state["grade"]}

এই কার্যক্রমের জন্য শ্রেণিকক্ষে কী কী শিক্ষা উপকরণ লাগবে?
শুধুমাত্র বাস্তব উপকরণের নাম লেখো (যেমন: বোর্ড, মার্কার, অনুশীলনপত্র ইত্যাদি)।
প্রতিটি আলাদা লাইনে। কোনো নম্বর বা বুলেট নয়। কোনো নির্দেশনা বা ব্যাখ্যা লিখবে না।"""
    return {**state, "guided_materials": clean(call_llm(llm, prompt))}


# ── Nodes: Independent Practice ────────────────────────────────────────────────

def node_indep_teacher(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])
    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}
বিষয়: {state["subject"]} | শ্রেণি: {state["grade"]}
Guided Practice সমস্যা: {state["guided_teacher"][:300]}

Independent Practice পর্বে (৪ মিনিট) শিক্ষকের কাজ ক্লাসরুমের দৃশ্য বর্ণনার মতো অনুচ্ছেদে লেখো।
গুরুত্বপূর্ণ: সমস্যাগুলো অবশ্যই "{state["learning_outcome"]}" এর সাথে সম্পর্কিত হতে হবে এবং Guided Practice এর চেয়ে সামান্য কঠিন হতে হবে।
শিক্ষক বলেন, "..." ফরম্যাট ব্যবহার করো।

প্রতিটি উপ-বিভাগ নিচের ঠিক এই শিরোনামে (একা একটি লাইনে):

Teacher Action
- নীরব কাজের নির্দেশনা: শিক্ষক বলেন, "এখন পুরো ক্লাস ৩ মিনিট একদম নীরব থাকবে। কেউ কারো খাতা দেখবে না, নিজে চেষ্টা করো।"
- Positive Framing: শিক্ষক বলেন, "আমি দেখছি তোমরা অনেকক্ষণ ধরে সমস্যাটি সমাধানের চেষ্টা করছো, হার না মানার এই মানসিকতাই তোমাদের শিখতে সাহায্য করবে।"

Task 1:
Learning Outcome সম্পর্কিত একটি স্বতন্ত্র অনুশীলন সমস্যা।

Task 2:
Learning Outcome সম্পর্কিত আরেকটি সমস্যা (একটু বেশি কঠিন)।

গুরুত্বপূর্ণ: শিরোনামগুলো হুবহু উপরের ইংরেজিতে লিখবে। কোনো ** বা # নয়। সব শুদ্ধ বাংলায়।"""
    return {**state, "indep_teacher": clean(call_llm(llm, prompt + VARIETY_RULE))}


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
    prompt = f"""Independent Practice পর্বে শিক্ষক যা করবেন:
{state["indep_teacher"][:200]}

বিষয়: {state["subject"]} | শ্রেণি: {state["grade"]}

এই কার্যক্রমের জন্য শ্রেণিকক্ষে কী কী শিক্ষা উপকরণ লাগবে?
শুধুমাত্র বাস্তব উপকরণের নাম লেখো (যেমন: অনুশীলনপত্র, খাতা, কলম ইত্যাদি)।
প্রতিটি আলাদা লাইনে। কোনো নম্বর বা বুলেট নয়। কোনো নির্দেশনা বা ব্যাখ্যা লিখবে না।"""
    return {**state, "indep_materials": clean(call_llm(llm, prompt))}


# ── Nodes: Lesson Closing ──────────────────────────────────────────────────────

def node_closing_teacher(state: LessonPlanState) -> LessonPlanState:
    if state.get("error"): return state
    llm = get_llm(state["model_name"])

    # Extract first question from assessment for exit ticket
    assess_text = state.get("assess_questions", "")
    # Try to get just the first question
    first_q = assess_text
    for sep in ["প্রশ্ন ২", "প্রশ্ন ২)", "2)", "২)"]:
        if sep in assess_text:
            first_q = assess_text[:assess_text.index(sep)].strip()
            break

    prompt = f"""শিক্ষার্থীর শিক্ষার ফলাফল: {state["learning_outcome"]}
বিষয়: {state["subject"]} | শ্রেণি: {state["grade"]}

Lesson Closing পর্বে (৭ মিনিট) শিক্ষকের কাজ লেখো। ক্লাসরুমের দৃশ্য বর্ণনার মতো লেখো।
প্রতিটি উপ-বিভাগ নিচের ঠিক এই শিরোনামে (একা একটি লাইনে):

Teacher Action
শিক্ষক বলবেন, "আজ আমরা কী শিখলাম?" — মূল পয়েন্ট পর্যালোচনা করো।
নির্দিষ্ট প্রশংসা দাও (যেমন: "দেখো, রাফিক সব ধাপ সঠিকভাবে অনুসরণ করেছে")।

Exit Ticket
নিচের প্রশ্নটি হুবহু Exit Ticket হিসেবে লিখবে:
{first_q}

ESR (Expected Student Response):
উপরের Exit Ticket প্রশ্নের সঠিক উত্তর ধাপে ধাপে লেখো।

গুরুত্বপূর্ণ: শিরোনামগুলো হুবহু উপরের ইংরেজিতে লিখবে। কোনো ** বা # নয়।"""
    return {**state, "closing_teacher": clean(call_llm(llm, prompt + VARIETY_RULE))}


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
    prompt = f"""Lesson Closing পর্বে শিক্ষক যা করবেন:
{state["closing_teacher"][:200]}

বিষয়: {state["subject"]} | শ্রেণি: {state["grade"]}

এই কার্যক্রমের জন্য শ্রেণিকক্ষে কী কী শিক্ষা উপকরণ লাগবে?
শুধুমাত্র বাস্তব উপকরণের নাম লেখো (যেমন: Exit Ticket কাগজ, বোর্ড, হোমওয়ার্ক শীট ইত্যাদি)।
প্রতিটি আলাদা লাইনে। কোনো নম্বর বা বুলেট নয়। কোনো নির্দেশনা বা ব্যাখ্যা লিখবে না।"""
    return {**state, "closing_materials": clean(call_llm(llm, prompt))}


# ── Node: Assemble ─────────────────────────────────────────────────────────────

def node_assemble(state: LessonPlanState) -> LessonPlanState:
    lesson_plan = {
        "teacher_name":     state["teacher_name"],
        "subject":          state["subject"],
        "grade":            state["grade"],
        "duration":         state["duration"],
        "learning_outcome": state["learning_outcome"],

        # Pass retrieved context back so app.py can display it
        "retrieved_context": state.get("context", ""),

        "assess_time":      state.get("assess_time", ""),
        "assess_questions": state.get("assess_questions", ""),
        "assess_exemplar":  state.get("assess_exemplar", ""),
        "vision_why_ac":    state.get("vision_why_ac", ""),
        "vision_why_no":    state.get("vision_why_no", ""),
        "vision_what":      state.get("vision_what", ""),
        "key_points":       state.get("key_points", ""),
        "knowledge_text":   state.get("knowledge_text", ""),
        "blooms_skills":    state.get("blooms_skills", ""),

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
        ("node_retrieve_context",    node_retrieve_context),
        ("node_assess_time",         node_assess_time),
        ("node_assess_questions",    node_assess_questions),
        ("node_assess_exemplar",     node_assess_exemplar),
        ("node_vision_why_ac",       node_vision_why_ac),
        ("node_vision_why_no",       node_vision_why_no),
        ("node_vision_what",         node_vision_what),
        ("node_key_points",          node_key_points),
        ("node_knowledge_text",      node_knowledge_text),
        ("node_blooms_skills",       node_blooms_skills),
        ("node_launch_teacher",      node_launch_teacher),
        ("node_launch_student",      node_launch_student),
        ("node_launch_materials",    node_launch_materials),
        ("node_explore_teacher",     node_explore_teacher),
        ("node_explore_student",     node_explore_student),
        ("node_explore_materials",   node_explore_materials),
        ("node_concept_teacher",     node_concept_teacher),
        ("node_concept_student",     node_concept_student),
        ("node_concept_materials",   node_concept_materials),
        ("node_guided_teacher",      node_guided_teacher),
        ("node_guided_student",      node_guided_student),
        ("node_guided_materials",    node_guided_materials),
        ("node_indep_teacher",       node_indep_teacher),
        ("node_indep_student",       node_indep_student),
        ("node_indep_materials",     node_indep_materials),
        ("node_closing_teacher",     node_closing_teacher),
        ("node_closing_student",     node_closing_student),
        ("node_closing_materials",   node_closing_materials),
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
    manual_context: str = "",
    user_assess_questions: str = "",
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
        "manual_context": manual_context,
        "user_assess_questions": user_assess_questions,
        "context": "",
        "assess_time": "", "assess_questions": "", "assess_exemplar": "",
        "vision_why_ac": "", "vision_why_no": "", "vision_what": "",
        "key_points": "", "knowledge_text": "", "blooms_skills": "",
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