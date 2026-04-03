"""
Lesson Plan Generator - Streamlit App
Run with: streamlit run app.py
"""

import os
import logging
import tempfile
import streamlit as st

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="পাঠ পরিকল্পনা জেনারেটর",
    page_icon="📚",
    layout="centered",
)

st.markdown("""
<style>
  .main-title {
    text-align: center;
    color: var(--text-color);
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
  }
  .sub-title {
    text-align: center;
    color: var(--text-color);
    opacity: 0.55;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
  }
  .info-box {
    border-left: 3px solid #2E75B6;
    padding: 12px 16px;
    border-radius: 4px;
    margin-bottom: 1rem;
    background: color-mix(in srgb, #2E75B6 10%, transparent);
    color: var(--text-color);
  }
  .embed-box {
    border-left: 3px solid #d97706;
    padding: 12px 16px;
    border-radius: 4px;
    margin-bottom: 1rem;
    background: color-mix(in srgb, #d97706 10%, transparent);
    color: var(--text-color);
  }
  .context-box {
    border-left: 3px solid #16a34a;
    padding: 12px 16px;
    border-radius: 4px;
    margin-bottom: 1rem;
    background: color-mix(in srgb, #16a34a 10%, transparent);
    color: var(--text-color);
  }
  .stButton > button {
    background-color: #2E75B6;
    color: white;
    font-size: 0.95rem;
    font-weight: 600;
    width: 100%;
    padding: 0.55rem;
    border-radius: 6px;
    border: none;
  }
  .stButton > button:hover {
    background-color: #1a5a9a;
    color: white;
  }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
for key, default in [
    ("docx_bytes", None),
    ("docx_filename", None),
    ("lesson_plan", None),
    ("pdf_vectorstore_ready", False),
    ("pdf_processed_name", None),
    ("pdf_processing_error", None),
    # Context review step
    ("retrieved_chunks", None),       # list of retrieved context strings
    ("context_ready_for_gen", False),  # True after user reviews context
    ("final_context", ""),             # final context after user review/edit
    ("extra_contexts", []),            # user-added extra context strings
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Font setup ────────────────────────────────────────────────────────────────
@st.cache_resource
def setup_fonts():
    try:
        from font_manager import ensure_fonts
        return ensure_fonts()
    except Exception as e:
        logger.warning(f"Font setup: {e}")
        return []

setup_fonts()


# ── API Key ───────────────────────────────────────────────────────────────────
def get_api_key():
    return os.environ.get("OPENAI_API_KEY", "")


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">পাঠ পরিকল্পনা জেনারেটর</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Lesson Plan Generator · LangGraph + OpenAI + Bengali Template</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<b>কীভাবে ব্যবহার করবেন</b><br>
১) শিক্ষকের তথ্য ও শিক্ষার ফলাফল লিখুন<br>
২) Context পদ্ধতি বেছে নিন: PDF আপলোড, সংরক্ষিত Embedding, বা সরাসরি টেক্সট<br>
৩) মডেল বেছে নিন → <b>পাঠ পরিকল্পনা তৈরি করুন</b><br>
৪) DOCX ডাউনলোড করুন
</div>
""", unsafe_allow_html=True)

# ── API Key input ─────────────────────────────────────────────────────────────
current_key = get_api_key()
if not current_key:
    st.warning("OPENAI_API_KEY পাওয়া যাচ্ছে না।")
    api_key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input.strip()
        st.success("API Key সেট হয়েছে।")
        st.rerun()
else:
    masked = current_key[:7] + "..." + current_key[-4:] if len(current_key) > 12 else "***"
    col_key_info, col_key_action = st.columns([3, 1])
    with col_key_info:
        st.success(f"API Key সেট আছে: `{masked}`")
    with col_key_action:
        if st.button("🔄 পরিবর্তন", key="change_api_key"):
            os.environ.pop("OPENAI_API_KEY", None)
            st.rerun()

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Teacher Info
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("শিক্ষকের তথ্য")
col1, col2 = st.columns(2)
with col1:
    teacher_name = st.text_input("শিক্ষকের নাম *", placeholder="যেমন: Fahim Morshed")
    grade        = st.text_input("শ্রেণি (Grade) *", placeholder="যেমন: 7")
with col2:
    subject  = st.text_input("বিষয় (Subject) *", placeholder="যেমন: Math")
    duration = st.text_input("সময় (Time) *", placeholder="যেমন: 30 minutes")

st.subheader("শিক্ষার ফলাফল")
learning_outcome = st.text_area(
    "Learning Outcome (বাংলায়) *",
    placeholder="যেমন: শিক্ষার্থীরা বীজগাণিতীয় ভগ্নাংশ লঘুকরণ করতে পারবে",
    height=90,
)

st.subheader("মূল্যায়ন প্রশ্ন (ঐচ্ছিক)")
st.markdown("""
<div class="context-box">
<b>ঐচ্ছিক:</b> নিজের মূল্যায়ন প্রশ্ন দিতে চাইলে এখানে লিখুন। না দিলে AI নিজে তৈরি করবে।
</div>
""", unsafe_allow_html=True)
user_assess_questions = st.text_area(
    "Assessment Questions (ঐচ্ছিক)",
    placeholder="যেমন:\nপ্রশ্ন ১) 3(x+2) এর মান কত?\nক) 3x+2  খ) 3x+6  গ) 3x+8  ঘ) 6x+2\n\nপ্রশ্ন ২) সরল করো: 5(2a+3) - 2(a+4)",
    height=120,
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Context / Embedding Section
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("পাঠ্যপুস্তক ও Context সেটিংস")

st.markdown("""
<div class="embed-box">
<b>Context কী?</b> AI পাঠ পরিকল্পনা তৈরিতে পাঠ্যপুস্তকের প্রাসঙ্গিক অংশ ব্যবহার করে।
PDF আপলোড করতে না পারলে সরাসরি টেক্সট লিখেও context দিতে পারবেন।
</div>
""", unsafe_allow_html=True)

from embedding_manager import list_aliases, list_saved_books, delete_vectorstore
saved_aliases = list_aliases()

CONTEXT_MODE_PDF     = "PDF আপলোড করো (প্রতিবার index করো)"
CONTEXT_MODE_SAVED   = "সংরক্ষিত Embedding লোড করো"
CONTEXT_MODE_MANUAL  = "সরাসরি context লিখুন (PDF ছাড়া)"

context_mode = st.radio(
    "Context পদ্ধতি:",
    options=[CONTEXT_MODE_PDF, CONTEXT_MODE_SAVED, CONTEXT_MODE_MANUAL],
    index=0,
)

# Defaults
use_saved         = False
saved_alias_choice = ""
textbook_pdf      = None
save_embedding    = False
new_alias         = ""
manual_context    = ""

# ── Mode: Saved Embedding ─────────────────────────────────────────────────────
if context_mode == CONTEXT_MODE_SAVED:
    if not saved_aliases:
        st.warning("কোনো সংরক্ষিত Embedding নেই। প্রথমে PDF আপলোড করে সংরক্ষণ করুন।")
        context_mode = CONTEXT_MODE_PDF   # fall back gracefully
    else:
        use_saved = True
        saved_alias_choice = st.selectbox("সংরক্ষিত বই:", options=saved_aliases)
        books = {b["alias"]: b for b in list_saved_books()}
        if saved_alias_choice in books:
            meta = books[saved_alias_choice]
            st.caption(f"PDF: {meta['pdf_name']} · সংরক্ষিত: {meta['created_at'][:10]}")

        with st.expander("সংরক্ষিত Embedding মুছে ফেলুন"):
            del_alias = st.selectbox("মুছতে চান কোনটি?", options=saved_aliases, key="del_select")
            if st.button("মুছে ফেলুন", key="delete_btn"):
                delete_vectorstore(del_alias)
                st.success(f"'{del_alias}' মুছে ফেলা হয়েছে।")
                st.rerun()

# ── Mode: PDF Upload ──────────────────────────────────────────────────────────
if context_mode == CONTEXT_MODE_PDF:
    textbook_pdf = st.file_uploader("পাঠ্যপুস্তক PDF আপলোড করুন *", type=["pdf"])

    if textbook_pdf is not None:
        uploaded_name = textbook_pdf.name
        st.caption(f"📄 {uploaded_name} ({round(textbook_pdf.size/1024, 1)} KB)")

        # Reset state if a different file is uploaded
        if st.session_state.pdf_processed_name != uploaded_name:
            st.session_state.pdf_vectorstore_ready = False
            st.session_state.pdf_processing_error = None
            st.session_state.pdf_processed_name = None

        # Show current status
        if st.session_state.pdf_vectorstore_ready:
            st.success(f"✅ Embedding প্রস্তুত আছে ({uploaded_name})")

            # Option to save the embedding permanently
            with st.expander("💾 এই Embedding সংরক্ষণ করুন (ঐচ্ছিক)", expanded=False):
                save_alias = st.text_input(
                    "বইয়ের নাম / Alias *",
                    placeholder="যেমন: class8_math_2026",
                    help="এই নামে Embedding সংরক্ষিত হবে। পরে 'সংরক্ষিত Embedding' থেকে লোড করতে পারবেন।",
                    key="save_alias_input",
                )
                if st.button("💾 সংরক্ষণ করুন", key="save_embed_btn"):
                    if not save_alias.strip():
                        st.error("বইয়ের নাম দিন।")
                    else:
                        try:
                            from pipeline import get_embeddings
                            from embedding_manager import load_vectorstore, save_vectorstore as _save_vs
                            embeddings = get_embeddings()
                            vs = load_vectorstore("__current_upload__", embeddings)
                            _save_vs(vs, save_alias.strip(), uploaded_name)
                            st.success(f"✅ '{save_alias.strip()}' নামে সংরক্ষিত হয়েছে!")
                        except Exception as e:
                            st.error(f"সংরক্ষণ ব্যর্থ: {str(e)}")

        elif st.session_state.pdf_processing_error:
            st.error(f"❌ প্রসেসিং ব্যর্থ: {st.session_state.pdf_processing_error}")
            st.info("আবার চেষ্টা করুন অথবা 'সরাসরি context লিখুন' ব্যবহার করুন।")

        # Generate Embedding button — only show if not yet processed
        if not st.session_state.pdf_vectorstore_ready:
            st.markdown("""
<div class="embed-box">
📌 PDF আপলোড হয়েছে। এখন <b>Embedding তৈরি করুন</b> বোতামে ক্লিক করুন।
বড় PDF বা স্ক্যান করা PDF-এ সময় বেশি লাগতে পারে (OCR ব্যবহার হবে)।
</div>
""", unsafe_allow_html=True)

            if st.button("⚡ Embedding তৈরি করুন", key="gen_embed_btn", use_container_width=True):
                with st.status("PDF প্রসেস হচ্ছে ও Embedding তৈরি হচ্ছে...", expanded=True) as pdf_status:
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(textbook_pdf.read())
                            tmp_pdf_path_for_embed = tmp.name

                        def _progress(msg):
                            st.write(msg)

                        from pipeline import build_vector_store_from_pdf
                        from embedding_manager import save_vectorstore as _save_vs

                        vectorstore = build_vector_store_from_pdf(
                            tmp_pdf_path_for_embed,
                            progress_callback=_progress,
                        )

                        # Save to temp alias for pipeline to use
                        _save_vs(vectorstore, "__current_upload__", uploaded_name)

                        st.session_state.pdf_vectorstore_ready = True
                        st.session_state.pdf_processed_name = uploaded_name
                        st.session_state.pdf_processing_error = None
                        doc_count = vectorstore.index.ntotal
                        pdf_status.update(
                            label=f"✅ সম্পন্ন — {doc_count} টি text chunk পাওয়া গেছে",
                            state="complete", expanded=False,
                        )
                        st.rerun()

                    except Exception as e:
                        logger.exception("PDF processing error")
                        st.session_state.pdf_processing_error = str(e)
                        st.session_state.pdf_vectorstore_ready = False
                        pdf_status.update(label="❌ ব্যর্থ", state="error", expanded=True)
                        st.error(f"ত্রুটি: {str(e)}")
                    finally:
                        try:
                            os.unlink(tmp_pdf_path_for_embed)
                        except Exception:
                            pass
                        textbook_pdf.seek(0)

    else:
        # No file uploaded — reset
        if st.session_state.pdf_processed_name is not None:
            st.session_state.pdf_vectorstore_ready = False
            st.session_state.pdf_processed_name = None
            st.session_state.pdf_processing_error = None

# ── Mode: Manual Context ──────────────────────────────────────────────────────
if context_mode == CONTEXT_MODE_MANUAL:
    st.markdown("""
<div class="context-box">
<b>সরাসরি context:</b> পাঠ্যপুস্তক থেকে প্রাসঙ্গিক অংশ কপি করে এখানে পেস্ট করুন,
অথবা নিজের ভাষায় বিষয়বস্তু লিখুন। এই টেক্সট সরাসরি AI-কে দেওয়া হবে — কোনো PDF বা embedding লাগবে না।
</div>
""", unsafe_allow_html=True)
    manual_context = st.text_area(
        "Context টেক্সট *",
        placeholder="এখানে পাঠ্যপুস্তকের প্রাসঙ্গিক অংশ বা বিষয়বস্তু লিখুন...",
        height=200,
    )

st.divider()

# ── Model selection ───────────────────────────────────────────────────────────
st.subheader("AI মডেল")
model_name = st.selectbox(
    "OpenAI মডেল:",
    options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-5.4"],
    index=0,
    help="GPT-4o সবচেয়ে ভালো। gpt-4o-mini সস্তা ও দ্রুত।",
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Guidelines Management — Main Page
# ══════════════════════════════════════════════════════════════════════════════
from guidelines_manager import (
    list_guidelines, toggle_guideline, add_guideline,
    delete_guideline, reset_to_defaults, get_active_guidelines,
)

all_guidelines = list_guidelines()
active_count = sum(1 for g in all_guidelines if g.get("active", True))

st.subheader("📋 শিক্ষণ নির্দেশিকা")
st.markdown(f"""
<div class="info-box">
<b>AI এই নির্দেশিকাগুলো মেনে পাঠ পরিকল্পনা তৈরি করে।</b><br>
সক্রিয়: <b>{active_count}</b> / {len(all_guidelines)} টি নির্দেশিকা ·
চেকবক্সে টিক দিয়ে on/off করুন · 🗑️ বোতামে user-added নির্দেশিকা মুছুন
</div>
""", unsafe_allow_html=True)

# ── Tabs: View / Add / Reset ──────────────────────────────────────────────
tab_view, tab_add = st.tabs(["📖 বর্তমান নির্দেশিকা", "➕ নতুন যোগ করুন"])

with tab_view:
    # Group by category
    categories = {}
    for g in all_guidelines:
        cat = g.get("category", "General")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(g)

    for cat_name, cat_guidelines in categories.items():
        active_in_cat = sum(1 for g in cat_guidelines if g.get("active", True))
        with st.expander(
            f"📂 {cat_name}  —  {active_in_cat}/{len(cat_guidelines)} সক্রিয়",
            expanded=False,
        ):
            for g in cat_guidelines:
                gid = g["id"]
                is_active = g.get("active", True)
                is_builtin = g.get("builtin", True)
                source_label = g.get("source", "Unknown")

                col_toggle, col_info, col_del = st.columns([0.5, 3.5, 0.5])

                with col_toggle:
                    new_val = st.checkbox(
                        "on",
                        value=is_active,
                        key=f"toggle_{gid}",
                        label_visibility="collapsed",
                    )
                    if new_val != is_active:
                        toggle_guideline(gid, new_val)
                        st.rerun()

                with col_info:
                    badge = "🟢" if is_active else "🔴"
                    tag = "📌 Built-in" if is_builtin else "👤 User-added"
                    st.markdown(
                        f"{badge} **{g['name']}** &nbsp; · &nbsp; "
                        f"_{source_label}_ &nbsp; · &nbsp; {tag}"
                    )

                with col_del:
                    if not is_builtin:
                        if st.button("🗑️", key=f"del_{gid}", help="মুছে ফেলুন"):
                            delete_guideline(gid)
                            st.rerun()

                # Content preview
                content_preview = g.get("content", "")
                if content_preview:
                    st.text_area(
                        f"content_{gid}",
                        value=content_preview,
                        height=100,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"content_{gid}",
                    )
                st.markdown("---")

    # Reset button at the bottom of view tab
    st.caption("সব user-added নির্দেশিকা মুছে built-in default এ ফিরে যেতে চাইলে:")
    if st.button("🔄 ডিফল্ট নির্দেশিকায় রিসেট করুন", key="reset_guidelines"):
        reset_to_defaults()
        st.success("ডিফল্টে ফিরে গেছে। সব user-added নির্দেশিকা মুছে গেছে।")
        st.rerun()

with tab_add:
    st.markdown("""
<div class="context-box">
<b>নতুন নির্দেশিকা যোগ করুন:</b> আপনার নিজের শিক্ষণ কৌশল বা নিয়ম যোগ করুন।
এগুলো সংরক্ষিত থাকবে — প্রতিবার যোগ করার দরকার নেই।
</div>
""", unsafe_allow_html=True)

    with st.form("add_guideline_form", clear_on_submit=True):
        new_name = st.text_input(
            "নির্দেশিকার নাম *",
            placeholder="যেমন: Think-Pair-Share Technique",
        )
        new_category = st.selectbox(
            "Category (কোন পর্বের জন্য প্রযোজ্য)",
            options=[
                "General Pedagogy", "Lesson Vision", "Assessment",
                "Launch", "Explore", "Conceptualize",
                "Guided Practice", "Independent Practice", "Closing",
                "Classroom Management", "Literacy", "Other",
            ],
            index=0,
        )
        new_content = st.text_area(
            "নির্দেশিকার বিষয়বস্তু *",
            placeholder="এখানে বিস্তারিত লিখুন। AI এই নির্দেশনা মেনে পাঠ পরিকল্পনা তৈরি করবে...\n\nযেমন:\n- প্রথমে শিক্ষক প্রশ্ন করবেন\n- শিক্ষার্থীরা জোড়ায় আলোচনা করবে (৩০ সেকেন্ড)\n- একজন শিক্ষার্থী শ্রেণির সামনে উত্তর দেবে",
            height=180,
        )
        new_source = st.text_input(
            "Source / উৎস (ঐচ্ছিক)",
            value="User",
            placeholder="যেমন: TLAC Book, School Policy",
        )

        submitted = st.form_submit_button("✅ নির্দেশিকা যোগ করুন", use_container_width=True)
        if submitted:
            if not new_name.strip() or not new_content.strip():
                st.error("নাম ও বিষয়বস্তু উভয়ই দিতে হবে।")
            else:
                add_guideline(
                    name=new_name.strip(),
                    content=new_content.strip(),
                    category=new_category,
                    source=new_source.strip() or "User",
                )
                st.success(f"✅ '{new_name.strip()}' সফলভাবে যোগ হয়েছে!")
                st.rerun()

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Retrieve Context
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("ধাপ ১: Context বের করুন ও পর্যালোচনা করুন")

# ── Retrieval controls ────────────────────────────────────────────────────
if context_mode != CONTEXT_MODE_MANUAL:
    ctx_col1, ctx_col2 = st.columns(2)
    with ctx_col1:
        retrieval_k = st.number_input(
            "কতগুলো context chunk আনবে",
            min_value=1, value=6, step=1,
            help="যতগুলো চান লিখুন। Vector store-এ এর চেয়ে কম থাকলে যতগুলো আছে ততগুলোই আনবে।",
            key="retrieval_k_input",
        )
    with ctx_col2:
        use_entire_book = st.checkbox(
            "📚 সম্পূর্ণ বই context হিসেবে ব্যবহার করো",
            value=False,
            help="সব chunk একসাথে context হিসেবে পাঠাবে। বড় বইয়ের জন্য ধীর ও ব্যয়বহুল হতে পারে।",
            key="use_entire_book_cb",
        )
else:
    retrieval_k = 6
    use_entire_book = False

if st.button("🔍 Context বের করুন", key="retrieve_ctx_btn", use_container_width=True):

    errors = []
    if not learning_outcome.strip(): errors.append("শিক্ষার ফলাফল দিন।")
    if not os.environ.get("OPENAI_API_KEY", ""):
        errors.append("OpenAI API Key দিন।")
    if context_mode == CONTEXT_MODE_PDF and not st.session_state.pdf_vectorstore_ready:
        errors.append("প্রথমে PDF আপলোড করে Embedding তৈরি করুন।")
    if context_mode == CONTEXT_MODE_SAVED and not saved_alias_choice:
        errors.append("একটি সংরক্ষিত Embedding বেছে নিন।")
    if context_mode == CONTEXT_MODE_MANUAL and not manual_context.strip():
        errors.append("Context টেক্সট লিখুন।")

    for e in errors:
        st.error(e)

    if not errors:
        try:
            with st.spinner("Context বের করা হচ্ছে..."):
                if context_mode == CONTEXT_MODE_MANUAL:
                    # Manual — store as single chunk with score 0
                    st.session_state.retrieved_chunks = [
                        {"text": manual_context.strip(), "score": 0.0, "active": True}
                    ]
                else:
                    if context_mode == CONTEXT_MODE_PDF:
                        _alias = "__current_upload__"
                    else:
                        _alias = saved_alias_choice

                    from pipeline import get_embeddings, retrieve_context_with_scores, get_all_chunks_as_context
                    from embedding_manager import load_vectorstore

                    embeddings = get_embeddings()
                    vectorstore = load_vectorstore(_alias, embeddings)

                    if use_entire_book:
                        full_text = get_all_chunks_as_context(vectorstore)
                        if full_text:
                            parts = [p.strip() for p in full_text.split("\n\n---\n\n") if p.strip()]
                            st.session_state.retrieved_chunks = [
                                {"text": t, "score": 0.0, "active": True} for t in parts
                            ]
                        else:
                            st.session_state.retrieved_chunks = []
                    else:
                        scored = retrieve_context_with_scores(
                            vectorstore, learning_outcome.strip(), k=retrieval_k
                        )
                        st.session_state.retrieved_chunks = [
                            {"text": text, "score": score, "active": True}
                            for text, score in scored
                        ]

                st.session_state.extra_contexts = []
                st.rerun()

        except Exception as e:
            logger.exception("Context retrieval error")
            st.error(f"Context বের করতে সমস্যা: {str(e)}")

# ── Show retrieved context for review ─────────────────────────────────────
if st.session_state.retrieved_chunks is not None:
    chunks = st.session_state.retrieved_chunks
    active_chunks = [c for c in chunks if c.get("active", True)]

    if chunks:
        has_scores = any(c["score"] > 0 for c in chunks)
        st.markdown(f"""
<div class="context-box">
<b>{len(active_chunks)}/{len(chunks)} টি context সক্রিয়।</b>
অপ্রাসঙ্গিক অংশ ❌ বোতামে সরান, নতুন context যোগ করুন।
সন্তুষ্ট হলে নিচে <b>"পাঠ পরিকল্পনা তৈরি করুন"</b> ক্লিক করুন।
</div>
""", unsafe_allow_html=True)

        # Optional similarity threshold filter
        if has_scores:
            threshold = st.slider(
                "Similarity threshold (ঐচ্ছিক — বেশি score = কম প্রাসঙ্গিক)",
                min_value=0.0, max_value=3.0, value=3.0, step=0.1,
                help="এর নিচে score থাকলে context রাখা হবে। 3.0 = কোনো ফিল্টার নেই।",
                key="sim_threshold",
            )
            if threshold < 3.0:
                for c in chunks:
                    if c["score"] > threshold:
                        c["active"] = False
                    else:
                        # Only reactivate if it was deactivated by threshold, not manually
                        pass
    else:
        st.warning("কোনো context পাওয়া যায়নি। নিজে context যোগ করতে পারেন।")

    # Display each chunk with score badge and remove button
    for i, chunk in enumerate(chunks):
        is_active = chunk.get("active", True)
        score = chunk.get("score", 0)
        text = chunk.get("text", "")

        if is_active:
            col_label, col_remove = st.columns([5, 1])
            with col_label:
                score_badge = f" · Score: `{score:.3f}`" if score > 0 else ""
                st.markdown(f"**অংশ {i+1}**{score_badge}")
            with col_remove:
                if st.button("❌", key=f"remove_ctx_{i}", help="এই context সরান"):
                    st.session_state.retrieved_chunks[i]["active"] = False
                    st.rerun()

            st.text_area(
                f"ctx_{i}",
                value=text,
                height=110,
                key=f"ctx_edit_{i}",
                label_visibility="collapsed",
            )
        else:
            # Show as removed — option to restore
            col_label2, col_restore = st.columns([5, 1])
            with col_label2:
                st.markdown(f"~~অংশ {i+1}~~ (সরানো হয়েছে) · Score: `{score:.3f}`")
            with col_restore:
                if st.button("♻️", key=f"restore_ctx_{i}", help="পুনরুদ্ধার করুন"):
                    st.session_state.retrieved_chunks[i]["active"] = True
                    st.rerun()

    # ── Extra contexts section ────────────────────────────────────────────
    extra_list = st.session_state.extra_contexts
    if extra_list:
        st.markdown("---")
        st.markdown("**অতিরিক্ত context সমূহ**")
    for j, extra in enumerate(extra_list):
        col_ex_label, col_ex_del = st.columns([5, 1])
        with col_ex_label:
            st.markdown(f"**অতিরিক্ত {j+1}**")
        with col_ex_del:
            if st.button("❌", key=f"del_extra_{j}", help="সরান"):
                st.session_state.extra_contexts.pop(j)
                st.rerun()
        st.text_area(
            f"extra_{j}",
            value=extra,
            height=110,
            key=f"extra_edit_{j}",
            label_visibility="collapsed",
        )

    # ── Add new context ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**➕ নতুন context যোগ করুন**")
    new_extra_ctx = st.text_area(
        "নতুন context",
        placeholder="পাঠ্যপুস্তক থেকে কপি-পেস্ট করুন বা নিজে লিখুন...",
        height=120,
        key="new_extra_ctx_input",
        label_visibility="collapsed",
    )
    if st.button("➕ যোগ করুন", key="add_extra_ctx_btn"):
        if new_extra_ctx.strip():
            st.session_state.extra_contexts.append(new_extra_ctx.strip())
            st.rerun()
        else:
            st.warning("কিছু লিখুন।")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: Generate Lesson Plan
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("ধাপ ২: পাঠ পরিকল্পনা তৈরি করুন")

    # Validation
    gen_errors = []
    if not teacher_name.strip():     gen_errors.append("শিক্ষকের নাম দিন।")
    if not subject.strip():          gen_errors.append("বিষয় দিন।")
    if not grade.strip():            gen_errors.append("শ্রেণি দিন।")
    if not duration.strip():         gen_errors.append("সময় দিন।")

    if gen_errors:
        for e in gen_errors:
            st.warning(e)

    if st.button("📝 পাঠ পরিকল্পনা তৈরি করুন", use_container_width=True, key="gen_lp_btn"):

        final_errors = []
        if not teacher_name.strip():     final_errors.append("শিক্ষকের নাম দিন।")
        if not subject.strip():          final_errors.append("বিষয় দিন।")
        if not grade.strip():            final_errors.append("শ্রেণি দিন।")
        if not duration.strip():         final_errors.append("সময় দিন।")
        if not learning_outcome.strip(): final_errors.append("শিক্ষার ফলাফল দিন।")
        if not os.environ.get("OPENAI_API_KEY", ""):
            final_errors.append("OpenAI API Key দিন।")

        for e in final_errors:
            st.error(e)

        if not final_errors:
            # Build final context from active chunks + edited extras
            all_context_parts = []
            for i, chunk in enumerate(chunks):
                if chunk.get("active", True):
                    edited_val = st.session_state.get(f"ctx_edit_{i}", chunk["text"]).strip()
                    if edited_val:
                        all_context_parts.append(edited_val)
            for j, extra in enumerate(st.session_state.extra_contexts):
                edited_val = st.session_state.get(f"extra_edit_{j}", extra).strip()
                if edited_val:
                    all_context_parts.append(edited_val)

            final_context = "\n\n---\n\n".join(all_context_parts)

            try:
                with st.status("পাঠ পরিকল্পনা তৈরি হচ্ছে...", expanded=True) as status_box:
                    st.write("AI পাঠ পরিকল্পনার সকল অংশ তৈরি করছে (১–২ মিনিট)...")

                    from pipeline import run_pipeline
                    lesson_plan = run_pipeline(
                        teacher_name=teacher_name.strip(),
                        subject=subject.strip(),
                        grade=grade.strip(),
                        duration=duration.strip(),
                        learning_outcome=learning_outcome.strip(),
                        textbook_pdf_path="",
                        model_name=model_name,
                        use_saved_embedding=False,
                        saved_embedding_alias="",
                        save_new_embedding=False,
                        new_embedding_alias="",
                        manual_context=final_context,
                        user_assess_questions=user_assess_questions.strip(),
                    )

                    st.write("DOCX তৈরি হচ্ছে...")
                    out_dir = tempfile.mkdtemp()
                    safe = "".join(c for c in teacher_name if c.isalnum() or c in " _-")[:20].replace(" ", "_")
                    fname = f"lesson_plan_{safe}.docx"
                    docx_path = os.path.join(out_dir, fname)

                    from docx_generator import generate_docx
                    generate_docx(lesson_plan, docx_path)

                    with open(docx_path, "rb") as f:
                        st.session_state.docx_bytes = f.read()
                    st.session_state.docx_filename = fname
                    st.session_state.lesson_plan = lesson_plan

                    status_box.update(label="সম্পন্ন!", state="complete", expanded=False)

            except Exception as e:
                logger.exception("Generation error")
                st.error(f"ত্রুটি: {str(e)}")
                st.info("API Key ও ইন্টারনেট সংযোগ চেক করুন।")

# ══════════════════════════════════════════════════════════════════════════════
# Download Section (persistent)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.docx_bytes is not None:
    st.divider()
    st.success("পাঠ পরিকল্পনা প্রস্তুত।")

    st.download_button(
        label="DOCX ডাউনলোড করুন",
        data=st.session_state.docx_bytes,
        file_name=st.session_state.docx_filename,
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True,
        key="persistent_download",
    )

    if st.session_state.lesson_plan:
        lp = st.session_state.lesson_plan

        # ── Retrieved Context display ─────────────────────────────────────
        retrieved_ctx = lp.get("retrieved_context", "").strip()
        if retrieved_ctx:
            with st.expander("পাঠ্যপুস্তক থেকে যে context ব্যবহার হয়েছে", expanded=False):
                st.markdown("""
<div class="context-box">
নিচের অংশগুলো পাঠ্যপুস্তক থেকে vector similarity search-এর মাধ্যমে বেছে নেওয়া হয়েছে
এবং AI পাঠ পরিকল্পনা তৈরিতে ব্যবহার করেছে।
</div>
""", unsafe_allow_html=True)
                # Split by the separator used in retrieve_context()
                chunks = retrieved_ctx.split("\n\n---\n\n")
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"**অংশ {i}**")
                    st.text_area(
                        f"chunk_{i}",
                        value=chunk.strip(),
                        height=120,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"ctx_chunk_{i}",
                    )

        # ── Lesson plan preview ───────────────────────────────────────────
        with st.expander("পাঠ পরিকল্পনার প্রিভিউ", expanded=False):
            st.markdown(
                f"**শিক্ষক:** {lp.get('teacher_name')} &nbsp;|&nbsp; "
                f"**বিষয়:** {lp.get('subject')} &nbsp;|&nbsp; "
                f"**শ্রেণি:** {lp.get('grade')} &nbsp;|&nbsp; "
                f"**সময়:** {lp.get('duration')}"
            )
            st.divider()
            sections = [
                ("শিক্ষার ফলাফল",       "learning_outcome"),
                ("Vision — What",         "vision_what"),
                ("Vision — Why (Academic)", "vision_why_ac"),
                ("Vision — Why (Non-Academic)", "vision_why_no"),
                ("Key Points",            "key_points"),
                ("Assessment Questions",  "assess_questions"),
                ("Assessment Exemplar",   "assess_exemplar"),
                ("Launch — Teacher",      "launch_teacher"),
                ("Launch — Student",      "launch_student"),
                ("Explore — Teacher",     "explore_teacher"),
                ("Conceptualize — Teacher", "concept_teacher"),
                ("Guided Practice — Teacher", "guided_teacher"),
                ("Independent Practice — Teacher", "indep_teacher"),
                ("Lesson Closing — Teacher", "closing_teacher"),
            ]
            for label, key in sections:
                content = lp.get(key, "")
                if content:
                    st.markdown(f"**{label}**")
                    st.text_area(label, value=content, height=120,
                                 disabled=True, label_visibility="collapsed",
                                 key=f"prev_{key}")

    if st.button("পরিষ্কার করুন (নতুন পাঠ পরিকল্পনার জন্য)", key="clear_btn"):
        st.session_state.docx_bytes = None
        st.session_state.docx_filename = None
        st.session_state.lesson_plan = None
        st.session_state.retrieved_chunks = None
        st.session_state.context_ready_for_gen = False
        st.session_state.final_context = ""
        st.session_state.extra_contexts = []
        st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("GPT-4o: ~$0.05–0.15 প্রতি পাঠ পরিকল্পনা · gpt-4o-mini: ~10x সস্তা")