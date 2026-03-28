"""
Lesson Plan Generator - Streamlit App
Run with: streamlit run app.py

Changes:
  - DOCX output only (no PDF)
  - Download button persists after generation (stored in session_state)
  - Embedding management: save, load, delete saved book embeddings
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
  /* Use Streamlit's own theme tokens so text stays readable in dark mode */
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
    /* background adapts: light blue in light mode, muted in dark */
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
  /* Primary action button */
  .stButton > button[kind="primary"],
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
if "docx_bytes" not in st.session_state:
    st.session_state.docx_bytes = None
if "docx_filename" not in st.session_state:
    st.session_state.docx_filename = None
if "lesson_plan" not in st.session_state:
    st.session_state.lesson_plan = None


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
২) পাঠ্যপুস্তক PDF আপলোড করুন <i>অথবা</i> সংরক্ষিত Embedding বেছে নিন<br>
৩) মডেল বেছে নিন → <b>পাঠ পরিকল্পনা তৈরি করুন</b><br>
৪) DOCX ডাউনলোড করুন
</div>
""", unsafe_allow_html=True)

# ── API Key input ─────────────────────────────────────────────────────────────
if not get_api_key():
    st.warning("OPENAI_API_KEY পাওয়া যাচ্ছে না।")
    api_key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input.strip()
        st.success("API Key সেট হয়েছে।")
else:
    st.success("OpenAI API Key লোড হয়েছে।")

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

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Embedding Management Section
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("পাঠ্যপুস্তক ও Embedding সেটিংস")

st.markdown("""
<div class="embed-box">
<b>Embedding সম্পর্কে:</b> পাঠ্যপুস্তক একবার প্রসেস করে সংরক্ষণ করলে পরের বার
ঐ বই আবার আপলোড না করেও ব্যবহার করা যাবে — সময় ও API খরচ দুটোই বাঁচবে।
</div>
""", unsafe_allow_html=True)

from embedding_manager import list_aliases, list_saved_books, delete_vectorstore

saved_aliases = list_aliases()

embedding_mode = st.radio(
    "Embedding পদ্ধতি:",
    options=[
        "PDF আপলোড করে নতুনভাবে তৈরি করো",
        "সংরক্ষিত Embedding লোড করো",
    ],
    index=0,
    horizontal=False,
)

use_saved = "সংরক্ষিত" in embedding_mode
saved_alias_choice = ""
textbook_pdf = None
save_embedding = False
new_alias = ""

if use_saved:
    if not saved_aliases:
        st.warning("কোনো সংরক্ষিত Embedding নেই। প্রথমে PDF আপলোড করে সংরক্ষণ করুন।")
        use_saved = False
    else:
        saved_alias_choice = st.selectbox(
            "সংরক্ষিত বই:",
            options=saved_aliases,
        )
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

else:
    textbook_pdf = st.file_uploader(
        "পাঠ্যপুস্তক PDF আপলোড করুন *",
        type=["pdf"],
    )
    if textbook_pdf:
        st.caption(f"{textbook_pdf.name} ({round(textbook_pdf.size/1024, 1)} KB)")

    save_embedding = st.checkbox("এই বইয়ের Embedding সংরক্ষণ করো (পরে পুনরায় ব্যবহারের জন্য)")
    if save_embedding:
        new_alias = st.text_input(
            "বইয়ের নাম / Alias *",
            placeholder="যেমন: class7_math_2024",
            help="এই নামে Embedding সংরক্ষিত হবে।",
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
# Generate Button
# ══════════════════════════════════════════════════════════════════════════════
if st.button("পাঠ পরিকল্পনা তৈরি করুন", use_container_width=True):

    errors = []
    if not teacher_name.strip():     errors.append("শিক্ষকের নাম দিন।")
    if not subject.strip():          errors.append("বিষয় দিন।")
    if not grade.strip():            errors.append("শ্রেণি দিন।")
    if not duration.strip():         errors.append("সময় দিন।")
    if not learning_outcome.strip(): errors.append("শিক্ষার ফলাফল দিন।")
    if not os.environ.get("OPENAI_API_KEY", ""):
        errors.append("OpenAI API Key দিন।")
    if use_saved and not saved_alias_choice:
        errors.append("একটি সংরক্ষিত Embedding বেছে নিন।")
    if not use_saved and textbook_pdf is None:
        errors.append("পাঠ্যপুস্তকের PDF আপলোড করুন।")
    if not use_saved and save_embedding and not new_alias.strip():
        errors.append("Embedding সংরক্ষণের জন্য বইয়ের নাম (Alias) দিন।")

    for e in errors:
        st.error(e)

    if not errors:
        tmp_pdf_path = ""
        if not use_saved and textbook_pdf is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(textbook_pdf.read())
                tmp_pdf_path = tmp.name

        try:
            with st.status("পাঠ পরিকল্পনা তৈরি হচ্ছে...", expanded=True) as status_box:

                if use_saved:
                    st.write(f"'{saved_alias_choice}' থেকে Embedding লোড হচ্ছে...")
                else:
                    st.write("পাঠ্যপুস্তক প্রসেস হচ্ছে...")
                    if save_embedding and new_alias.strip():
                        st.write(f"Embedding '{new_alias.strip()}' নামে সংরক্ষিত হবে...")

                st.write("AI পাঠ পরিকল্পনার সকল অংশ তৈরি করছে (১–২ মিনিট)...")

                from pipeline import run_pipeline
                lesson_plan = run_pipeline(
                    teacher_name=teacher_name.strip(),
                    subject=subject.strip(),
                    grade=grade.strip(),
                    duration=duration.strip(),
                    learning_outcome=learning_outcome.strip(),
                    textbook_pdf_path=tmp_pdf_path,
                    model_name=model_name,
                    use_saved_embedding=use_saved,
                    saved_embedding_alias=saved_alias_choice,
                    save_new_embedding=save_embedding,
                    new_embedding_alias=new_alias.strip(),
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

            if save_embedding and new_alias.strip() and not use_saved:
                st.success(f"Embedding '{new_alias.strip()}' সফলভাবে সংরক্ষিত হয়েছে।")

        except Exception as e:
            logger.exception("Generation error")
            st.error(f"ত্রুটি: {str(e)}")
            st.info("API Key ও ইন্টারনেট সংযোগ চেক করুন।")
        finally:
            if tmp_pdf_path and os.path.exists(tmp_pdf_path):
                try:
                    os.unlink(tmp_pdf_path)
                except Exception:
                    pass

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
        with st.expander("পাঠ পরিকল্পনার প্রিভিউ", expanded=False):
            lp = st.session_state.lesson_plan
            st.markdown(
                f"**শিক্ষক:** {lp.get('teacher_name')} &nbsp;|&nbsp; "
                f"**বিষয়:** {lp.get('subject')} &nbsp;|&nbsp; "
                f"**শ্রেণি:** {lp.get('grade')} &nbsp;|&nbsp; "
                f"**সময়:** {lp.get('duration')}"
            )
            st.divider()
            sections = [
                ("শিক্ষার ফলাফল",       "learning_outcome"),
                ("Lesson Vision",         "lesson_vision"),
                ("Key Points",            "key_points"),
                ("মূল্যায়ন",             "assessment"),
                ("Launch",                "launch"),
                ("Explore",               "explore"),
                ("Conceptualize",         "conceptualize"),
                ("Guided Practice",       "guided_practice"),
                ("Independent Practice",  "independent_practice"),
                ("Lesson Closing",        "lesson_closing"),
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
        st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("GPT-4o: ~$0.05–0.15 প্রতি পাঠ পরিকল্পনা · gpt-4o-mini: ~10x সস্তা")