"""
Lesson Plan Generator - Streamlit App
Run with: streamlit run app.py
"""

import os
import logging
import tempfile
import streamlit as st

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="পাঠ পরিকল্পনা জেনারেটর",
    page_icon="📚",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2E75B6;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #555;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .info-box {
        background: #f0f7ff;
        border-left: 4px solid #2E75B6;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 1rem;
        font-size: 0.95rem;
    }
    .stButton > button {
        background-color: #2E75B6;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        padding: 0.6rem;
        border-radius: 6px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #1a5a9a;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ── Font download on first run ────────────────────────────────────────────────
@st.cache_resource
def setup_fonts():
    try:
        from font_manager import ensure_fonts
        return ensure_fonts()
    except Exception as e:
        logger.warning(f"Font setup warning: {e}")
        return []

setup_fonts()


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_api_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📚 পাঠ পরিকল্পনা জেনারেটর</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Lesson Plan Generator · LangGraph + OpenAI + Bengali Template</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<b>কীভাবে ব্যবহার করবেন:</b><br>
১) শিক্ষকের তথ্য ও শিক্ষার ফলাফল বাংলায় লিখুন<br>
২) পাঠ্যপুস্তকের PDF আপলোড করুন<br>
৩) OpenAI মডেল বেছে নিন<br>
৪) <b>"পাঠ পরিকল্পনা তৈরি করুন"</b> বাটনে ক্লিক করুন<br>
৫) PDF ও DOCX ডাউনলোড করুন
</div>
""", unsafe_allow_html=True)

# ── API Key ───────────────────────────────────────────────────────────────────
api_key_env = get_api_key()
if not api_key_env:
    st.warning("⚠️ OPENAI_API_KEY পাওয়া যাচ্ছে না। নিচে সরাসরি দিন:")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="আপনার OpenAI API key। শুধু এই session এ ব্যবহার হবে।",
    )
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input.strip()
        st.success("✅ API Key সেট হয়েছে।")
else:
    st.success("✅ OpenAI API Key লোড হয়েছে (environment থেকে)।")

st.divider()

# ── Teacher Info ──────────────────────────────────────────────────────────────
st.subheader("👤 শিক্ষকের তথ্য")

col1, col2 = st.columns(2)
with col1:
    teacher_name = st.text_input("শিক্ষকের নাম *", placeholder="যেমন: Fahim Morshed")
    grade        = st.text_input("শ্রেণি (Grade) *", placeholder="যেমন: 7")
with col2:
    subject  = st.text_input("বিষয় (Subject) *", placeholder="যেমন: Math")
    duration = st.text_input("সময় (Time) *", placeholder="যেমন: 30 minutes")

# ── Learning Outcome ──────────────────────────────────────────────────────────
st.subheader("🎯 শিক্ষার ফলাফল")
learning_outcome = st.text_area(
    "Learning Outcome (বাংলায় লিখুন) *",
    placeholder="যেমন: শিক্ষার্থীরা বীজগাণিতীয় ভগ্নাংশ লঘুকরণ করতে পারবে (BL1)",
    height=100,
)

# ── Model & PDF ───────────────────────────────────────────────────────────────
st.subheader("⚙️ AI সেটিংস ও পাঠ্যপুস্তক")

col3, col4 = st.columns(2)
with col3:
    model_name = st.selectbox(
        "OpenAI মডেল",
        options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo","gpt-5.4"],
        index=0,
        help="GPT-4o সবচেয়ে ভালো। gpt-4o-mini সস্তা ও দ্রুত।",
    )
with col4:
    textbook_pdf = st.file_uploader(
        "📖 পাঠ্যপুস্তক PDF আপলোড করুন *",
        type=["pdf"],
    )
    if textbook_pdf:
        st.caption(f"✅ আপলোড: {textbook_pdf.name} ({round(textbook_pdf.size/1024, 1)} KB)")

st.divider()

# ── Generate ──────────────────────────────────────────────────────────────────
if st.button("🚀 পাঠ পরিকল্পনা তৈরি করুন", use_container_width=True):

    # Validate
    errors = []
    if not teacher_name.strip():   errors.append("শিক্ষকের নাম দিন।")
    if not subject.strip():        errors.append("বিষয় দিন।")
    if not grade.strip():          errors.append("শ্রেণি দিন।")
    if not duration.strip():       errors.append("সময় দিন।")
    if not learning_outcome.strip(): errors.append("শিক্ষার ফলাফল দিন।")
    if textbook_pdf is None:       errors.append("পাঠ্যপুস্তকের PDF আপলোড করুন।")
    if not os.environ.get("OPENAI_API_KEY", ""):
        errors.append("OpenAI API Key দিন।")

    if errors:
        for e in errors:
            st.error(f"❌ {e}")
        st.stop()

    # Save PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(textbook_pdf.read())
        tmp_pdf_path = tmp.name

    try:
        # Step indicators
        with st.status("পাঠ পরিকল্পনা তৈরি হচ্ছে...", expanded=True) as status_box:

            st.write("📚 পাঠ্যপুস্তক লোড ও ভেক্টর স্টোর তৈরি হচ্ছে...")
            from pipeline import run_pipeline

            st.write("🤖 AI দিয়ে সকল সেকশন তৈরি হচ্ছে (১-২ মিনিট লাগতে পারে)...")
            lesson_plan = run_pipeline(
                teacher_name=teacher_name.strip(),
                subject=subject.strip(),
                grade=grade.strip(),
                duration=duration.strip(),
                learning_outcome=learning_outcome.strip(),
                textbook_pdf_path=tmp_pdf_path,
                model_name=model_name,
            )

            st.write("📄 PDF তৈরি হচ্ছে...")
            out_dir   = tempfile.mkdtemp()
            safe_name = "".join(
                c for c in teacher_name if c.isalnum() or c in " _-"
            )[:20].replace(" ", "_")
            base_name = f"lesson_plan_{safe_name}"
            pdf_path  = os.path.join(out_dir, f"{base_name}.pdf")
            docx_path = os.path.join(out_dir, f"{base_name}.docx")

            from pdf_generator import generate_pdf
            generate_pdf(lesson_plan, pdf_path)

            st.write("📝 Word document তৈরি হচ্ছে...")
            from docx_generator import generate_docx
            generate_docx(lesson_plan, docx_path)

            status_box.update(label="✅ সম্পন্ন!", state="complete", expanded=False)

        # ── Downloads ─────────────────────────────────────────────────────────
        st.success("🎉 পাঠ পরিকল্পনা সফলভাবে তৈরি হয়েছে!")

        dl1, dl2 = st.columns(2)
        with dl1:
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📄 PDF ডাউনলোড করুন",
                    data=f.read(),
                    file_name=f"{base_name}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        with dl2:
            with open(docx_path, "rb") as f:
                st.download_button(
                    label="📝 DOCX ডাউনলোড করুন",
                    data=f.read(),
                    file_name=f"{base_name}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )

        # ── Preview ───────────────────────────────────────────────────────────
        with st.expander("👁️ পাঠ পরিকল্পনার প্রিভিউ", expanded=False):
            st.markdown(
                f"**শিক্ষক:** {lesson_plan.get('teacher_name')} &nbsp;|&nbsp; "
                f"**বিষয়:** {lesson_plan.get('subject')} &nbsp;|&nbsp; "
                f"**শ্রেণি:** {lesson_plan.get('grade')} &nbsp;|&nbsp; "
                f"**সময়:** {lesson_plan.get('duration')}"
            )
            st.divider()
            sections = [
                ("🎯 শিক্ষার ফলাফল",        "learning_outcome"),
                ("📖 Lesson Vision",          "lesson_vision"),
                ("🔑 Key Points (How)",       "key_points"),
                ("📝 মূল্যায়ন",              "assessment"),
                ("🚀 Launch",                 "launch"),
                ("🔍 Explore",                "explore"),
                ("💡 Conceptualize",          "conceptualize"),
                ("✏️ Guided Practice",        "guided_practice"),
                ("🧪 Independent Practice",   "independent_practice"),
                ("🏁 Lesson Closing",         "lesson_closing"),
            ]
            for label, key in sections:
                content = lesson_plan.get(key, "")
                if content:
                    st.markdown(f"**{label}**")
                    st.text_area(
                        label=label,
                        value=content,
                        height=150,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"preview_{key}",
                    )

    except Exception as e:
        logger.exception("Generation error")
        st.error(f"❌ ত্রুটি হয়েছে: {str(e)}")
        st.info("💡 API key সঠিক কিনা এবং ইন্টারনেট সংযোগ আছে কিনা চেক করুন।")
    finally:
        try:
            os.unlink(tmp_pdf_path)
        except Exception:
            pass

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "💡 GPT-4o তে প্রতি পাঠ পরিকল্পনায় আনুমানিক $0.05–0.15 খরচ। "
    "gpt-4o-mini তে ~10x কম।"
)