"""
Guidelines Manager: stores, loads, and manages teaching guidelines.

Each guideline has:
  - id: unique string key
  - name: display name (e.g. "Wait Time", "100% Technique")
  - source: where it came from (e.g. "TLAC", "TFB Winter Academy", "User")
  - category: which lesson section(s) it applies to
  - content: the actual guideline text
  - active: bool — whether this guideline is currently enforced
  - builtin: bool — True for pre-loaded guidelines, False for user-added

Storage:
  guidelines_store/
    guidelines.json   ← all guidelines
"""

import os
import json
import copy
from datetime import datetime

STORE_DIR = os.path.join(os.path.dirname(__file__), "guidelines_store")
GUIDELINES_FILE = os.path.join(STORE_DIR, "guidelines.json")


def _ensure_dir():
    os.makedirs(STORE_DIR, exist_ok=True)


def _load_all() -> dict:
    _ensure_dir()
    if not os.path.exists(GUIDELINES_FILE):
        # First run — seed with built-in guidelines
        guidelines = _get_builtin_guidelines()
        _save_all(guidelines)
        return guidelines

    with open(GUIDELINES_FILE, "r", encoding="utf-8") as f:
        existing = json.load(f)

    # Merge: add any new built-in guidelines that don't exist yet
    # This ensures new built-ins appear after code updates without
    # losing user-added guidelines
    builtins = _get_builtin_guidelines()
    added = 0
    for gid, g in builtins.items():
        if gid not in existing:
            existing[gid] = g
            added += 1
    if added > 0:
        _save_all(existing)

    return existing


def _save_all(guidelines: dict):
    _ensure_dir()
    with open(GUIDELINES_FILE, "w", encoding="utf-8") as f:
        json.dump(guidelines, f, ensure_ascii=False, indent=2)


# ── Public API ────────────────────────────────────────────────────────────────

def list_guidelines() -> list[dict]:
    """Return list of all guidelines sorted by category then name."""
    data = _load_all()
    items = list(data.values())
    return sorted(items, key=lambda x: (x.get("category", ""), x.get("name", "")))


def get_active_guidelines() -> list[dict]:
    """Return only active guidelines."""
    return [g for g in list_guidelines() if g.get("active", True)]


def get_guideline(gid: str) -> dict | None:
    data = _load_all()
    return data.get(gid)


def toggle_guideline(gid: str, active: bool):
    """Enable or disable a guideline."""
    data = _load_all()
    if gid in data:
        data[gid]["active"] = active
        _save_all(data)


def add_guideline(name: str, content: str, category: str = "General",
                  source: str = "User") -> str:
    """Add a new user guideline. Returns its id."""
    data = _load_all()
    gid = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(data)}"
    data[gid] = {
        "id": gid,
        "name": name.strip(),
        "source": source.strip(),
        "category": category.strip(),
        "content": content.strip(),
        "active": True,
        "builtin": False,
        "created_at": datetime.now().isoformat(),
    }
    _save_all(data)
    return gid


def delete_guideline(gid: str) -> bool:
    """Delete a user-added guideline. Cannot delete built-in ones."""
    data = _load_all()
    if gid in data and not data[gid].get("builtin", True):
        del data[gid]
        _save_all(data)
        return True
    return False


def reset_to_defaults():
    """Reset all guidelines to built-in defaults (removes user guidelines)."""
    guidelines = _get_builtin_guidelines()
    _save_all(guidelines)


def build_system_prompt_section() -> str:
    """Build the guidelines text block to inject into the LLM system prompt."""
    active = get_active_guidelines()
    if not active:
        return ""

    lines = []
    lines.append("তোমাকে নিচের শিক্ষণ নির্দেশিকাগুলো কঠোরভাবে মেনে পাঠ পরিকল্পনা তৈরি করতে হবে:\n")

    current_cat = None
    for g in active:
        cat = g.get("category", "General")
        if cat != current_cat:
            lines.append(f"\n=== {cat} ===")
            current_cat = cat
        lines.append(f"\n[{g['name']}] ({g['source']})")
        lines.append(g["content"])

    return "\n".join(lines)


# ── Built-in Guidelines ───────────────────────────────────────────────────────

def _get_builtin_guidelines() -> dict:
    """Return the default built-in guidelines extracted from the provided documents."""
    guidelines = {}

    def _add(gid, name, source, category, content):
        guidelines[gid] = {
            "id": gid,
            "name": name,
            "source": source,
            "category": category,
            "content": content,
            "active": True,
            "builtin": True,
            "created_at": "2025-01-01T00:00:00",
        }

    # ── Lesson Vision ─────────────────────────────────────────────────────
    _add("lv_what_how_why", "Lesson Vision: What-How-Why Structure",
         "TFB Winter Academy", "Lesson Vision",
         "Lesson Vision একটি blueprint যা শিক্ষককে বলে ঠিক কী শিখবে শিক্ষার্থীরা, কেন শিখবে, এবং কীভাবে শিখবে।\n"
         "WHAT: পাঠ্যপুস্তক থেকে প্রয়োজনীয় সংজ্ঞা, ধারণা, সূত্র, শব্দভাণ্ডার চিহ্নিত করো। Learning outcome-এর সাথে aligned হতে হবে।\n"
         "HOW: শিক্ষার্থীরা কীভাবে দক্ষতা অর্জন করবে তার ধাপ (শিক্ষকের পদ্ধতি নয়)। সর্বোচ্চ ৬টি ধাপ। Assessment সমাধান করে ধাপগুলো বের করো।\n"
         "   - নির্দেশনা (instructions) HOW keypoint এর অংশ নয়।\n"
         "   - ধারণা (concept) নয়, বরং ধারণা ব্যবহারের ধাপ লিখতে হবে।\n"
         "   - শিক্ষার্থীদের পক্ষে সহজে অনুসরণযোগ্য হতে হবে।\n"
         "WHY (Academic): এই বিষয় শেখার একাডেমিক গুরুত্ব — পরবর্তী পাঠ, পরীক্ষা, উচ্চতর শ্রেণিতে কীভাবে কাজে আসবে।\n"
         "WHY (Non-Academic): দৈনন্দিন জীবনে এই জ্ঞান কীভাবে প্রয়োগ হয় — যথেষ্ট আকর্ষণীয় যাতে শিক্ষার্থীরা engaged হয়।\n"
         "All components must align to the learning outcome. Key points must be accurately derived from the objective.")

    # ── Assessment ────────────────────────────────────────────────────────
    _add("assess_sar", "Assessment: SAR Criteria",
         "TFB Winter Academy", "Assessment",
         "Assessment অবশ্যই SAR criteria মেনে হতে হবে:\n"
         "Scaffolded: প্রশ্নগুলো সহজ থেকে কঠিনের দিকে যাবে (low rigor → high rigor)। এটি শিক্ষার্থীদের বোঝার ঘাটতি দেখার সুযোগ দেয়।\n"
         "Aligned: প্রশ্নগুলো সরাসরি learning outcome এর content/knowledge ও specific verb/skill পরীক্ষা করবে। শিক্ষার্থীরা যা শিখেছে তা প্রদর্শনের সুযোগ পাবে।\n"
         "Rigorous: প্রশ্নের কাঠিন্যের মাত্রা এমন হবে যাতে শিক্ষার্থীরা হতাশ না হয় (একটি প্রশ্নও চেষ্টা করতে না পারা), আবার এত সহজও নয় যে কোনো চ্যালেঞ্জই থাকে না।\n"
         "Time: 5 minutes for assessment.")

    # ── Launch ────────────────────────────────────────────────────────────
    _add("launch_guide", "Launch: Structure & Guidelines",
         "TFB Winter Academy", "Launch",
         "Launch পর্ব (৫ মিনিট) — Focus: SEL + Checking Prior Knowledge + Sparking Curiosity\n"
         "Teacher's Role:\n"
         "  1. SEL activity: একটি socio-emotional learning কার্যক্রম দিয়ে ক্লাস শুরু (মুড চেক, fist-to-five)।\n"
         "  2. Grounding with rules: প্রসঙ্গ অনুযায়ী classroom rules মনে করিয়ে দাও বা শিক্ষার্থীদের বলাও।\n"
         "  3. Prior Knowledge check: পূর্ববর্তী জ্ঞান যাচাই করো — কম চাপের প্রশ্ন বা কার্যক্রম।\n"
         "  4. Sparking Curiosity: পাঠের 'কেন' (Why) প্রতিষ্ঠা করো — একটি গল্প, প্রশ্ন, বা পরিস্থিতি।\n"
         "  5. Learning outcome ঘোষণা করো।\n"
         "Student's Role: SEL কার্যক্রমে অংশ নেওয়া, পূর্বজ্ঞান স্মরণ করা।\n"
         "Guiding Questions: কোন উত্তরগুলো পূর্বজ্ঞান নির্দেশ করবে? শিক্ষার্থীরা কেন পাঠটি আকর্ষণীয় মনে করবে?")

    # ── Explore ───────────────────────────────────────────────────────────
    _add("explore_guide", "Explore: Structure & Guidelines",
         "TFB Winter Academy", "Explore",
         "Explore পর্ব (৭-১৫ মিনিট) — Low rigor welcoming exploration\n"
         "Teacher's Role:\n"
         "  - শিক্ষার্থীদের সামনে একটি সমস্যা বা উদ্দীপক উপস্থাপন করো (পর্যবেক্ষণ, পরীক্ষণ, প্যাটার্ন চিহ্নিতকরণ)।\n"
         "  - Probing questions ব্যবহার করো যা চিন্তার দিকনির্দেশনা দেয় (সরাসরি উত্তরের দিকে নয়, বরং প্রক্রিয়ার দিকে focus)।\n"
         "  - নির্দিষ্ট সময়সীমা বেঁধে দাও।\n"
         "Student's Role: পর্যবেক্ষণের ভিত্তিতে অনুমান বা সিদ্ধান্ত তৈরি করা।\n"
         "Key Rule: Explore starts with LOW rigor to engage ALL students. The probe does NOT direct towards the answer.")

    # ── Conceptualize ─────────────────────────────────────────────────────
    _add("concept_guide", "Conceptualize: Structure & Guidelines",
         "TFB Winter Academy", "Conceptualize",
         "Conceptualize পর্ব (৮-১০ মিনিট) — Abstracting steps + Generalization\n"
         "Teacher's Role:\n"
         "  - Explore এ শিক্ষার্থীরা কী আবিষ্কার করলো তা জিজ্ঞেস করে শুরু করো।\n"
         "  - HOW keypoints শিক্ষার্থীদের সামনে দৃশ্যমান করো (বোর্ডে লেখো)।\n"
         "  - Teacher ও student মিলে HOW keypoint ব্যবহার করে সমস্যা সমাধান করো (modelling)।\n"
         "  - Worked example সম্পূর্ণ দেখাও — ধাপে ধাপে।\n"
         "Student's Role: মূল ধারণা ও ধাপ বোঝা, একই ধরনের সমস্যা সমাধানের পদ্ধতি শেখা।\n"
         "Key Rule: The HOW keypoint is made VISIBLE for the students during this step.")

    # ── Guided Practice ───────────────────────────────────────────────────
    _add("guided_guide", "Guided Practice: Structure & Guidelines",
         "TFB Winter Academy", "Guided Practice",
         "Guided Practice পর্ব (৫-১০ মিনিট)\n"
         "  - শিক্ষার্থীরা দলে/জোড়ায় কাজ করে। Peer learning ঘটে।\n"
         "  - Rigor Explore এর চেয়ে বেশি।\n"
         "  - HOW keypoint এখনো দৃশ্যমান থাকে।\n"
         "  - শিক্ষক সক্রিয়ভাবে পর্যবেক্ষণ করেন এবং প্রয়োজনে দলকে feedback দেন।\n"
         "  - শিক্ষক SOCS পদ্ধতিতে (Specific, Observable, Concrete, Sequential) নির্দেশনা দেবেন।")

    # ── Independent Practice ──────────────────────────────────────────────
    _add("indep_guide", "Independent Practice: Structure & Guidelines",
         "TFB Winter Academy", "Independent Practice",
         "Independent Practice পর্ব (৫-৮ মিনিট)\n"
         "  - শিক্ষার্থীরা একা কাজ করে।\n"
         "  - Rigor Guided Practice এর চেয়ে বেশি।\n"
         "  - HOW keypoint আর দৃশ্যমান থাকে না।\n"
         "  - শিক্ষক সকল শিক্ষার্থীর কাজ পর্যবেক্ষণ করেন এবং যাদের সাহায্য দরকার তাদের সাহায্য করেন।")

    # ── Closing ───────────────────────────────────────────────────────────
    _add("closing_guide", "Closing: Structure & Guidelines",
         "TFB Winter Academy", "Closing",
         "Closing পর্ব (৫-৭ মিনিট)\n"
         "  - শিক্ষার্থীরা শেখা বিষয় সংহত করে (consolidate learning)।\n"
         "  - Exit Ticket হিসেবে assessment এ অংশ নেয়।\n"
         "  - শিক্ষক বাড়ির কাজ দেন।\n"
         "  - মূল পয়েন্ট পুনরালোচনা ও প্রশংসা।")

    # ── Bloom's Taxonomy ──────────────────────────────────────────────────
    _add("blooms", "Bloom's Taxonomy & Rigor Progression",
         "TFB Winter Academy", "General Pedagogy",
         "পাঠের বিভিন্ন পর্বে Bloom's Taxonomy অনুযায়ী rigor ক্রমান্বয়ে বাড়বে:\n"
         "Remember (মনে করা) → Understand (বোঝা) → Apply (প্রয়োগ করা) → Analyze (বিশ্লেষণ করা) → Evaluate (মূল্যায়ন করা) → Create (সৃষ্টি করা)\n"
         "Launch: Remember/Understand level (low barrier)\n"
         "Explore: Understand/Apply level\n"
         "Conceptualize: Apply/Analyze level\n"
         "Guided Practice: Apply/Analyze level\n"
         "Independent Practice: Analyze/Evaluate level\n"
         "Assessment: spans multiple levels (scaffolded)")

    # ── Gradual Release ───────────────────────────────────────────────────
    _add("gradual_release", "Gradual Release of Responsibility",
         "TFB Winter Academy", "General Pedagogy",
         "শিক্ষার দায়িত্ব ধীরে ধীরে শিক্ষক থেকে শিক্ষার্থীর কাছে স্থানান্তরিত হয়:\n"
         "Conceptualize: 'I do it' — শিক্ষক মডেলিং করেন।\n"
         "Guided Practice: 'We do it' — শিক্ষক ও শিক্ষার্থী একসাথে।\n"
         "Independent Practice: 'You do it alone' — শিক্ষার্থী একা করে।")

    # ── What To Do (SOCS) ─────────────────────────────────────────────────
    _add("what_to_do", "What To Do (SOCS Instructions)",
         "TLAC / TFB", "Classroom Management",
         "What To Do: শিক্ষক নির্দেশনা দেবেন SOCS পদ্ধতিতে:\n"
         "Specific: সুনির্দিষ্ট — সময়, গতিবিধি, বা কণ্ঠস্বর উল্লেখ করো।\n"
         "Observable: পর্যবেক্ষণযোগ্য — শিক্ষক compliance দেখতে পারবেন।\n"
         "Concrete: বাস্তব — কী adjust করতে হবে স্পষ্ট বলো।\n"
         "Sequential: ধারাবাহিক — ছোট ছোট ধাপে ভাঙো।\n"
         "কী করতে হবে বলো, কী করা যাবে না তা নয়।")

    # ── Wait Time ─────────────────────────────────────────────────────────
    _add("wait_time", "Wait Time Technique",
         "TLAC / TFB", "Classroom Management",
         "Wait Time: প্রশ্ন ও উত্তরের মধ্যে ইচ্ছাকৃত বিরতি রেখে অংশগ্রহণ বাড়াও।\n"
         "  - Narrate hands: হাত গণনা করো, উৎসাহিত করো।\n"
         "  - Promote thinking skills: ভাবার সময়ে কী করবে বলে দাও।\n"
         "  - Make wait time transparent: কতটুকু সময় দিচ্ছো বলে দাও।\n"
         "  - Stop talking: চুপ থাকো, ঘুরে দেখো।")

    # ── 100% Technique ────────────────────────────────────────────────────
    _add("hundred_percent", "100% Technique",
         "TLAC / TFB", "Classroom Management",
         "100%: ১০০% শিক্ষার্থী, ১০০% সময়, ১০০% ভাবে মনোযোগী থাকবে।\n"
         "  - Compliance you can see: Radar build করো — strategic কোণে দাঁড়াও, Be Seen Looking করো।\n"
         "  - Least invasive form: Non-verbal cue → Positive group correction → Allow anonymity → Private individual correction।")

    # ── Strong Voice ──────────────────────────────────────────────────────
    _add("strong_voice", "Strong Voice Technique",
         "TLAC / TFB", "Classroom Management",
         "Strong Voice: গুরুত্বপূর্ণ নির্দেশনার সময়:\n"
         "  - Square Up, Stand Still: সোজা দাঁড়াও, নড়াচড়া করো না, পূর্ণ মনোযোগ না পাওয়া পর্যন্ত কথা বলো না।\n"
         "  - Formal Register: একাডেমিক ভাষা ও আনুষ্ঠানিক ভঙ্গি ব্যবহার করো।\n"
         "  - Self-interruption: শিক্ষার্থীরা কথা বললে মাঝপথে থামো, ঘর স্ক্যান করো, তারপর বন্ধুত্বপূর্ণভাবে পুনরায় শুরু করো।\n"
         "  - Do not engage: আচরণ সংক্রান্ত আলোচনায় অন্য বিষয়ে জড়িও না।")

    # ── Positive Framing ──────────────────────────────────────────────────
    _add("positive_framing", "Positive Framing Technique",
         "TLAC / TFB", "Classroom Management",
         "Positive Framing: ইতিবাচক দিক তুলে ধরে শিক্ষার্থীদের উন্নত কাজের দিকে guide করো।\n"
         "  - Narrate the positive: ইতিবাচক আচরণ বর্ণনা করো, নেতিবাচক নয়।\n"
         "  - Live in the now: সমাধান বলো, সমস্যায় আটকে থেকো না (What To Do instruction দাও)।\n"
         "  - Challenge, talk aspirations: সময়সীমা দাও, প্রতিযোগিতা তৈরি করো, শিক্ষার্থীদের স্বপ্নের সাথে যুক্ত করো।")

    # ── Precise Praise ────────────────────────────────────────────────────
    _add("precise_praise", "Precise Praise Technique",
         "TLAC / TFB", "Classroom Management",
         "Precise Praise: প্রশংসা সর্বোচ্চ কার্যকর করো।\n"
         "  - Reinforce the process, not the kid: কাজটির প্রশংসা করো, ব্যক্তিগত বৈশিষ্ট্যের নয়।\n"
         "  - Objective-aligned praise: পাঠের বিষয়বস্তুর সাথে প্রশংসা align করো।\n"
         "  - Acknowledge vs Praise: প্রত্যাশা পূরণে acknowledge করো (great, fantastic ছাড়া), প্রত্যাশা ছাড়িয়ে গেলে praise করো।\n"
         "  - Economy of language: সংক্ষিপ্ত ও নির্দিষ্ট প্রশংসা দাও।")

    # ── CFU ────────────────────────────────────────────────────────────────
    _add("cfu_technique", "Check for Understanding (CFU)",
         "TLAC / TFB", "Classroom Management",
         "Check for Understanding (CFU): পাঠের প্রতিটি গুরুত্বপূর্ণ মুহূর্তে বোঝাপড়া যাচাই করো।\n"
         "  - প্রশ্ন representative শিক্ষার্থীদের কাছে নির্দেশিত করো।\n"
         "  - Scaffolded প্রশ্ন ব্যবহার করো (কতটুকু বুঝেছে তা নির্ণয়ের জন্য)।\n"
         "  - সবচেয়ে গুরুত্বপূর্ণ ধারণাগুলো সম্পর্কে পুরো পাঠ জুড়ে প্রশ্ন করো।\n"
         "  - উচ্চ প্রত্যাশা বজায় রাখো এবং কেন সঠিক বা ভুল তা ব্যাখ্যা করো।")

    # ── Right is Right / No Opt Out ───────────────────────────────────────
    _add("right_no_opt", "Right is Right / No Opt Out",
         "TLAC / TFB", "Classroom Management",
         "Right is Right: আংশিক সঠিক উত্তরকে সম্পূর্ণ সঠিক হিসেবে গ্রহণ করো না। পুরোপুরি সঠিক না হওয়া পর্যন্ত প্রশ্ন চালিয়ে যাও।\n"
         "No Opt Out: কোনো শিক্ষার্থী 'জানি না' বলে পার পাবে না। অন্য শিক্ষার্থীর সাহায্যে বা clue দিয়ে উত্তর বের করাও, তারপর আবার প্রথম শিক্ষার্থীকে জিজ্ঞেস করো।")

    # ── Reading Strategies ────────────────────────────────────────────────
    _add("reading_strategies", "7 Reading Strategies (Before-During-After)",
         "TFB Winter Academy", "Literacy",
         "পড়ার তিনটি পর্যায়ে ৭টি কৌশল:\n"
         "Before Reading: Predict (পূর্বানুমান করো), Visualize (কল্পনা করো)\n"
         "During Reading: Question (প্রশ্ন করো), Connect (সংযোগ করো), Identify (চিহ্নিত করো)\n"
         "After Reading: Infer (অনুমান করো), Evaluate (মূল্যায়ন করো)\n"
         "Literacy পাঠের explore ও conceptualize পর্বে এই কৌশলগুলো ব্যবহার করো।")

    # ── TAL Rubric: Plan Purposefully ─────────────────────────────────────
    _add("tal_plan", "TAL Rubric: Plan Purposefully (P-3)",
         "Teaching As Leadership / TFA", "General Pedagogy",
         "P-3 (Exemplary level) মেনে পাঠ পরিকল্পনা তৈরি করো:\n"
         "  - Key points learning outcome থেকে সঠিকভাবে ও যথাযথভাবে derived হবে।\n"
         "  - পাঠের সব components learning outcome, key points, ও mastery demonstration পদ্ধতির সাথে aligned হবে।\n"
         "  - Activities innovative, student-centered হবে এবং effective lesson planning এর principles মেনে চলবে।\n"
         "  - কার্যক্রম: prior knowledge activate করা, key ideas articulate করা, misunderstandings anticipate করা, scaffolded student practice, understanding assess করা।\n"
         "  - Timing feasible হবে এবং mastery support করবে, পাশাপাশি real-time adjustment এর সুযোগ রাখবে।")

    # ── Learning Outcome Format ───────────────────────────────────────────
    _add("lo_format", "Effective Learning Outcome Format",
         "TFB Winter Academy", "General Pedagogy",
         "কার্যকর Learning Outcome এর ফরম্যাট: SWBAT + Specific Verb + Specific Knowledge Item\n"
         "উদাহরণ: SWBAT identify key details of the text\n"
         "SWBAT examine parts of a plant\n"
         "SWBAT differentiate between geometric shapes - quadrilateral and triangle\n"
         "পাঠ পরিকল্পনার সব কার্যক্রম এই learning outcome achieve করার দিকে নির্দেশিত হবে।")

    # ══════════════════════════════════════════════════════════════════════
    # Master Prompt Guidelines
    # ══════════════════════════════════════════════════════════════════════

    # ── Classroom Scene Description Style ─────────────────────────────────
    _add("mp_scene_style", "Classroom Scene Description Style",
         "Master Prompt", "General Pedagogy",
         "পাঠ পরিকল্পনা একটি ক্লাসরুমের দৃশ্য বর্ণনার মতো লিখতে হবে — শিক্ষক ধাপে ধাপে কী করেন ও কী বলেন তা দেখাতে হবে।\n"
         "বেশিরভাগ অংশ অনুচ্ছেদ আকারে (paragraph) লিখতে হবে, বুলেট-ভারী নয়।\n"
         "তবে কোনো নির্দেশনা বা 'What To Do' অংশ পরিষ্কার নম্বরযুক্ত পয়েন্টে লিখতে হবে।\n"
         "শিক্ষক যখন কথা বলেন: শিক্ষক বলেন, \"শিক্ষকের কথা...\"\n"
         "শিক্ষক যখন বোর্ডে লেখেন: শিক্ষক লিখেন, \"শিক্ষকের লেখা...\"\n"
         "সম্পূর্ণ পাঠ পরিকল্পনা শুদ্ধ বাংলায় (Raw Bengali) লিখতে হবে। কোনো Banglish ব্যবহার করা যাবে না।")

    # ── 4-Point WTD Instructions ──────────────────────────────────────────
    _add("mp_4point_wtd", "4-Point WTD Instructions",
         "Master Prompt", "Classroom Management",
         "কোনো কাজের নির্দেশনা দেওয়ার সময় ঠিক ৪টি সুনির্দিষ্ট, কার্যকর ধাপ দিতে হবে (What To Do - 4 Point Instructions)।\n"
         "প্রতিটি ধাপ Specific, Observable, Concrete, Sequential হবে।\n"
         "যেমন: ১) খাতা বের করো। ২) পৃষ্ঠা ১৫ খোলো। ৩) প্রশ্ন ৩ পড়ো। ৪) ২ মিনিটে উত্তর লেখো।")

    # ── Real-Life Connection First ────────────────────────────────────────
    _add("mp_reallife_launch", "Real-Life Connection First",
         "Master Prompt", "Launch",
         "Launch এ বিষয়টিকে বাস্তব জীবনের পরিস্থিতির সাথে সংযুক্ত করে শুরু করতে হবে।\n"
         "একটি সহজ ভূমিকা দিয়ে শুরু করো যা দৈনন্দিন জীবনের সাথে সম্পর্কিত।\n"
         "শিক্ষার্থীদের চিন্তার প্রশ্ন (thinking questions) দিয়ে engage করো।")

    # ── Worked Examples with Explanation ──────────────────────────────────
    _add("mp_worked_examples", "Worked Examples with Explanation",
         "Master Prompt", "Conceptualize",
         "Conceptualize পর্বে সম্পূর্ণ সমাধান সহ worked examples (সমাধানকৃত সমস্যা) থাকতে হবে।\n"
         "সহজ থেকে কঠিনের দিকে ক্রমান্বয়ে এগিয়ে যাবে।\n"
         "প্রতিটি ধাপের ব্যাখ্যা থাকবে — শুধু উত্তর নয়।\n"
         "ভাষা সহজ ও স্পষ্ট হবে, ৭ম-৮ম শ্রেণির শিক্ষার্থীদের উপযোগী।")

    # ── Practice Scaffolding ──────────────────────────────────────────────
    _add("mp_practice_scaffold", "Practice Scaffolding (Easy to Hard)",
         "Master Prompt", "Guided Practice",
         "Guided Practice ও Independent Practice এ সমস্যাগুলো সহজ থেকে চ্যালেঞ্জিং ক্রমে সাজাতে হবে।\n"
         "Guided Practice: 'Try it, Pair, Share' — প্রথমে একা চেষ্টা, তারপর জোড়ায় আলোচনা, তারপর শ্রেণিতে শেয়ার।\n"
         "Independent Practice: একা কাজ — শিক্ষক পর্যবেক্ষণ করেন ও struggling শিক্ষার্থীদের সাহায্য করেন।")

    # ── Assessment ESR & Exit Ticket Rule ─────────────────────────────────
    _add("mp_esr_exit", "Assessment ESR & Exit Ticket Rule",
         "Master Prompt", "Assessment",
         "Assessment এর প্রথম প্রশ্নটি হুবহু একই শব্দে Exit Ticket এ পুনরায় ব্যবহার করতে হবে।\n"
         "সঠিক উত্তরের লেবেল: ESR (Expected Student Response)।\n"
         "Assessment প্রশ্নে সঠিক উত্তর lesson plan এ অন্তর্ভুক্ত থাকতে হবে।")

    # ── Positive Correction ───────────────────────────────────────────────
    _add("mp_positive_correction", "Positive Correction (ইতিবাচক সংশোধন)",
         "Master Prompt", "Classroom Management",
         "ভুল সংশোধন করো কিন্তু লজ্জা দিও না।\n"
         "যেমন: শিক্ষক বলেন, \"তোমার চেষ্টাটা চমৎকার হয়েছে, তবে সূত্রের এই অংশটা আরেকবার মিলিয়ে দেখো তো?\"\n"
         "Guided Practice ও Independent Practice পর্বে এই পদ্ধতি ব্যবহার করো।\n"
         "Positive Framing: শিক্ষক বলেন, \"আমি দেখছি তোমরা অনেকক্ষণ ধরে সমস্যাটি সমাধানের চেষ্টা করছো, হার না মানার এই মানসিকতাই তোমাদের শিখতে সাহায্য করবে।\"")

    return guidelines