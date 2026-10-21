import streamlit as st
import google.generativeai as genai
import html

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LinkedIn Post Generator – Roqeeb",
    page_icon="✦",
    layout="centered",
)

# ── Custom CSS (brand colours: yellow #FED34D, charcoal #444444) ──────────────
st.markdown("""
<style>
    /* Background and base text */
    .stApp { background-color: #0f0f0f; color: #f0ece4; }

    /* Hide default Streamlit hamburger and footer */
    #MainMenu, footer { visibility: hidden; }

    /* Header */
    .header-box {
        background: #181818;
        border: 1px solid #2e2e2e;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .logo {
        background: #FED34D;
        border-radius: 8px;
        width: 42px; height: 42px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.2rem; font-weight: 800; color: #444444;
        flex-shrink: 0;
    }
    .header-text h2 { margin: 0; font-size: 1.05rem; color: #f0ece4; }
    .header-text p  { margin: 2px 0 0; font-size: 0.78rem; color: #888; }

    /* Post preview card */
    .post-card {
        background: #181818;
        border: 1px solid #2e2e2e;
        border-radius: 12px;
        padding: 20px 22px;
        margin-top: 16px;
        white-space: pre-wrap;
        line-height: 1.8;
        font-size: 0.92rem;
        color: #f0ece4;
        max-height: 420px;
        overflow-y: auto;
        scrollbar-width: thin;
        scrollbar-color: #FED34D #222;
    }

    /* Section label */
    .section-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #888;
        font-weight: 600;
        margin-bottom: 8px;
    }

    /* Streamlit button override */
    .stButton > button {
        background: #FED34D !important;
        color: #444444 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 9px !important;
        padding: 12px 20px !important;
        width: 100% !important;
        font-size: 0.95rem !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    /* Input fields */
    .stTextArea textarea, .stSelectbox select {
        background: #181818 !important;
        border: 1px solid #2e2e2e !important;
        color: #f0ece4 !important;
        border-radius: 8px !important;
    }

    /* Character count */
    .char-info {
        font-size: 0.72rem;
        color: #888;
        text-align: right;
        margin-top: 4px;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        background: #1a2a1a;
        border: 1px solid #2e4a2e;
        color: #4ade80;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── API key check ─────────────────────────────────────────────────────────────
def get_api_key() -> str | None:
    """Return API key from Streamlit secrets or st.session_state (manual entry)."""
    # 1. Streamlit secrets (Streamlit Cloud deployment)
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    # 2. Manually entered in sidebar
    return st.session_state.get("api_key_input", "").strip() or None


# ── Profile context (your LinkedIn info baked in) ─────────────────────────────
PROFILE_CONTEXT = """
You are a LinkedIn content writer for Busari (Gbolahan) Roqeeb — a Litigation Clerk 
based in Ibadan, Nigeria, actively transitioning into AI Engineering through self-teaching.

PROFILE CONTEXT:
- Background: Litigation Clerk with strong skills in legal research, document analysis, 
  and attention to detail
- Current journey: Self-taught AI/ML learner studying Python, Modern AI, Prompt Engineering, 
  Neural Networks, and practical machine learning applications
- Unique angle: His legal background gives him a distinct perspective on AI — especially 
  document synthesis, structured data handling, and evaluation
- Mission: Documenting his learning journey publicly — sharing insights, lessons, 
  code snippets, and real-world applications
- Target audience: Fellow learners, AI enthusiasts, and professionals in the AI/Web3 space
- University: Adekunle Ajasin University
- He is building in public, learning daily, and bridging law + AI
"""


# ── Helper: call Google Gemini API with streaming ─────────────────────────────
def generate_post_stream(topic: str, tone: str, goal: str, length: str, extra: str, api_key: str):
    """
    Calls the Google Gemini API using streaming and yields text chunks.

    Parameters
    ----------
    topic   : What the post is about
    tone    : Writing tone (e.g. Professional, Storytelling)
    goal    : Post objective (e.g. Build authority)
    length  : Desired post length
    extra   : Any extra instructions from the user
    api_key : Google API key

    Yields
    ------
    str : Text chunks as they stream in
    """
    prompt = f"""{PROFILE_CONTEXT}

Write a LinkedIn post with these parameters:
- Topic  : {topic}
- Tone   : {tone}
- Goal   : {goal}
- Length : {length}
{f'- Extra : {extra}' if extra else ''}

Rules:
1. Write in first person as Busari Roqeeb
2. Start with a strong hook — a bold statement, surprising insight, or relatable moment
3. Use short paragraphs with line breaks for LinkedIn readability
4. Feel authentic to someone genuinely learning and building in public
5. Lean into his law-to-AI story when relevant
6. Include 3–5 relevant hashtags at the end
7. Max 2 emojis — only if they add meaning
8. Plain text only — no markdown like **bold**
9. Natural CTA if needed — not "drop a comment below!"

Return ONLY the post text. No intro, no explanation.
"""

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    response = model.generate_content(prompt, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text


# ── Sidebar: API key entry ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("**API Key**")

    # Only show the manual input if not already in secrets
    has_secret = False
    try:
        _ = st.secrets["GOOGLE_API_KEY"]
        has_secret = True
    except (KeyError, FileNotFoundError):
        pass

    if has_secret:
        st.success("API key loaded from secrets ✓")
    else:
        st.text_input(
            "Google API Key",
            type="password",
            placeholder="AIza...",
            key="api_key_input",
            help="Get your free key at aistudio.google.com/apikey",
        )
        st.caption("Your key is only stored in session memory.")

    st.divider()
    st.markdown("**About**")
    st.caption(
        "Generates personalised LinkedIn posts for Busari Roqeeb using Claude. "
        "Powered by the Anthropic API."
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <div class="logo">BR</div>
    <div class="header-text">
        <h2>LinkedIn Post Generator</h2>
        <p>Busari (Gbolahan) Roqeeb · Aspiring AI Engineer</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Input section ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">What\'s the post about?</div>', unsafe_allow_html=True)
topic = st.text_area(
    label="topic",
    label_visibility="collapsed",
    placeholder=(
        "e.g. What I learned after 30 days of self-teaching Python, "
        "how my legal background helps me think about AI, "
        "just finished a prompt engineering project…"
    ),
    height=100,
)

# Two columns for tone and goal
col1, col2 = st.columns(2)

with col1:
    tone = st.selectbox("Tone", [
        "Professional",
        "Conversational",
        "Storytelling",
        "Bold & Direct",
        "Educational",
        "Inspirational",
    ])

with col2:
    goal = st.selectbox("Post Goal", [
        "Build authority in AI",
        "Document my learning journey",
        "Get engagement from AI community",
        "Attract opportunities / recruiters",
        "Share a milestone or win",
        "Start a conversation about AI or career transition",
    ])

length = st.selectbox("Length", [
    "Short (under 150 words)",
    "Medium (150–250 words)",
    "Long (250–400 words)",
])

extra = st.text_input(
    "Extra instructions (optional)",
    placeholder="e.g. include a CTA, mention I'm open to mentorship, add hashtags about Python…"
)

# Generate + Regenerate buttons
btn_col1, btn_col2 = st.columns([3, 1])
with btn_col1:
    generate_clicked = st.button("✦ Generate Post")
with btn_col2:
    regenerate_clicked = st.button(
        "↺ Regenerate",
        disabled="last_post" not in st.session_state,
        help="Generate a new version of the same post",
    )


# ── Output section ────────────────────────────────────────────────────────────
should_generate = generate_clicked or regenerate_clicked

if should_generate:
    if not topic.strip():
        st.warning("Please describe what the post is about before generating.")
    else:
        api_key = get_api_key()
        if not api_key:
            st.error("⚠️ No API key found. Add your Anthropic API key in the sidebar.")
        else:
            st.markdown('<div class="section-label" style="margin-top:20px">Generated Post</div>', unsafe_allow_html=True)

            # Stream the post into a placeholder
            post_placeholder = st.empty()
            accumulated = ""

            try:
                for chunk in generate_post_stream(topic, tone, goal, length, extra, api_key):
                    accumulated += chunk
                    # Safely escape content before injecting into HTML
                    safe_text = html.escape(accumulated)
                    post_placeholder.markdown(
                        f'<div class="post-card">{safe_text}</div>',
                        unsafe_allow_html=True,
                    )

                # Save to session state so it persists across re-runs
                st.session_state["last_post"] = accumulated
                st.session_state["last_topic"] = topic
                st.session_state["last_tone"] = tone
                st.session_state["last_goal"] = goal
                st.session_state["last_length"] = length
                st.session_state["last_extra"] = extra

            except Exception as e:
                err = str(e)
                if "API_KEY_INVALID" in err or "invalid" in err.lower():
                    st.error("❌ Invalid API key. Check your Google API key.")
                elif "quota" in err.lower() or "429" in err:
                    st.error("⚠️ Rate limit hit. Please wait a moment and try again.")
                else:
                    st.error(f"Something went wrong: {e}")

# Show persisted post (non-streaming render after generation)
elif "last_post" in st.session_state:
    post = st.session_state["last_post"]

    st.markdown('<div class="section-label" style="margin-top:20px">Generated Post</div>', unsafe_allow_html=True)

    safe_text = html.escape(post)
    st.markdown(f'<div class="post-card">{safe_text}</div>', unsafe_allow_html=True)

# Char count + copy box (shown whenever a post exists in session)
if "last_post" in st.session_state and not should_generate:
    post = st.session_state["last_post"]

    char_count = len(post)
    color = "#f87171" if char_count > 3000 else "#f59e0b" if char_count > 2000 else "#888"
    st.markdown(
        f'<div class="char-info" style="color:{color}">{char_count} characters '
        f'{"· ⚠️ LinkedIn limit is 3 000" if char_count > 3000 else ""}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label" style="margin-top:16px">Copy Post</div>', unsafe_allow_html=True)
    st.text_area(
        label="copy",
        label_visibility="collapsed",
        value=post,
        height=180,
        key="copy_box",
    )
    st.caption("👆 Click inside the box above, select all (Ctrl+A / Cmd+A), then copy.")
