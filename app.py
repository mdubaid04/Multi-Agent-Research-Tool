import streamlit as st
from datetime import date

from agents import build_searchAgent, build_readerAgent, writter_chain, critics_chain

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔎",
    layout="wide",
)

# ---------------------------------------------------------
# Session state init
# ---------------------------------------------------------
if "state" not in st.session_state:
    st.session_state.state = {}
if "running" not in st.session_state:
    st.session_state.running = False


def reset_state():
    st.session_state.state = {}


# ---------------------------------------------------------
# Pipeline runner (mirrors main_pipeline.run_pipeline, but
# reports progress to the Streamlit UI step by step instead
# of only printing to console)
# ---------------------------------------------------------
def run_pipeline_ui(topic: str):
    today = date.today().strftime("%B-%d-%Y")
    state = {}

    # ---------------- Step 1: Search agent ----------------
    with st.status("🔍 Step 1/4 — Searching the web...", expanded=True) as status:
        try:
            search_agent = build_searchAgent()
            response = search_agent.invoke({
                "messages": [(
                    "user",
                    f"Today's date is {today}.\n"
                    f"Find recent, reliable and detailed information about: {topic}.\n"
                    "Always use the search tool results for current information, "
                    "never rely on your own knowledge for recent events."
                )]
            })
            state["search_result"] = response["messages"][-1].content
            st.write(state["search_result"])
            status.update(label="✅ Step 1/4 — Search complete", state="complete")
        except Exception as e:
            status.update(label="❌ Step 1/4 — Search failed", state="error")
            st.error(f"Search agent failed: {e}")
            return None

    # ---------------- Step 2: Reader agent -----------------
    with st.status("📄 Step 2/4 — Reading top source in depth...", expanded=True) as status:
        try:
            reader_agent = build_readerAgent()
            response = reader_agent.invoke({
                "messages": [(
                    "user",
                    f"Based on the following search result about '{topic}', "
                    f"pick the most relevant URL and scrape it for deeper content.\n\n"
                    f"Search Result:\n{state['search_result'][:800]}"
                    "You must call the scrapeTextFromUrl tool exactly once with the most "
                    "relevant URL. Do not attempt multiple scrapes. If the result is "
                    "insufficient, work with what you have and return the best you can."
                )]
            })
            state["scraped_content"] = response["messages"][-1].content
            st.write(state["scraped_content"])
            status.update(label="✅ Step 2/4 — Reading complete", state="complete")
        except Exception as e:
            status.update(label="❌ Step 2/4 — Reading failed", state="error")
            st.error(f"Reader agent failed: {e}")
            return None

    # ---------------- Step 3: Writer chain -----------------
    with st.status("✍️ Step 3/4 — Writing the report...", expanded=True) as status:
        try:
            research_combined = (
                f"SEARCH RESULT : \n{state['search_result']}\n\n"
                f"DETAILED SCRAPED CONTENT : \n{state['scraped_content']}"
            )
            state["report"] = writter_chain.invoke({
                "topic": topic,
                "research": research_combined,
            })
            st.write(state["report"])
            status.update(label="✅ Step 3/4 — Report drafted", state="complete")
        except Exception as e:
            status.update(label="❌ Step 3/4 — Writing failed", state="error")
            st.error(f"Writer chain failed: {e}")
            return None

    # ---------------- Step 4: Critic chain -----------------
    with st.status("🧐 Step 4/4 — Reviewing the report...", expanded=True) as status:
        try:
            state["feedback"] = critics_chain.invoke({"report": state["report"]})
            st.write(state["feedback"])
            status.update(label="✅ Step 4/4 — Review complete", state="complete")
        except Exception as e:
            status.update(label="❌ Step 4/4 — Review failed", state="error")
            st.error(f"Critic chain failed: {e}")
            return None

    return state


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown(
        "This app runs a 4-agent research pipeline:\n"
        "1. **Search agent** (Mistral) — finds sources\n"
        "2. **Reader agent** (Groq Llama) — scrapes best source\n"
        "3. **Writer chain** (Mistral) — drafts the report\n"
        "4. **Critic chain** (Groq Qwen) — reviews the report"
    )
    st.divider()
    if st.button("🗑️ Clear results"):
        reset_state()
        st.rerun()

# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
st.title("🔎 Multi-Agent Research Assistant")
st.caption("Powered by LangChain · LangGraph · Grok · Mistral · Groq")

topic = st.text_input(
    "What topic do you want researched?",
    placeholder="e.g. Latest developments in solid-state batteries",
)

col1, col2 = st.columns([1, 5])
with col1:
    run_clicked = st.button("🚀 Run Research", type="primary", disabled=st.session_state.running)

if run_clicked:
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        st.session_state.running = True
        result = run_pipeline_ui(topic)
        st.session_state.running = False
        if result:
            st.session_state.state = result

# ---------------------------------------------------------
# Final results (persist across reruns)
# ---------------------------------------------------------
state = st.session_state.state
if state.get("report"):
    st.divider()
    st.subheader("📘 Final Report")

    tab_report, tab_feedback, tab_raw = st.tabs(["Report", "Critic Feedback", "Raw Research"])

    with tab_report:
        st.markdown(state["report"])
        st.download_button(
            "⬇️ Download report (.md)",
            data=state["report"],
            file_name="research_report.md",
            mime="text/markdown",
        )

    with tab_feedback:
        st.markdown(state.get("feedback", "No feedback available."))

    with tab_raw:
        st.markdown("**Search Result**")
        st.write(state.get("search_result", ""))
        st.markdown("**Scraped Content**")
        st.write(state.get("scraped_content", ""))