import streamlit as st
import requests

BACKEND = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Conversational Q&A", layout="wide")
st.title("RAG Conversational PDF Q&A")

st.markdown("""
Upload a PDF.\n
Start a chat session and ask follow-up questions.
""")

# ========== SIDEBAR ========== #
with st.sidebar:
    st.header("📄 Document Ingestion")

    uploaded = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False)
    if uploaded:
        if st.button("Ingest PDF"):
            with st.spinner("Uploading and ingesting..."):
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                resp = requests.post(f"{BACKEND}/upload_pdf", files=files, timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Ingested doc_id: {data['doc_id']}")
                    st.session_state["doc_id"] = data["doc_id"]
                else:
                    st.error(f"Upload failed: {resp.status_code} {resp.text}")

    doc_id = st.text_input("doc_id (from ingestion)", value=st.session_state.get("doc_id", ""))

    if st.button("Start new chat session"):
        r = requests.post(f"{BACKEND}/chat/start")
        if r.ok:
            sid = r.json().get("session_id")
            st.session_state["session_id"] = sid
            st.success(f"New session started: {sid}")
        else:
            st.error("Failed to start session")

    if "session_id" in st.session_state and st.session_state["session_id"]:
        st.info(f"Using session: {st.session_state['session_id']}")

    st.markdown("---")
    st.subheader("⚙️ Retrieval Settings")
    top_k = st.slider("Top-k retrieved chunks/rows", min_value=1, max_value=12, value=6)

# ========== MAIN PAGE (Chat UI) ========== #
st.subheader("💬 Chat")
input_col, button_col = st.columns([8, 1])
with input_col:
    user_question = st.text_input("Your question", key="user_question")
with button_col:
    send = st.button("Send")

if send and user_question:
    if not doc_id:
        st.error("Please provide a doc_id (ingest a PDF first).")
    else:
        payload = {
            "doc_id": doc_id,
            "session_id": st.session_state.get("session_id"),
            "question": user_question,
            "top_k": top_k
        }
        with st.spinner("Thinking..."):
            resp = requests.post(f"{BACKEND}/chat", json=payload, timeout=120)

        if resp.ok:
            out = resp.json()
            # store session id
            sid = out.get("session_id")
            if sid:
                st.session_state["session_id"] = sid
            # display chat history
            history = out.get("chat_history", [])
            for u, a in history:
                st.markdown(f"**You:** {u}")
                st.markdown(f"**Assistant:** {a}")
            # show sources for latest turn
            st.subheader("📚 Sources for latest answer")
            for s in out.get("sources", []):
                label = f"Page: {s.get('page')} • Type: {s.get('type')} • Source: {s.get('source')}"
                with st.expander(label):
                    st.text(s.get("snippet"))
        else:
            st.error(f"Chat error: {resp.status_code} {resp.text}")

# Button to for session history
if st.button("Show current session history"):
    sid = st.session_state.get("session_id")
    if not sid:
        st.warning("No session started.")
    else:
        r = requests.get(f"{BACKEND}/chat/history/{sid}")
        if r.ok:
            data = r.json()
            st.write(data)
        else:
            st.error("Could not fetch history.")
