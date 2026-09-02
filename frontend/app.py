import streamlit as st

from api_client import BackendRequestError, BackendUnavailableError, ask_question, check_health, get_api_base_url

st.set_page_config(page_title="AI StudyMate", page_icon="📚", layout="centered")

INTENT_LABELS = {
    "in_scope": "📚 Answered from verified sources",
    "chitchat": "💬 Chitchat",
    "out_of_scope": "🚫 Out of scope",
}

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("📚 AI StudyMate")
st.caption("A RAG-powered Machine Learning and Deep Learning study assistant")

with st.sidebar:
    st.subheader("Backend status")
    st.code(get_api_base_url(), language=None)
    if st.button("Check connection"):
        try:
            health = check_health()
            st.success(f"Connected — {health.get('chunk_count', '?')} chunks loaded")
            st.caption(f"Model: {health.get('ollama_model', 'unknown')}")
        except BackendUnavailableError:
            st.error("Backend is unreachable. Is it running?")
        except BackendRequestError as exc:
            st.error(f"Backend error ({exc.status_code})")

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources_detail"):
            with st.expander(f"Sources ({len(message['sources_detail'])})"):
                for source in message["sources_detail"]:
                    st.markdown(
                        f"**{source['title']}** — page {source['page']}, "
                        f"Section: {source['section']}  \n"
                        f"[{source['source_url']}]({source['source_url']})"
                    )
        if message["role"] == "assistant" and message.get("intent"):
            st.caption(INTENT_LABELS.get(message["intent"], message["intent"]))

question = st.chat_input("Ask a Machine Learning or Deep Learning question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving sources and generating an answer..."):
            try:
                result = ask_question(question)
                answer = result["answer"]
                sources_detail = result.get("sources_detail", [])
                intent = result.get("intent", "in_scope")

                st.markdown(answer)
                if sources_detail:
                    with st.expander(f"Sources ({len(sources_detail)})"):
                        for source in sources_detail:
                            st.markdown(
                                f"**{source['title']}** — page {source['page']}, "
                                f"Section: {source['section']}  \n"
                                f"[{source['source_url']}]({source['source_url']})"
                            )
                st.caption(INTENT_LABELS.get(intent, intent))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources_detail": sources_detail,
                    "intent": intent,
                })

            except BackendUnavailableError:
                error_text = (
                    "Could not reach the AI StudyMate backend. "
                    "Make sure it's running (`uvicorn app.main:app`) and that "
                    f"API_BASE_URL ({get_api_base_url()}) is correct."
                )
                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text})

            except BackendRequestError as exc:
                error_text = f"The backend returned an error ({exc.status_code}). Please try rephrasing your question."
                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text})
