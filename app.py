import json
import logging
import traceback

import streamlit as st

from config import CATALOGUE, POLICY
from core import ProductIndex, answer_query, load_catalogue, load_policy

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("trustcart.app")

st.set_page_config(page_title="TrustCart AI", page_icon="🛒", layout="wide")
st.markdown(
    """
<style>
.stApp {background:#f5f7fb}
.block-container {max-width:1100px;padding-top:2rem}
.hero {padding:1.4rem 1.6rem;border-radius:18px;background:linear-gradient(120deg,#10243a,#176b68);color:white}
.hero h1 {color:white;margin:0}
.hero p {color:#e5f4f2;margin:.35rem 0 0}
</style>
<div class="hero"><h1>🛒 TrustCart AI</h1>
<p>The Conversational Shopping Assistant That Doesn't Lie · R3</p></div>
""",
    unsafe_allow_html=True,
)

EXAMPLES = [
    "What is the return window?",
    "Do you sell iPhones?",
    "65W charger dikhao",
    "blue running shoes price?",
]


@st.cache_resource(show_spinner="Validating official catalogue and policy…")
def resources():
    products = load_catalogue(CATALOGUE)
    policy = load_policy(POLICY)
    return products, ProductIndex(products), policy


def respond(prompt: str) -> None:
    """Answer one shopper message and store it in the chat history."""
    st.session_state.messages.append({"role": "user", "text": prompt})
    try:
        answer = answer_query(prompt, st.session_state.history, products, index, policy)
        label = {"answer": "ANSWER", "clarify": "CLARIFY", "refuse": "REFUSE"}.get(
            answer.action, answer.action.upper()
        )
        meta = (
            f"{label} · {answer.latency_ms:.2f} ms · "
            f"tokens in/out {answer.tokens_in}/{answer.tokens_out}"
        )
        detail = json.dumps(answer.to_dict(), ensure_ascii=False, indent=2, default=str)
        text = answer.answer_text or "I couldn't find a supported answer in the catalogue."
    except Exception:  # keep the demo alive and show the cause
        LOGGER.exception("answer_query failed")
        text = "Sorry, something went wrong while checking the catalogue."
        meta = "ERROR"
        detail = traceback.format_exc()
    st.session_state.history = (st.session_state.history + [prompt])[-8:]
    st.session_state.messages.append(
        {"role": "assistant", "text": text, "meta": meta, "detail": detail}
    )


with st.sidebar:
    st.subheader("Trust controls")
    st.success("Catalogue-grounded answers")
    st.info("Rules + retrieval · no LLM calls")
    st.caption("Synthetic catalogue snapshot: 1 October 2026")
    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

try:
    products, index, policy = resources()
except (FileNotFoundError, ValueError) as exc:
    st.error("Official R3 data could not be loaded.")
    st.code(str(exc))
    st.stop()

st.caption(f"Validated records: {len(products):,} products · {len(policy)} policy subsections")
st.session_state.setdefault("messages", [])
st.session_state.setdefault("history", [])

if not st.session_state.messages:
    st.markdown("### Try these questions")
    columns = st.columns(len(EXAMPLES))
    for column, example in zip(columns, EXAMPLES):
        if column.button(example, use_container_width=True):
            respond(example)
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        if message.get("meta"):
            st.caption(message["meta"])
        if message.get("detail"):
            with st.expander("Evidence and structured response"):
                st.code(message["detail"], language="json")

prompt = st.chat_input("Ask about products, price, stock or store policy…", max_chars=500)
if prompt:
    with st.spinner("Checking catalogue evidence…"):
        respond(prompt)
    st.rerun()
